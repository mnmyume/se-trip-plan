import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt, transportation_agent_prompt, attraction_agent_prompt, accommodation_agent_prompt, restaurant_agent_prompt, combination_agent_prompt
from agents.server_llm import serverLLM

import dspy


class TransportationPlan(dspy.Signature):
    user_request = dspy.InputField()
    transportation = dspy.OutputField()

class AttractionPlan(dspy.Signature):
    user_request = dspy.InputField()
    transportation = dspy.InputField()
    attraction = dspy.OutputField()

class AccommodationPlan(dspy.Signature):
    user_request = dspy.InputField()
    transportation = dspy.InputField()
    accommodation = dspy.OutputField()

class RestaurantPlan(dspy.Signature):
    user_request = dspy.InputField()
    transportation = dspy.InputField()
    restaurant = dspy.OutputField()

class CombinationPlan(dspy.Signature):
    user_request = dspy.InputField()
    transportation = dspy.InputField()
    attraction = dspy.InputField()
    accommodation = dspy.InputField()
    restaurant = dspy.InputField()
    final_plan = dspy.OutputField()


class Planner:
    def __init__(self,
                 # args,
                 planner_prompt: PromptTemplate = planner_agent_prompt,
                 transportation_prompt: PromptTemplate = transportation_agent_prompt,
                 attraction_prompt: PromptTemplate = attraction_agent_prompt,
                 accommodation_prompt: PromptTemplate = accommodation_agent_prompt,
                 restaurant_prompt: PromptTemplate = restaurant_agent_prompt,
                 combination_prompt: PromptTemplate = combination_agent_prompt,
                 model_name: str = "gemma-3-27b-it",    # 'qwen2.5:7b'
                 node_mode: str = "separate",
                 ) -> None:

        self.planner_prompt = planner_prompt
        self.transportation_prompt = transportation_prompt
        self.attraction_prompt = attraction_prompt
        self.accommodation_prompt = accommodation_prompt
        self.restaurant_prompt = restaurant_prompt
        self.combination_prompt = combination_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
        self.node_mode = node_mode
        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        # self.llm = dspy.LM(f"ollama_chat/{model_name}",
        #                     api_base="http://localhost:11434",
        #                     api_key="")

        self.llm = serverLLM(
            model_name=model_name,
            api_url="https://django.cair.mun.ca/llm/v1/chat/completions",
            api_key="ADAjs78ehDSS87hs3edcf4edr5"
        )
        dspy.configure(lm=self.llm)

        print(f"PlannerAgent {model_name} loaded.")

    # def run(self, text, query, log_file=None) -> str:
    #     if log_file:
    #         log_file.write('\n---------------Planner\n'+self._build_agent_prompt(text, query))
        # print(self._build_agent_prompt(text, query))
        # if len(self.enc.encode(self._build_agent_prompt(text, query))) > 12000:
        #     return 'Max Token Length Exceeded.'
        # else:
        # return self.llm([HumanMessage(content=self._build_agent_prompt(text, query))]).content
    def run(self, text, query, log_file=None) -> str:
        planner_prompt = self._build_agent_prompt(self.planner_prompt, text, query)
        trans_prompt = self._build_agent_prompt(self.transportation_prompt, text, query)
        attra_prompt = self._build_agent_prompt(self.attraction_prompt, text, query)
        accom_prompt = self._build_agent_prompt(self.accommodation_prompt, text, query)
        resta_prompt = self._build_agent_prompt(self.restaurant_prompt, text, query)
        combi_prompt = self._build_agent_prompt(self.combination_prompt, text, query)


        # if log_file:
        #     log_file.write('\n---------------Planner\n' + prompt)
        try:
            trans_resp = self.llm(messages=[{"role": "user", "content": trans_prompt}])
            route_prompt = "Route already generated is:" + trans_resp


            if self.node_mode == "base":
                response = self.llm(messages=[{"role": "user", "content": planner_prompt}])
                return response

            if self.node_mode == 'separate':
                attra_resp = self.llm(messages=[{"role": "user", "content": f"{route_prompt} \n {attra_prompt}"}])
                accom_resp = self.llm(messages=[{"role": "user", "content": f"{route_prompt} \n {accom_prompt}"}])
                resta_resp = self.llm(messages=[{"role": "user", "content": f"{route_prompt} \n {resta_prompt}"}])

            elif self.node_mode == "merge_attra_accom":
                merged_prompt = f"{route_prompt}\n{attra_prompt}\n{accom_prompt}"
                merged_resp = self.llm(messages=[{"role": "user", "content": merged_prompt}])
                attra_resp = accom_resp = merged_resp
                resta_resp = self.llm(messages=[{"role": "user", "content": f"{route_prompt}\n{resta_prompt}"}])

            elif self.node_mode == "merge_attra_resta":
                merged_prompt = f"{route_prompt}\n{attra_prompt}\n{resta_prompt}"
                merged_resp = self.llm(messages=[{"role": "user", "content": merged_prompt}])
                attra_resp = resta_resp = merged_resp
                accom_resp = self.llm(messages=[{"role": "user", "content": f"{route_prompt}\n{accom_prompt}"}])

            elif self.node_mode == "merge_accom_resta":
                merged_prompt = f"{route_prompt}\n{accom_prompt}\n{resta_prompt}"
                merged_resp = self.llm(messages=[{"role": "user", "content": merged_prompt}])
                accom_resp = resta_resp = merged_resp
                attra_resp = self.llm(messages=[{"role": "user", "content": f"{route_prompt}\n{attra_prompt}"}])

            elif self.node_mode == "merge_all":
                merged_prompt = f"{route_prompt}\n{attra_prompt}\n{accom_prompt}\n{resta_prompt}"
                merged_resp = self.llm(messages=[{"role": "user", "content": merged_prompt}])
                attra_resp = accom_resp = resta_resp = merged_resp

            final_prompt = (f"The separated parts of the trip plan is: "
                       f"1. Transportation plan: {trans_resp}\n"
                       f"2. Attraction plan: {attra_resp}\n"
                       f"3. Accommodation plan: {accom_resp}\n"
                       f"4. Restaurant plan: {resta_resp}\n"
                       f"{combi_prompt}")
            final_resp = self.llm(messages=[{"role": "user", "content": final_prompt}])
            return final_resp

        except Exception as e:
            return f"LLM Error: {e}"


    def _build_agent_prompt(self, agent_prompt, text, query) -> str:
        return agent_prompt.format(
            text=text,
            query=query)
    