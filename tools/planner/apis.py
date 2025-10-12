import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import planner_agent_prompt, cot_planner_agent_prompt, react_planner_agent_prompt,reflect_prompt,react_reflect_planner_agent_prompt, REFLECTION_HEADER
from agents.server_llm import serverLLM

import dspy


class Planner:
    def __init__(self,
                 # args,
                 agent_prompt: PromptTemplate = planner_agent_prompt,
                 model_name: str = "gemma-3-27b-it",    # 'qwen2.5:7b'
                 ) -> None:

        self.agent_prompt = agent_prompt
        self.scratchpad: str = ''
        self.model_name = model_name
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
        prompt = self._build_agent_prompt(text, query)

        if log_file:
            log_file.write('\n---------------Planner\n' + prompt)
        try:
            response = self.llm(messages=[{"role": "user", "content": prompt}])
            return response
        except Exception as e:
            return f"LLM Error: {e}"

    def _build_agent_prompt(self, text, query) -> str:
        return self.agent_prompt.format(
            text=text,
            query=query)
    