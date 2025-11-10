import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import PlannerSignature
from agents.server_llm import serverLLM

import dspy


class Planner(dspy.Module):
    def __init__(self,
                 # args,
                 model_name: str = "gemma-3-27b-it",
                 node_mode: str = "base",
                 ) -> None:

        self.scratchpad: str = ''
        self.model_name = model_name
        self.node_mode = node_mode
        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.llm = serverLLM(
            model_name=model_name,
            api_url="https://django.cair.mun.ca/llm/v1/chat/completions",
            api_key="ADAjs78ehDSS87hs3edcf4edr5"
        )
        # response = self.llm(messages=[{"role": "user", "content": "how are you?"}])
        dspy.configure(lm=self.llm)
        print(f"PlannerAgent {model_name} loaded.")

        self.predict = dspy.Predict(PlannerSignature)

    def forward(self, text, query, log_file=None) -> str:
        # if log_file:
        #     log_file.write('\n---------------Planner\n' + prompt)
        try:

            if self.node_mode == "base":
                response = self.predict(text=text, query=query)
                print(response)
                return response

            if self.node_mode == "tuning":
                return 'tuning'

        except Exception as e:
            return f"LLM Error: {e}"
    