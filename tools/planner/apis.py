import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from langchain.prompts import PromptTemplate
from agents.prompts import JSONPlannerSignature, PlannerSignature, AccommodationSignature

import dspy


class Planner(dspy.Module):
    def __init__(self,
                 # args,
                 model_name: str = "qwen3:1.7b",
                 node_mode: str = "base",
                 ) -> None:

        self.scratchpad: str = ''
        self.model_name = model_name
        self.node_mode = node_mode
        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.llm = dspy.LM(f"ollama_chat/{model_name}", api_base="http://localhost:11434", api_key="")

        dspy.configure(lm=self.llm)
        print(f"PlannerAgent {model_name} loaded.")

        self.predict = dspy.Predict(JSONPlannerSignature)

    def forward(self, text, query, log_file=None) -> str:

        try:
            response = self.predict(text=text, query=query)
            return response

        except Exception as e:
            return f"LLM Error: {e}"


class IndieAccomPlanner(dspy.Module):
    def __init__(self,
                 # args,
                 model_name: str = "gemma-3-27b-it",
                 node_mode: str = "base",
                 ) -> None:

        self.scratchpad: str = ''
        self.model_name = model_name
        self.node_mode = node_mode
        # self.enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

        self.llm = dspy.LM(f"ollama_chat/{model_name}", api_base="http://localhost:11434", api_key="")


        dspy.configure(lm=self.llm)
        print(f"PlannerAgent {model_name} loaded.")

        self.predict = dspy.Predict(PlannerSignature)
        self.accommodation = dspy.Predict(AccommodationSignature)

    def forward(self, text, query, log_file=None) -> str:

        try:
            accom = self.accommodation(text=text, query=query)
            response = self.predict(text=text, query=query, accommodation=accom)
            return response

        except Exception as e:
            return f"LLM Error: {e}"
