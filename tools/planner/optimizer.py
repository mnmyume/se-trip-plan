import dspy
from datasets import load_dataset

from tools.planner.apis import Planner, IndieAccomPlanner
from evaluation.commonsense_constraint import evaluation as commonsense_eval
from evaluation.hard_constraint import evaluation as hard_eval


lm = dspy.LM(f"ollama_chat/qwen3:1.7b", api_base="http://localhost:11434", api_key="")
dspy.configure(lm=lm)


trainset  = load_dataset('osunlp/TravelPlanner','train')['train']
valset = load_dataset('osunlp/TravelPlanner','validation')['validation']


bot = Planner()


# ---- Optimizer -------------------------------------------------------------
optimizer = dspy.MIPROv2(metric=travel_binary_metric, auto="light", verbose=True)


def main():

    print("Compiling optimised bot")
    optimized_bot = optimizer.compile(bot, trainset=trainset, valset=trainset)

    optimized_bot.save(f"optimized.json")


if __name__ == "__main__":
    main()

