import dspy
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from agents.server_llm import serverLLM

# Import the optimizer
from dspy.teleprompt import MIPROv2

# Initialize the LM
lm = serverLLM(
            base_url="https://django.cair.mun.ca/llm/v1/chat/completions",
            model="model_name",
            api_key="ADAjs78ehDSS87hs3edcf4edr5",
        )
dspy.configure(lm=lm)

# Initialize optimizer
teleprompter = MIPROv2(
    metric=gsm8k_metric,
    auto="medium", # Can choose between light, medium, and heavy optimization runs
)

# Optimize program
print(f"Optimizing program with MIPROv2...")
gsm8k = GSM8K()
optimized_program = teleprompter.compile(
    dspy.ChainOfThought("question -> answer"),
    trainset=gsm8k.train,
)

# Save optimize program for future use
optimized_program.save(f"optimized.json")