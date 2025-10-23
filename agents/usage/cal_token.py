import json


with open("token_merge_all.json", "r", encoding="utf-8") as f:
    data = json.load(f)


prompt_sum = sum(item.get("prompt_tokens", 0) for item in data)
completion_sum = sum(item.get("completion_tokens", 0) for item in data)
total_sum = sum(item.get("total_tokens", 0) for item in data)

print(f"Prompt tokens total: {prompt_sum}")
print(f"Completion tokens total: {completion_sum}")
print(f"All tokens total: {total_sum}")
