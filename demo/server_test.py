import requests
import dspy

class MyLocalLM(dspy.LM):
    def __init__(self, api_url, api_key):
        self.url = api_url
        self.api_key = api_key

    def __call__(self, messages, temperature=0.7, max_tokens=128):
        payload = {
            "model": "gemma-3-27b-it",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        r = requests.post(self.url, headers=headers, json=payload, verify=False, timeout=10)
        return {"completion": r.json()["choices"][0]["message"]["content"]}

lm = MyLocalLM(
    api_url="https://django.cair.mun.ca/llm/v1/chat/completions",
    api_key="ADAjs78ehDSS87hs3edcf4edr5"
)
dspy.configure(lm=lm)

print(lm(messages=[{"role": "user", "content": "Hello!"}]))





# url = "https://django.cair.mun.ca/llm/v1/chat/completions"
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": "Bearer ADAjs78ehDSS87hs3edcf4edr5"
# }
# data = {
#     "model": "gemma-3-27b-it",
#     "messages": [{"role": "user", "content": "Hello, how are you?"}],
#     "temperature": 0.7,
#     "max_tokens": 100
# }
#
# try:
#     response = requests.post(url, headers=headers, json=data, timeout=10, verify=False)
#     print("Status:", response.status_code)
#     print("Body:", response.text[:500])
# except Exception as e:
#     print("Error:", e)
