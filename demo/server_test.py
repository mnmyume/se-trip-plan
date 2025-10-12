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
        return r.json()["choices"][0]["message"]["content"]

lm = MyLocalLM(
    api_url="https://django.cair.mun.ca/llm/v1/chat/completions",
    api_key="ADAjs78ehDSS87hs3edcf4edr5"
)
dspy.configure(lm=lm)
print(lm(messages=[{"role": "user", "content": "Hello!"}]))
