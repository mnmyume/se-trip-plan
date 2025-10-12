import dspy
import requests


class serverLLM(dspy.LM):
    def __init__(self, model_name="gemma-3-27b-it", api_url=None, api_key=None):
        self.model_name = model_name
        self.url = api_url
        self.api_key = api_key

    def __call__(self, messages, temperature=0.7, max_tokens=12000):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        r = requests.post(self.url, headers=headers, json=payload, verify=False, timeout=100)
        return {"completion": r.json()["choices"][0]["message"]["content"]}