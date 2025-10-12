import dspy
import requests

# verify-False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class serverLLM(dspy.LM):
    def __init__(self, model_name="gemma-3-27b-it", api_url=None, api_key=None, temperature=0.7, max_tokens=12000):
        super().__init__(model=model_name)
        self.model_name = model_name
        self.url = api_url
        self.api_key = api_key

        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }

    def __call__(self, messages, **kwargs):
        params = {**self.kwargs, **kwargs}
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 12000)
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        r = requests.post(self.url, headers=headers, json=payload, verify=False, timeout=100)
        return r.json()["choices"][0]["message"]["content"]