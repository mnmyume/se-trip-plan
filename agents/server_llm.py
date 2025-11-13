import requests
import dspy
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import json


class serverLLM(dspy.BaseLM):
    def __init__(self, base_url, model, api_key, temperature=1, max_tokens=128000):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.kwargs = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _post(self, messages, temperature=None, max_tokens=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature or self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            verify=False
        )
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"].strip()
        return [{"travel_plan": "",
                 "text":content
                }]


    def __call__(self, *args, **kwargs):

        if len(args) == 1 and isinstance(args[0], str):
            messages = [{"role": "user", "content": args[0]}]

        elif "messages" in kwargs:
            messages = kwargs["messages"]

        else:
            raise ValueError(f"Unexpected __call__ args: {args}, kwargs: {kwargs}")

        return self._post(messages)

