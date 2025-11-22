import time
import traceback

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
        self.history = []

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

        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                verify=False,
                timeout=30
            )

            # network or HTTP errors
            response.raise_for_status()

            try:
                data = response.json()
            except json.JSONDecodeError:
                print("\n❌ JSON PARSE ERROR — raw response:")
                print(response.text)
                raise

            # API structure error
            if "choices" not in data or len(data["choices"]) == 0:
                print("\n❌ INVALID RESPONSE STRUCTURE:")
                print(data)
                raise ValueError("Missing 'choices' in response")

            content = data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print("\n================ LLM REQUEST ERROR ================")
            print("messages:", messages)
            print("payload:", payload)
            print("ERROR:", str(e))
            print("TRACEBACK:")
            traceback.print_exc()
            print("===================================================\n")

            # Return a safe fallback so DSPy doesn't crash parallel workers
            content = "ERROR"

        # Build DSPy history entry
        prompt = messages[-1]["content"]

        record = {
            "prompt": prompt,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
            ],
            "outputs": content,
            "timestamp": time.time(),
        }

        self.history.append(record)

        # DSPy expects a dict, not string
        return [{
            "travel_plan": "",
            "text": content
        }]

    def __call__(self, *args, **kwargs):

        try:
            if len(args) == 1 and isinstance(args[0], str):
                messages = [{"role": "user", "content": args[0]}]

            elif "messages" in kwargs:
                messages = kwargs["messages"]

            else:
                raise ValueError(f"Unexpected __call__ args: {args}, kwargs: {kwargs}")

            return self._post(messages)

        except Exception as e:
            print("\n================ __call__ ERROR ================")
            print("args:", args)
            print("kwargs:", kwargs)
            print("ERROR:", e)
            traceback.print_exc()
            print("================================================\n")

            return [{
                "travel_plan": "",
                "text": "ERROR"
            }]

