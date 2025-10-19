import dspy
import requests
import json, os, datetime

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

        usage_data = r.json().get("usage", {})
        usage_data["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "usage", "token.json")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # 如果文件不存在 -> 创建并写入列表
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([usage_data], f, indent=2, ensure_ascii=False)

        # 如果文件存在 -> 读取旧内容，追加，再写回
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = [data]
            data.append(usage_data)

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)


        return r.json()["choices"][0]["message"]["content"]