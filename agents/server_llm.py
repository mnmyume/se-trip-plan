import dspy
import requests
import json, os, datetime
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
        # 添加 history 属性，用于 dspy.inspect_history() 调试
        self.history = []

    # 关键修复：将 'messages' 参数名改为 'prompt'
    # dspy.Module (如 dspy.Predict) 会传入一个'prompt'字符串，而不是'messages'列表
    def __call__(self, prompt, **kwargs):

        # 将 DSPy 传入的 prompt 记录到 history 中，便于调试
        self.history.append({'prompt': prompt, 'response': None, 'kwargs': kwargs})

        params = {**self.kwargs, **kwargs}

        # 关键修复：在这里将 'prompt' 字符串包装成 API 需要的 'messages' 列表
        # 这就是你 curl 命令中的 -d '{"messages": [{"role": "user", "content": "..."}]}'
        messages_payload = [{"role": "user", "content": prompt}]

        payload = {
            "model": self.model_name,
            "messages": messages_payload,  # <--- 使用新创建的列表
            "temperature": params.get("temperature", 0.7),
            "max_tokens": params.get("max_tokens", 12000)
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            r = requests.post(self.url, headers=headers, json=payload, verify=False, timeout=100)
            r.raise_for_status()  # 自动检查 4xx/5xx 错误

            response_json = r.json()

            # 检查返回的json中是否有 'choices'
            if "choices" not in response_json or not response_json["choices"]:
                raise ValueError(f"LLM 响应中缺少 'choices' 字段或 'choices' 为空。响应: {response_json}")

            # 提取返回的文本内容
            response_content = response_json["choices"][0]["message"]["content"]

            # --- 你的日志记录代码 (完全保留) ---
            usage_data = response_json.get("usage", {})
            usage_data["time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            base_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(base_dir, "usage", "token.json")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            if not os.path.exists(file_path):
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump([usage_data], f, indent=2, ensure_ascii=False)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if not isinstance(data, list): data = [data]
                data.append(usage_data)
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            # --- 日志记录结束 ---

            # 将成功的响应存入 history
            self.history[-1]['response'] = response_content

            # DSPy 期望 __call__ 只返回一个字符串
            return response_content

        except Exception as e:
            # 捕获并打印更详细的错误
            error_msg = f"LLM 调用失败: {e}"
            if 'r' in locals():
                error_msg += f"\n原始服务器响应: {r.text}"
            print(error_msg)

            # 将错误信息存入 history
            self.history[-1]['response'] = error_msg

            # 抛出一个 DSPy 能理解的异常
            raise dspy.dsp.LMFormatError(f"Failed to parse LM output: {error_msg}")

    # (可选但强烈推荐) 添加这个方法来帮助调试
    def inspect_history(self, n=1):
        """打印最后n次 LLM 调用。"""
        if not self.history:
            print("No history to inspect.")
            return

        for i, entry in enumerate(self.history[-n:]):
            print(f"--- History Entry {-n + i} ---")
            print(f"Prompt (DSPy 传入的字符串):\n{entry['prompt']}")
            print(f"\nResponse (LLM 返回的字符串):\n{entry['response']}")
            print(f"\nKwargs: {entry['kwargs']}")
            print("-" * 30)