from typing import Optional, List, Dict

import requests


class DeepSeekClient:
    """
    DeepSeek 单轮 / 多轮对话客户端
    """

    def __init__(
        self,
        base_url: str = "https://api.deepseek.com"
    ):
        self.api_key = "sk-d66bc1b5da854fe889f72e544c9aa9e8"
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")

        self.base_url = base_url.rstrip("/")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def chat(
        self,
        messages: List[Dict],
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 30
    ) -> str:
        """
        通用对话接口（支持多轮）
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=timeout
            )

            response.raise_for_status()
            data = response.json()

            return data["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            raise RuntimeError("DeepSeek 请求超时")
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP 错误: {response.text}") from e
        except Exception as e:
            raise RuntimeError(f"DeepSeek 请求失败: {e}") from e

    def complete_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        单轮调用（封装 chat）
        """
        messages = []

        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        return self.chat(messages, **kwargs)

