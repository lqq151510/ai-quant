"""基础设施层：本地 Qwen3.5-9B (LM Studio/Ollama) 与 DeepSeek API 双模型路由客户端.

职责边界：仅负责 HTTP / OpenAI 兼容接口通信与 JSON 反序列化，不包含任何金融领域业务规则。
"""

import json
import os
from typing import Any, Dict, Optional
import requests


class DualModelLLMClient:
    """支持本地 Qwen3.5-9B (优先用于高频新闻清洗) 与 DeepSeek (用于深度推理裁决) 的客户端."""

    def __init__(
        self,
        local_base_url: Optional[str] = None,
        local_model: str = "qwen3.5-9b",
        deepseek_base_url: str = "https://api.deepseek.com/v1",
        deepseek_model: str = "deepseek-chat",
        timeout: float = 8.0,
    ) -> None:
        self.local_base_url = local_base_url or os.environ.get(
            "LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1"
        )
        self.local_model = local_model
        self.deepseek_base_url = deepseek_base_url
        self.deepseek_model = deepseek_model
        self.deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        self.timeout = timeout

    def is_local_online(self) -> bool:
        """探测本地 Qwen 模型服务是否在线."""
        try:
            resp = requests.get(f"{self.local_base_url}/models", timeout=1.2)
            return resp.status_code == 200
        except Exception:
            return False

    def is_deepseek_configured(self) -> bool:
        """检查是否配置了 DeepSeek API 密钥."""
        return bool(self.deepseek_api_key.strip())

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        prefer_tier: str = "local_fast",
    ) -> Optional[Dict[str, Any]]:
        """发起结构化 JSON 推理请求.

        Args:
            system_prompt: 领域层传入的系统角色提示词
            user_prompt: 领域层组装的结构化上下文
            prefer_tier: 'local_fast' (优先本地 Qwen3.5-9B) 或 'deepseek_reason' (优先 DeepSeek)
        """
        endpoints = []
        if prefer_tier == "deepseek_reason" and self.is_deepseek_configured():
            endpoints.append((self.deepseek_base_url, self.deepseek_model, self.deepseek_api_key))
        if self.is_local_online():
            endpoints.append((self.local_base_url, self.local_model, "lm-studio"))
        if prefer_tier == "local_fast" and self.is_deepseek_configured():
            endpoints.append((self.deepseek_base_url, self.deepseek_model, self.deepseek_api_key))

        for base_url, model_name, api_key in endpoints:
            try:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                }
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.1,
                }
                resp = requests.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 200:
                    content = resp.json()["choices"][0]["message"]["content"]
                    return self._extract_json(content)
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        cleaned = text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
        try:
            return json.loads(cleaned)
        except Exception:
            return None
