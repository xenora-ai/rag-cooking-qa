# src/llm/llm_client.py
import requests


class LLMClient:
    def __init__(
        self,
        model_name="llama-3.3-70b-versatile",
        api_key=None
    ):
        self.model_name = model_name
        self.api_key = api_key

        self.url = "https://api.groq.com/openai/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
        }

        response = requests.post(self.url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]
