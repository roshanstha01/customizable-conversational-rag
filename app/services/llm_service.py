#Ollama

from typing import List, Dict

import httpx
import ollama

from app.errors import unavailable_on

OLLAMA_ERRORS = (ConnectionError, httpx.ConnectError, httpx.TimeoutException)


class OllamaService:
    def __init__(self, host: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        with unavailable_on(OLLAMA_ERRORS, "Ollama"):
            response = self.client.chat(
                model=self.model_name,
                messages=messages,
            )
        return response["message"]["content"]

    def list_models(self) -> List[str]:
        with unavailable_on(OLLAMA_ERRORS, "Ollama"):
            response = self.client.list()
        return [model.model for model in response.models]
