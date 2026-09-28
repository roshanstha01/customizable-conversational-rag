#Ollama

import logging
from typing import Dict, List, Optional

import httpx
import ollama

from app.errors import ServiceUnavailableError, unavailable_on

logger = logging.getLogger(__name__)

OLLAMA_ERRORS = (ConnectionError, httpx.ConnectError, httpx.TimeoutException)


class OllamaService:
    def __init__(self, host: str, model_name: str) -> None:
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> str:
        options = {"temperature": temperature} if temperature is not None else None
        try:
            with unavailable_on(OLLAMA_ERRORS, "Ollama"):
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                )
        except ollama.ResponseError as error:
            logger.warning("Ollama returned an error: %s", error)
            if error.status_code == 404:
                message = (
                    f"LLM model '{self.model_name}' is not available in Ollama yet. "
                    f"Pull it with 'ollama pull {self.model_name}' and try again."
                )
            else:
                message = "The LLM service (Ollama) could not generate a response. Please try again later."
            raise ServiceUnavailableError("Ollama", message) from error
        return response["message"]["content"]

    def list_models(self) -> List[str]:
        with unavailable_on(OLLAMA_ERRORS, "Ollama"):
            response = self.client.list()
        return [model.model for model in response.models]
