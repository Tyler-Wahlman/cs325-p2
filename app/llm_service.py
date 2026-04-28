import os

import ollama

from interfaces import LLMServiceInterface


class OllamaService(LLMServiceInterface):
    """
    Concrete implementation of LLMServiceInterface using a local Ollama server.

    Parameters
    ----------
    model : str
        The Ollama model name to use (e.g. 'llama3').
    host : str
        URL of the Ollama server. Defaults to the OLLAMA_HOST environment
        variable, falling back to http://host.docker.internal:11434.
    """

    def __init__(self, model: str = "llama3", host: str = None):
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
        self._client = ollama.Client(host=self.host)

    def generate(self, prompt: str) -> str:
        """Send a prompt to Ollama and return the text response."""
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
