from core.config import Settings
from services.llm.interface import LLMProvider
from services.llm.providers.openai_compatible import OpenAICompatibleLLMProvider


def create_llm_provider(settings: Settings) -> LLMProvider:
    provider = settings.llm_provider.lower().strip()
    if provider == "openai_compatible":
        return OpenAICompatibleLLMProvider(settings)
    raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
