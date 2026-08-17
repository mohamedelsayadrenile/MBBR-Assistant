from core.config import Settings
from services.asr.interface import ASRProvider
from services.asr.providers.cohere import CohereASRProvider


def create_asr_provider(settings: Settings) -> ASRProvider:
    provider = settings.asr_provider.lower().strip()
    if provider == "cohere":
        return CohereASRProvider(settings)
    raise ValueError(f"Unsupported ASR provider: {settings.asr_provider}")
