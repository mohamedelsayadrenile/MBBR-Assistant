from core.config import Settings
from services.tts.interface import TTSProvider
from services.tts.providers.voicetut import VoiceTutTTSProvider


def create_tts_provider(settings: Settings) -> TTSProvider:
    provider = settings.tts_provider.lower().strip()
    if provider == "voicetut":
        return VoiceTutTTSProvider(settings)
    raise ValueError(f"Unsupported TTS provider: {settings.tts_provider}")
