from pydantic import BaseModel


class VoicesResponse(BaseModel):
    voices: list[str]
    default: str
