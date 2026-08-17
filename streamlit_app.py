"""Manual tester for the MBBR voice assistant.

Talks to the running FastAPI app over HTTP, so it exercises the real pipeline
(ASR, agent, MBBR APIs, TTS) rather than importing anything from src/.

    uv run uvicorn main:app --reload          # terminal 1
    uv run streamlit run streamlit_app.py     # terminal 2
"""

import base64
import os
import uuid

import httpx
import streamlit as st

DEFAULT_API_BASE_URL = os.getenv("CHAT_API_BASE_URL", "http://localhost:8000")
REQUEST_TIMEOUT_SECONDS = 300.0

st.set_page_config(page_title="MBBR Assistant tester", page_icon="🎙️")


def new_conversation_id() -> str:
    return f"ui-{uuid.uuid4().hex[:8]}"


if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = new_conversation_id()
if "turns" not in st.session_state:
    st.session_state.turns = []


with st.sidebar:
    st.header("Settings")
    api_base_url = st.text_input("API base URL", value=DEFAULT_API_BASE_URL)
    jwt = st.text_input(
        "JWT",
        type="password",
        help="Forwarded to the MBBR APIs. Never sent to the LLM or stored in Redis.",
    )

    st.caption(f"Conversation: `{st.session_state.conversation_id}`")
    if st.button("New conversation", use_container_width=True):
        st.session_state.conversation_id = new_conversation_id()
        st.session_state.turns = []
        st.rerun()

    if st.button("Check health", use_container_width=True):
        try:
            health = httpx.get(f"{api_base_url}/health", timeout=5)
            st.success(health.json()) if health.status_code == 200 else st.error(
                f"{health.status_code}: {health.text}"
            )
        except httpx.HTTPError as exc:
            st.error(f"Cannot reach the API: {type(exc).__name__}")


st.title("🎙️ MBBR Assistant")
st.caption(
    "Ask a question in Egyptian Arabic — for example "
    "«عايز درجة حرارة الماية دلوقتي» — by voice or by typing, then pick a device "
    "when asked. A spoken question comes back spoken; a typed one comes back as text."
)

audio = st.audio_input("Record a question")
uploaded = st.file_uploader("…or upload a WAV", type=["wav"])
audio_bytes = (audio or uploaded).getvalue() if (audio or uploaded) else None

send = st.button(
    "Send", type="primary", disabled=audio_bytes is None or not jwt, use_container_width=True
)
if not jwt:
    st.info("Enter a JWT in the sidebar to send.")


def send_turn(*, audio_payload: bytes | None = None, text: str | None = None) -> None:
    """POST one turn — voice or text, never both — and append it to the history."""
    data = {"conversation_id": st.session_state.conversation_id, "jwt": jwt}
    files = None
    if text is None:
        files = {"audio": ("voice.wav", audio_payload, "audio/wav")}
    else:
        data["text"] = text

    try:
        response = httpx.post(
            f"{api_base_url}/api/v1/chat",
            data=data,
            files=files,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        st.error(f"Request failed: {type(exc).__name__}")
        return

    if response.status_code != 200:
        detail = response.json().get("detail", response.text)
        st.error(f"{response.status_code}: {detail}")
        return

    body = response.json()
    reply_audio = body.get("audio_base64")
    st.session_state.turns.append(
        {
            "transcript": body["transcript"],
            "reply": body["reply"],
            "audio": base64.b64decode(reply_audio) if reply_audio else None,
            # Only a voice turn is meant to come back spoken, so only a voice turn
            # can be missing its audio.
            "spoken": text is None,
        }
    )


typed = st.chat_input("…or type a question", disabled=not jwt)

if send and audio_bytes:
    with st.spinner("Transcribing, thinking, speaking…"):
        send_turn(audio_payload=audio_bytes)
elif typed and typed.strip():
    with st.spinner("Thinking…"):
        send_turn(text=typed.strip())

for turn in st.session_state.turns:
    with st.chat_message("user"):
        st.write(turn["transcript"])
    with st.chat_message("assistant"):
        st.write(turn["reply"])
        if turn["audio"]:
            st.audio(turn["audio"], format="audio/wav")
        elif turn["spoken"]:
            st.caption("No audio — speech synthesis failed for this reply.")
