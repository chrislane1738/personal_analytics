from openai import OpenAI


def whisper(audio_bytes: bytes, api_key: str, mime: str = "audio/ogg") -> str:
    if not audio_bytes:
        raise ValueError("empty audio")
    client = OpenAI(api_key=api_key)
    ext = ".ogg" if "ogg" in mime else ".mp3"
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=(f"voice{ext}", audio_bytes, mime),
    )
    return result.text.strip()
