import asyncio
from fastrtc import (
    ReplyOnPause,
    Stream,
    AdditionalOutputs,
    get_stt_model,
    get_tts_model,
    KokoroTTSOptions
)
import numpy as np
from numpy.typing import NDArray
from dotenv import load_dotenv
from groq import AsyncGroq

import os
import time

# Load environment variables
load_dotenv()

# Initialize clients
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
tts_client = get_tts_model(model="kokoro")
stt_model = get_stt_model(model="moonshine/base")

options = KokoroTTSOptions(
    voice="af_heart",
    speed=1.0,
    lang="en-us"
)

async def generate_response_streaming(
    audio: tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None
):
    chatbot = chatbot or []
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in chatbot]

    text = stt_model.stt(audio)
    print("User:", text)

    chatbot.append({"role": "user", "content": text})
    messages.append({"role": "user", "content": text})

    yield AdditionalOutputs(chatbot)

    response_text = ""
    buffer = ""

    stream = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        stream=True,
    )

    async for chunk in stream:

        if hasattr(chunk, "choices"):
            token = chunk.choices[0].delta.content or ""
        else:
            token = str(chunk)
            
        response_text += token  
        buffer += token

        if any(punct in buffer for punct in [".", "!", "?", "\n"]):
            segment = buffer.strip()
            buffer = ""
            print(segment)
            async for audio_chunk in tts_client.stream_tts(segment, options=options):
                yield audio_chunk

    if buffer.strip():
        segment = buffer.strip()
        async for audio_chunk in tts_client.stream_tts(segment, options=options):
            yield audio_chunk

    chatbot.append({"role": "assistant", "content": response_text})


async def response(audio, chatbot=None):
    async for out in generate_response_streaming(audio, chatbot):
        yield out

# Stream setup
stream = Stream(
    handler=ReplyOnPause(response, input_sample_rate=16000),
    modality="audio",
    mode="send-receive",
    ui_args={
        "title": "LLM Voice Chat (Streaming Response 🎙️)",
    },
)

# Launch the UI
stream.ui.launch()

