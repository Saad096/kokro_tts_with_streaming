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
from TTS.api import TTS
import soundfile as sf
import tempfile
from concurrent.futures import ThreadPoolExecutor
import librosa
from typing import Union, Tuple

# Load environment variables
load_dotenv()

# Initialize clients
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
tts_client = get_tts_model(model="kokoro")
stt_model = get_stt_model(model="moonshine/base")

# Voice Conversion Setup
vc_tts = TTS(model_name="voice_conversion_models/multilingual/vctk/freevc24", progress_bar=False).to("cuda")
target_wav_path = "speech.wav"  # Replace with your target speaker file
executor = ThreadPoolExecutor(max_workers=2)

# Audio Config
TTS_SAMPLE_RATE = 24000
VC_SAMPLE_RATE = 16000

options = KokoroTTSOptions(
    voice="af_heart",
    speed=1.0,
    lang="en-us"
)

async def convert_audio(tts_audio: np.ndarray, original_sr: int) -> np.ndarray:
    """Convert TTS audio with robust error handling."""
    try:
        if tts_audio is None or len(tts_audio) == 0:
            return np.array([], dtype=np.int16)
        
        loop = asyncio.get_event_loop()
        with tempfile.NamedTemporaryFile(suffix=".wav") as source_temp, \
             tempfile.NamedTemporaryFile(suffix=".wav") as output_temp:

            # Convert to float32 for processing
            audio_data = tts_audio.astype(np.float32)
            
            # Resample if needed
            if original_sr != VC_SAMPLE_RATE:
                audio_data = librosa.resample(
                    audio_data,
                    orig_sr=original_sr,
                    target_sr=VC_SAMPLE_RATE
                )

            # Normalize and save
            peak = np.max(np.abs(audio_data))
            if peak > 0:
                audio_data /= peak
                audio_data *= 0.95  # Add 5% headroom
                
            sf.write(source_temp.name, audio_data, VC_SAMPLE_RATE, subtype='PCM_16')
            
            # Run VC conversion
            await loop.run_in_executor(
                executor,
                lambda: vc_tts.voice_conversion_to_file(
                    source_wav=source_temp.name,
                    target_wav=target_wav_path,
                    file_path=output_temp.name
                )
            )
            
            # Read converted audio
            try:
                converted_audio, _ = sf.read(output_temp.name, dtype='float32')
                return (converted_audio * 32767).clip(-32768, 32767).astype(np.int16)
            except Exception as e:
                print(f"Error reading converted audio: {str(e)}")
                return np.array([], dtype=np.int16)
                
    except Exception as e:
        print(f"Voice conversion failed: {str(e)}")
        return np.array([], dtype=np.int16)

def normalize_audio(audio: Union[np.ndarray, bytes, bytearray, Tuple[int, np.ndarray]]) -> np.ndarray:
    """Normalize audio input to 16-bit PCM format."""
    try:
        # Convert bytearray to bytes first
        if isinstance(audio, bytearray):
            audio = bytes(audio)
            
        # Handle different input types
        if isinstance(audio, tuple):
            _, audio = audio  # Extract from (rate, data) tuple
        if isinstance(audio, (bytes, bytearray)):
            audio = np.frombuffer(audio, dtype=np.int16)
            
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Unsupported audio type: {type(audio)}")
            
        # Convert to float32 for processing
        audio = audio.astype(np.float32)
        
        if len(audio) == 0:
            return np.array([], dtype=np.int16)
        
        # Normalize with headroom
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = (audio / peak) * 0.95  # 5% headroom
        
        # Convert to 16-bit PCM
        return (audio * 32767).clip(-32768, 32767).astype(np.int16).flatten()
        
    except Exception as e:
        print(f"Normalization error: {str(e)}")
        return np.array([], dtype=np.int16)

async def process_tts_segment(segment: str) -> bytearray:
    """Process TTS audio for a text segment."""
    tts_audio_bytes = bytearray()
    
    try:
        async for audio_chunk in tts_client.stream_tts(segment, options=options):
            if audio_chunk is None:
                continue
                
            # Normalize and convert to bytes
            normalized = normalize_audio(audio_chunk)
            if normalized is not None and len(normalized) > 0:
                tts_audio_bytes.extend(normalized.tobytes())
                
    except Exception as e:
        print(f"TTS streaming error: {str(e)}")
        
    return tts_audio_bytes

async def generate_response_streaming(
    audio: Tuple[int, NDArray[np.int16 | np.float32]],
    chatbot: list[dict] | None = None
):
    chatbot = chatbot or []
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in chatbot]
    messages.insert(0, {"role": "system", "content": "You are a basic conversational assistant. Keep responses short, to-the-point, and one line only."})

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
        max_completion_tokens=200,
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
            print("AI:", segment)
            
            tts_audio_bytes = await process_tts_segment(segment)
            
            if tts_audio_bytes:
                try:
                    tts_audio = normalize_audio(tts_audio_bytes)
                    if len(tts_audio) > 0:
                        converted_audio = await convert_audio(tts_audio, TTS_SAMPLE_RATE)
                        chunk_size = 1024 * 2
                        for i in range(0, len(converted_audio), chunk_size):
                            yield (VC_SAMPLE_RATE, converted_audio[i:i+chunk_size])
                except Exception as e:
                    print(f"Audio processing failed: {str(e)}")

    if buffer.strip():
        segment = buffer.strip()
        print("AI:", segment)
        tts_audio_bytes = await process_tts_segment(segment)
        
        if tts_audio_bytes:
            try:
                tts_audio = normalize_audio(tts_audio_bytes)
                if len(tts_audio) > 0:
                    converted_audio = await convert_audio(tts_audio, TTS_SAMPLE_RATE)
                    chunk_size = 1024 * 2
                    for i in range(0, len(converted_audio), chunk_size):
                        yield (VC_SAMPLE_RATE, converted_audio[i:i+chunk_size])
            except Exception as e:
                print(f"Final audio processing failed: {str(e)}")

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
        "title": "LLM Voice Chat with Voice Conversion",
    },
)

# Launch the UI
stream.ui.launch(share=True)