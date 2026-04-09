import asyncio
import base64
import json
import time
from enum import Enum
from dataclasses import dataclass, field

import cv2
import numpy as np
import pyaudio
import websockets

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------- SETTINGS ----------------
class Settings(BaseSettings):
    GEMINI_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()

MODEL_NAME = "gemini-3.1-flash-live-preview"
WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={settings.GEMINI_API_KEY}"
)

FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK = 1024


# ---------------- MODE ----------------
class Mode(str, Enum):
    FAST = "FAST"
    DEEP = "DEEP"


# ---------------- STATE ----------------
@dataclass
class AppState:
    mode: Mode = Mode.FAST
    playback_q: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=200))
    stop: asyncio.Event = field(default_factory=asyncio.Event)

    last_user_speech: float = 0
    last_model_output: float = 0
    interrupted_until: float = 0

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------- CORE ----------------
async def connect():
    state = AppState()

    async with websockets.connect(WS_URL) as ws:
        print("Connected")

        await ws.send(json.dumps({
            "setup": {
                "model": f"models/{MODEL_NAME}",
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "systemInstruction": {"parts": [{"text": "You are Eko."}]},
            }
        }))

        # wait setup
        while True:
            msg = json.loads(await ws.recv())
            if "setupComplete" in msg:
                break

        # init audio + cam
        p = pyaudio.PyAudio()
        mic = p.open(format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=CHUNK)
        speaker = p.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True)

        cam = cv2.VideoCapture(0)

        # ---------- HELPERS ----------
        async def send_text(text):
            await ws.send(json.dumps({"realtimeInput": {"text": text}}))

        async def interrupt_model():
            await ws.send(json.dumps({"clientContent": {"turnComplete": True}}))

        async def clear_queue():
            while not state.playback_q.empty():
                try:
                    state.playback_q.get_nowait()
                except:
                    break

        # ---------- MODE SWITCH ----------
        async def handle_mode(text):
            t = text.lower().strip()

            if t == "deep mode":
                new_mode = Mode.DEEP
            elif t == "fast mode":
                new_mode = Mode.FAST
            else:
                return

            async with state.lock:
                if state.mode == new_mode:
                    return
                state.mode = new_mode

            print("Switching to", new_mode)

            await interrupt_model()
            await clear_queue()

            await send_text(f"Switched to {new_mode.value} mode. Respond accordingly.")

        # ---------- TASKS ----------
        async def send_audio():
            while not state.stop.is_set():
                data = await asyncio.to_thread(mic.read, CHUNK, exception_on_overflow=False)
                await ws.send(json.dumps({
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(data).decode(),
                            "mimeType": f"audio/pcm;rate={INPUT_RATE}"
                        }
                    }
                }))

        async def send_frames():
            while not state.stop.is_set():
                ret, frame = await asyncio.to_thread(cam.read)
                if not ret:
                    continue

                _, buf = cv2.imencode(".jpg", frame)

                await ws.send(json.dumps({
                    "realtimeInput": {
                        "video": {
                            "data": base64.b64encode(buf.tobytes()).decode(),
                            "mimeType": "image/jpeg"
                        }
                    }
                }))

                await asyncio.sleep(2 if state.mode == Mode.FAST else 3)

        async def receive():
            async for raw in ws:
                msg = json.loads(raw)

                if "serverContent" not in msg:
                    continue

                sc = msg["serverContent"]

                # interrupt
                if sc.get("interrupted"):
                    state.interrupted_until = time.monotonic() + 2
                    await clear_queue()

                # user speech
                if "inputTranscription" in sc:
                    text = sc["inputTranscription"].get("text", "")
                    if text:
                        print("USER:", text)
                        await handle_mode(text)

                # model audio
                mt = sc.get("modelTurn")
                if mt:
                    for p in mt.get("parts", []):
                        data = p.get("inlineData", {}).get("data")
                        if data:
                            if time.monotonic() < state.interrupted_until:
                                continue

                            audio = base64.b64decode(data)
                            try:
                                state.playback_q.put_nowait(audio)
                            except:
                                await clear_queue()

        async def playback():
            while not state.stop.is_set():
                audio = await state.playback_q.get()
                arr = np.frombuffer(audio, dtype=np.int16)
                await asyncio.to_thread(speaker.write, arr.tobytes())

        # ---------- RUN ----------
        tasks = [
            send_audio(),
            send_frames(),
            receive(),
            playback()
        ]

        await asyncio.gather(*tasks, return_exceptions=True)


# ---------------- MAIN ----------------
async def main():
    while True:
        try:
            await connect()
        except Exception as e:
            print("Restarting...", e)
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())