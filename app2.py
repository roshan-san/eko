import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np
import pyaudio
import websockets

from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------- SETTINGS ----------------
class Settings(BaseSettings):
    modell: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={settings.modell}"
)

MODEL = "models/gemini-3.1-flash-live-preview"


# ---------------- MODE ----------------
class Mode(str, Enum):
    ACTIVE = "ACTIVE"
    ULTRA = "ULTRA"


# ---------------- STATE (ONLY SOURCE OF TRUTH) ----------------
@dataclass
class AppState:
    mode: Mode = Mode.ACTIVE
    playback_q: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=100))

    interrupted_until: float = 0
    ultra_started_at: float = 0

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------- CORE ----------------
async def run():
    state = AppState()

    async with websockets.connect(WS_URL) as ws:
        print("Connected")

        # Setup
        await ws.send(json.dumps({
            "setup": {
                "model": MODEL,
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "systemInstruction": {
                    "parts": [{"text": "You are Eko. ACTIVE = short. ULTRA = detailed."}]
                },
                "inputAudioTranscription": {},
            }
        }))

        while True:
            if "setupComplete" in json.loads(await ws.recv()):
                break

        # Audio
        p = pyaudio.PyAudio()
        mic = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        speaker = p.open(format=pyaudio.paInt16, channels=1, rate=24000, output=True)

        cam = cv2.VideoCapture(1)

        # ---------- CONTROL FUNCTIONS ----------
        async def interrupt():
            await ws.send(json.dumps({"clientContent": {"turnComplete": True}}))

        async def switch_to_ultra():
            async with state.lock:
                if state.mode == Mode.ULTRA:
                    return
                state.mode = Mode.ULTRA
                state.ultra_started_at = time.monotonic()

            print("→ ULTRA")

            await interrupt()
            await clear_queue()
            await send_frame()

            await ws.send(json.dumps({
                "realtimeInput": {"text": "Describe everything in detail."}
            }))

        async def switch_to_active():
            async with state.lock:
                if state.mode == Mode.ACTIVE:
                    return
                state.mode = Mode.ACTIVE

            print("→ ACTIVE")

            await interrupt()

        async def clear_queue():
            while not state.playback_q.empty():
                try:
                    state.playback_q.get_nowait()
                except:
                    break

        async def send_frame():
            ret, frame = await asyncio.to_thread(cam.read)
            if not ret:
                return
            _, buf = cv2.imencode(".jpg", frame)

            await ws.send(json.dumps({
                "realtimeInput": {
                    "video": {
                        "data": base64.b64encode(buf.tobytes()).decode(),
                        "mimeType": "image/jpeg"
                    }
                }
            }))

        # ---------- WORKERS ----------
        async def mic_worker():
            while True:
                data = await asyncio.to_thread(mic.read, 1024, exception_on_overflow=False)

                await ws.send(json.dumps({
                    "realtimeInput": {
                        "audio": {
                            "data": base64.b64encode(data).decode(),
                            "mimeType": "audio/pcm;rate=16000"
                        }
                    }
                }))

        async def cam_worker():
            while True:
                await send_frame()

                # controlled by state
                await asyncio.sleep(2 if state.mode == Mode.ACTIVE else 4)

        async def receiver():  # ONLY THIS CHANGES STATE
            async for raw in ws:
                msg = json.loads(raw)
                sc = msg.get("serverContent")
                if not sc:
                    continue

                # Interrupt
                if sc.get("interrupted"):
                    state.interrupted_until = time.monotonic() + 2
                    await clear_queue()

                # USER SPEECH → TRIGGER ULTRA
                if "inputTranscription" in sc:
                    text = sc["inputTranscription"].get("text", "")
                    if text:
                        print("USER:", text)
                        await switch_to_ultra()

                # MODEL FINISHED → BACK TO ACTIVE
                if sc.get("turnComplete"):
                    if state.mode == Mode.ULTRA:
                        if time.monotonic() - state.ultra_started_at > 4:
                            await switch_to_active()

                # AUDIO
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

        async def speaker_worker():
            while True:
                audio = await state.playback_q.get()
                arr = np.frombuffer(audio, dtype=np.int16)
                await asyncio.to_thread(speaker.write, arr.tobytes())

        # ---------- RUN ----------
        await asyncio.gather(
            mic_worker(),
            cam_worker(),
            receiver(),
            speaker_worker(),
        )


# ---------------- MAIN ----------------
async def main():
    while True:
        try:
            await run()
        except Exception as e:
            print("Restarting...", e)
            await asyncio.sleep(2)


if __name__ == "__main__":
    asyncio.run(main())