import asyncio
import base64
import json
import os
from enum import Enum
import time

import cv2
import numpy as np
import pyaudio
import websockets


API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAibZxgLfoLW9KN1D3tgE-1Z3fZIJNbEYM")
MODEL_NAME = "gemini-3.1-flash-live-preview"
WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={API_KEY}"
)

# Audio formats:
# - Input: 16-bit PCM, 16kHz, little-endian
# - Output: 16-bit PCM, 24kHz, little-endian
FORMAT = pyaudio.paInt16
CHANNELS = 1
INPUT_RATE = 16000
OUTPUT_RATE = 24000
CHUNK_FRAMES = 1024


class Mode(str, Enum):
    FAST = "FAST"
    DEEP = "DEEP"


FAST_STEERING = (
    "EKOSPEX FAST MODE.\n"
    "- You are guiding a visually-impaired user who is moving.\n"
    "- Be extremely brief and actionable.\n"
    "- Prefer: STOP, LEFT, RIGHT, STEP_UP, STEP_DOWN, CLEAR.\n"
    "- Mention only immediate hazards and what to do next.\n"
    "- No extra description unless it directly changes the action."
)

DEEP_STEERING = (
    "EKOSPEX DEEP MODE.\n"
    "- User is exploring the environment.\n"
    "- Give short, clear scene descriptions that help navigation.\n"
    "- Prioritize obstacles, stairs/curbs, doors, people, vehicles, and free path.\n"
    "- Include relative direction (left/right/center) and rough distance when helpful.\n"
    "- Keep cognitive load low: 1–2 key items per update."
)


async def connect_and_configure():
    async with websockets.connect(WS_URL) as websocket:
        print("WebSocket Connected")

        # 1) Live API expects the first message to include `setup`.
        # Your server is rejecting `config` with:
        # "Unknown name \"config\": Cannot find field."
        setup_message = {
            "setup": {
                "model": f"models/{MODEL_NAME}",
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "systemInstruction": {"parts": [{"text": "You are a helpful assistant."}]},
                # Turn-taking / barge-in knobs (optional).
                "realtimeInputConfig": {
                    "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
                    "automaticActivityDetection": {
                        "disabled": False,
                        "startOfSpeechSensitivity": "START_SENSITIVITY_HIGH",
                        "endOfSpeechSensitivity": "END_SENSITIVITY_HIGH",
                        "prefixPaddingMs": 200,
                        "silenceDurationMs": 600,
                    },
                },
                # Optional transcription (useful for debugging).
                "inputAudioTranscription": {},
                "outputAudioTranscription": {},
            }
        }
        await websocket.send(json.dumps(setup_message))
        print("Setup sent")

        # 2) Wait for setupComplete before sending realtime input.
        while True:
            raw = await websocket.recv()
            msg = json.loads(raw)
            if "setupComplete" in msg:
                print("Setup complete")
                break

        mode: Mode = Mode.FAST
        mode_lock = asyncio.Lock()

        p = pyaudio.PyAudio()
        in_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=INPUT_RATE,
            input=True,
            frames_per_buffer=CHUNK_FRAMES,
        )
        out_stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=OUTPUT_RATE,
            output=True,
            frames_per_buffer=CHUNK_FRAMES,
        )

        # Camera (optional). Live API video is sent as individual images (JPEG/PNG), max 1 fps.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        video_enabled = bool(cap.isOpened())
        if not video_enabled:
            print("Camera not available; continuing without video.")

        stop = asyncio.Event()
        playback_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)

        async def send_text(text: str):
            await websocket.send(json.dumps({"realtimeInput": {"text": text}}))

        async def clear_playback_queue():
            while True:
                try:
                    playback_q.get_nowait()
                    playback_q.task_done()
                except asyncio.QueueEmpty:
                    return

        def _normalize(s: str) -> str:
            return " ".join(s.lower().strip().split())

        async def maybe_handle_mode_command(transcript_text: str):
            nonlocal mode
            t = _normalize(transcript_text)

            to_deep = any(
                phrase in t
                for phrase in ("deep mode", "describe mode", "understanding mode")
            )
            to_fast = any(
                phrase in t
                for phrase in ("fast mode", "safety mode", "quick mode")
            )

            if not (to_deep or to_fast):
                return

            new_mode = Mode.DEEP if to_deep and not to_fast else Mode.FAST

            async with mode_lock:
                if new_mode == mode:
                    return
                mode = new_mode

            await clear_playback_queue()

            if new_mode == Mode.DEEP:
                print("[mode] DEEP")
                await send_text("Switching to deep mode. Start continuous brief descriptions.")
                await send_text(DEEP_STEERING)
            else:
                print("[mode] FAST")
                await send_text("Switching to fast mode. Keep guidance extremely brief and safety-first.")
                await send_text(FAST_STEERING)
                # Ensure Deep-mode loop doesn't immediately trigger by leaving playback empty.
                # (Deep loop is already guarded by mode check.)

        async def send_mic_audio():
            """Continuously sends mic PCM audio to Live via realtimeInput.audio."""
            try:
                while not stop.is_set():
                    chunk = await asyncio.to_thread(
                        in_stream.read, CHUNK_FRAMES, exception_on_overflow=False
                    )
                    audio_message = {
                        "realtimeInput": {
                            "audio": {
                                "data": base64.b64encode(chunk).decode("utf-8"),
                                "mimeType": f"audio/pcm;rate={INPUT_RATE}",
                            }
                        }
                    }
                    await websocket.send(json.dumps(audio_message))
            except websockets.ConnectionClosed:
                stop.set()

        async def playback_audio():
            """Plays model audio; can be interrupted by clearing the queue."""
            try:
                while not stop.is_set():
                    audio_bytes = await playback_q.get()
                    try:
                        # Output is 16-bit PCM @ 24kHz.
                        audio = np.frombuffer(audio_bytes, dtype=np.int16)
                        await asyncio.to_thread(out_stream.write, audio.tobytes())
                    finally:
                        playback_q.task_done()
            except websockets.ConnectionClosed:
                stop.set()

        async def send_camera_frames():
            """Sends ~1fps JPEG frames as realtimeInput.video."""
            if not video_enabled:
                return
            try:
                while not stop.is_set():
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret:
                        await asyncio.sleep(1.0)
                        continue
                    ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
                    if not ok:
                        await asyncio.sleep(1.0)
                        continue
                    frame_bytes = buf.tobytes()
                    video_message = {
                        "realtimeInput": {
                            "video": {
                                "data": base64.b64encode(frame_bytes).decode("utf-8"),
                                "mimeType": "image/jpeg",
                            }
                        }
                    }
                    await websocket.send(json.dumps(video_message))
                    await asyncio.sleep(1.0)
            except websockets.ConnectionClosed:
                stop.set()

        async def deep_mode_describer():
            """
            In DEEP mode, periodically ask for brief, actionable scene descriptions.
            Skips prompts while the model is actively speaking (playback queue non-empty).
            """
            last_sent = 0.0
            interval_s = 3.0
            try:
                while not stop.is_set():
                    await asyncio.sleep(0.25)
                    async with mode_lock:
                        current_mode = mode

                    if current_mode != Mode.DEEP:
                        continue

                    # Don't stack prompts if audio is already being played.
                    if playback_q.qsize() > 0:
                        continue

                    now = time.monotonic()
                    if now - last_sent < interval_s:
                        continue

                    last_sent = now
                    await send_text(
                        "Deep update: describe only the most important things for safe navigation "
                        "right now (1–2 items). Include left/right and rough distance if helpful."
                    )
            except websockets.ConnectionClosed:
                stop.set()

        async def receive_loop():
            """Receives server messages; enqueues audio and handles interruptions."""
            try:
                async for raw in websocket:
                    msg = json.loads(raw)

                    if "serverContent" in msg:
                        sc = msg["serverContent"]

                        # Barge-in / interruption signal: stop playback ASAP.
                        if sc.get("interrupted"):
                            print("[server] interrupted=True")
                            await clear_playback_queue()

                        if "inputTranscription" in sc:
                            user_text = sc["inputTranscription"].get("text", "")
                            if user_text:
                                print(f"[user] {user_text}")
                                await maybe_handle_mode_command(user_text)
                        if "outputTranscription" in sc:
                            print(f"[model] {sc['outputTranscription'].get('text', '')}")

                        mt = sc.get("modelTurn")
                        if mt and "parts" in mt:
                            for part in mt["parts"]:
                                inline = part.get("inlineData")
                                if inline and inline.get("data"):
                                    audio_bytes = base64.b64decode(inline["data"])
                                    try:
                                        playback_q.put_nowait(audio_bytes)
                                    except asyncio.QueueFull:
                                        # If we fall behind, drop queued audio to keep latency low.
                                        await clear_playback_queue()
            except websockets.ConnectionClosed as e:
                print(f"WebSocket closed: code={e.code} reason={e.reason}")
                stop.set()

        # 3) Run full duplex loops.
        try:
            # Seed initial mode behavior.
            await send_text(FAST_STEERING)
            await asyncio.gather(
                send_mic_audio(),
                receive_loop(),
                playback_audio(),
                send_camera_frames(),
                deep_mode_describer(),
            )
        except websockets.ConnectionClosed as e:
            print(f"WebSocket closed: code={e.code} reason={e.reason}")
        finally:
            stop.set()
            try:
                cap.release()
            except Exception:
                pass
            try:
                in_stream.stop_stream()
                in_stream.close()
            except Exception:
                pass
            try:
                out_stream.stop_stream()
                out_stream.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass


async def main():
    await connect_and_configure()


if __name__ == "__main__":
    asyncio.run(main())