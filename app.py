import asyncio
import base64
import json
import time
from enum import Enum

import cv2
import numpy as np
import pyaudio
import websockets




from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    GEMINI_API_KEY: str
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

MODEL_NAME = "gemini-3.1-flash-live-preview"
WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={settings.GEMINI_API_KEY}"
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


SYSTEM_INSTRUCTION = (
    "You are Eko, a friendly navigation buddy for a visually impaired person.\n"
    "You have a camera and a microphone. You can see and hear everything around the user "
    "right now. Think of yourself as their trusted friend walking beside them, "
    "being their eyes.\n"
    "\n"
    "## How you talk\n"
    "- Sound like a real person, not a robot. Use contractions, natural phrasing.\n"
    "- Say 'watch out' not 'CAUTION'. Say 'you're good' not 'CLEAR'.\n"
    "- Be warm but efficient. No filler, no fluff, no questions.\n"
    "- Use clock directions for spatial info: 12 o'clock is straight ahead, "
    "3 is to the right, 9 is to the left. Combine with distance in steps.\n"
    "- Example: 'Chair at your 2 o'clock, about 4 steps out.'\n"
    "\n"
    "## Modes\n"
    "You start in FAST mode. User can switch by saying 'deep mode' or 'fast mode'.\n"
    "\n"
    "### FAST mode (default) -- user is moving, keep it quick\n"
    "Use three urgency levels:\n"
    "- DANGER (collision/fall within 1-2 steps): Interrupt immediately. "
    "Be sharp and urgent. 'Whoa, stop! Stairs right in front of you.'\n"
    "- HEADS UP (obstacle in 3-5 steps): Warn casually. "
    "'Chair coming up on your right, about 3 steps.'\n"
    "- ALL GOOD (path is clear): Quick confirmation. "
    "'You're good, keep going straight.' or 'All clear ahead.'\n"
    "Keep it to ONE sentence max. Only mention what matters for the next few steps.\n"
    "\n"
    "### DEEP mode -- user wants to understand the space\n"
    "Paint a quick picture of the scene like a friend would:\n"
    "- Where are they? (hallway, sidewalk, room, intersection...)\n"
    "- What's ahead, left, right? Name specific objects with clock direction and distance.\n"
    "- Any people, vehicles, doors, stairs?\n"
    "- Where's the clear path?\n"
    "Example: 'OK so you're in a hallway. There's a door on your left about 4 steps away, "
    "and the hall keeps going straight for maybe 10 meters. Someone's walking toward you "
    "from 12 o'clock.'\n"
    "Keep it to 2-3 sentences max.\n"
    "\n"
    "## Rules (always)\n"
    "- If the user talks to you, STOP and respond to them immediately. "
    "Their voice always comes first.\n"
    "- You CAN see. You have a camera. Never say you can't see, "
    "never say you're just a language model.\n"
    "- Always describe what's actually in the camera frame. "
    "Even if it's blurry or dark, describe what you can make out.\n"
    "- Never ask the user questions. Just tell them what they need to know.\n"
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
                "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
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

        p = None
        in_stream = None
        out_stream = None
        cap = None
        video_enabled = False
        stop = asyncio.Event()
        playback_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        last_user_speech: float = 0.0
        last_model_output: float = 0.0
        interrupted_until: float = 0.0

        try:
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

            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            video_enabled = bool(cap.isOpened())
            if not video_enabled:
                print("Camera not available; continuing without video.")

            async def send_text(text: str):
                await websocket.send(json.dumps({"realtimeInput": {"text": text}}))

            async def send_fresh_frame():
                """Grab and send a fresh camera frame right now."""
                if not video_enabled:
                    return
                ret, frame = await asyncio.to_thread(cap.read)
                if not ret:
                    return
                ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
                if not ok:
                    return
                msg = {
                    "realtimeInput": {
                        "video": {
                            "data": base64.b64encode(buf.tobytes()).decode("utf-8"),
                            "mimeType": "image/jpeg",
                        }
                    }
                }
                await websocket.send(json.dumps(msg))

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
                    kw in t
                    for kw in (
                        "deep mode", "deep", "describe mode", "describe",
                        "understanding mode", "detail", "detailed",
                    )
                )
                to_fast = any(
                    kw in t
                    for kw in (
                        "fast mode", "fast", "safety mode", "quick mode",
                        "quick", "speed",
                    )
                )

                if not (to_deep or to_fast):
                    return

                new_mode = Mode.DEEP if to_deep and not to_fast else Mode.FAST

                async with mode_lock:
                    if new_mode == mode:
                        return
                    mode = new_mode

                await clear_playback_queue()
                await send_fresh_frame()

                if new_mode == Mode.DEEP:
                    print("[mode] DEEP")
                    await send_text(
                        "[MODE SWITCH: DEEP] Say 'OK, switching to deep mode' "
                        "and describe the full scene around them."
                    )
                else:
                    print("[mode] FAST")
                    await send_text(
                        "[MODE SWITCH: FAST] Say 'Got it, back to fast mode' "
                        "and give a quick safety check of what's ahead."
                    )

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
                """Sends JPEG frames: every 2s in FAST mode, 3s in DEEP mode."""
                if not video_enabled:
                    return
                try:
                    while not stop.is_set():
                        ret, frame = await asyncio.to_thread(cap.read)
                        if not ret:
                            await asyncio.sleep(0.5)
                            continue
                        ok, buf = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
                        if not ok:
                            await asyncio.sleep(0.5)
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

                        async with mode_lock:
                            current_mode = mode
                        await asyncio.sleep(2.0 if current_mode == Mode.FAST else 3.0)
                except websockets.ConnectionClosed:
                    stop.set()

            async def periodic_vision_prompter():
                """
                Periodically nudges the model to describe the scene.
                Mode-aware intervals and cooldowns.
                """
                last_prompt = 0.0
                try:
                    while not stop.is_set():
                        await asyncio.sleep(0.5)

                        if playback_q.qsize() > 0:
                            continue

                        now = time.monotonic()

                        async with mode_lock:
                            current_mode = mode

                        if current_mode == Mode.FAST:
                            prompt_interval = 3.0
                            speech_cooldown = 3.0
                            model_cooldown = 1.5
                        else:
                            prompt_interval = 8.0
                            speech_cooldown = 6.0
                            model_cooldown = 3.0

                        if now - last_user_speech < speech_cooldown:
                            continue

                        if now - last_model_output < model_cooldown:
                            continue

                        if now - last_prompt < prompt_interval:
                            continue

                        last_prompt = now

                        await send_fresh_frame()

                        if current_mode == Mode.FAST:
                            await send_text(
                                "Quick -- what's in front of them right now? "
                                "Any danger, anything to watch out for? "
                                "One sentence, clock directions and steps."
                            )
                        else:
                            await send_text(
                                "Describe the full scene around them. "
                                "Where are they? What's ahead, left, right? "
                                "Any people, doors, obstacles, vehicles? "
                                "Where's the clear path? "
                                "Use clock directions and steps, 2-3 sentences."
                            )
                except websockets.ConnectionClosed:
                    stop.set()

            async def receive_loop():
                """Receives server messages; enqueues audio and handles interruptions."""
                nonlocal last_user_speech, last_model_output, interrupted_until
                try:
                    async for raw in websocket:
                        msg = json.loads(raw)

                        if "serverContent" in msg:
                            sc = msg["serverContent"]

                            # Barge-in: user started speaking. Block model audio
                            # for 3 seconds so the user's speech gets through.
                            if sc.get("interrupted"):
                                now = time.monotonic()
                                last_user_speech = now
                                interrupted_until = now + 3.0
                                print("[server] interrupted=True")
                                await clear_playback_queue()

                            if "inputTranscription" in sc:
                                user_text = sc["inputTranscription"].get("text", "")
                                if user_text:
                                    last_user_speech = time.monotonic()
                                    print(f"[user] {user_text}")
                                    await maybe_handle_mode_command(user_text)
                            if "outputTranscription" in sc:
                                print(
                                    f"[model] {sc['outputTranscription'].get('text', '')}"
                                )

                            mt = sc.get("modelTurn")
                            if mt and "parts" in mt:
                                # Drop model audio that arrives during an interruption.
                                if time.monotonic() < interrupted_until:
                                    continue
                                for part in mt["parts"]:
                                    inline = part.get("inlineData")
                                    if inline and inline.get("data"):
                                        last_model_output = time.monotonic()
                                        audio_bytes = base64.b64decode(inline["data"])
                                        try:
                                            playback_q.put_nowait(audio_bytes)
                                        except asyncio.QueueFull:
                                            await clear_playback_queue()
                except websockets.ConnectionClosed as e:
                    print(f"WebSocket closed: code={e.code} reason={e.reason}")
                    stop.set()

            # 3) Run full duplex loops.
            try:
                await asyncio.gather(
                    send_mic_audio(),
                    receive_loop(),
                    playback_audio(),
                    send_camera_frames(),
                    periodic_vision_prompter(),
                )
            except websockets.ConnectionClosed as e:
                print(f"WebSocket closed: code={e.code} reason={e.reason}")


        finally:
            stop.set()
            try:
                if cap is not None:
                    cap.release()
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if in_stream is not None:
                    in_stream.stop_stream()
                    in_stream.close()
            except Exception:
                pass
            try:
                if out_stream is not None:
                    out_stream.stop_stream()
                    out_stream.close()
            except Exception:
                pass
            try:
                if p is not None:
                    p.terminate()
            except Exception:
                pass


async def main():
    await connect_and_configure()


if __name__ == "__main__":
    asyncio.run(main())