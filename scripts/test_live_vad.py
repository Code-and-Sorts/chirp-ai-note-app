#!/usr/bin/env python3
"""Test script to verify VAD is working with audio capture."""

from __future__ import annotations

import queue
import threading
import time

from config.settings import get_settings
from recorder.device_manager import DeviceManager
from recorder.live_audio import LiveAudioStream
from recorder.vad_chunker import VADChunker


def main():
    settings = get_settings()
    device_manager = DeviceManager()

    device_index = device_manager.get_recommended_device()
    if device_index is None:
        print("❌ No suitable audio input device found.")
        return 1

    print("Starting VAD test...")
    print("Speak into your microphone or play audio...")
    print("Press Ctrl+C to stop\n")

    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=500)
    chunk_queue = queue.Queue(maxsize=50)
    event_queue = queue.Queue(maxsize=200)
    level_queue = queue.Queue(maxsize=50)

    audio_stream = LiveAudioStream(
        settings=settings,
        device_manager=device_manager,
        frame_queue=frame_queue,
        stop_event=stop_event,
        level_queue=level_queue,
    )

    try:
        audio_stream.start()
        print("✓ Audio stream started")
        print(f"  Sample rate: {audio_stream.sample_rate} Hz")
        print(f"  Channels: {audio_stream.channels}")
        print(f"  Frame duration: {audio_stream.frame_duration * 1000:.1f} ms\n")
    except Exception as exc:
        print(f"❌ Failed to start audio: {exc}")
        return 1

    vad_chunker = VADChunker(
        frame_queue=frame_queue,
        chunk_queue=chunk_queue,
        stop_event=stop_event,
        sample_rate=audio_stream.sample_rate,
        energy_threshold=0.005,
        aggressiveness=1,
        event_queue=event_queue,
    )
    vad_chunker.start()
    print("✓ VAD chunker started\n")

    last_status_time = time.monotonic()
    frame_count = 0
    chunk_count = 0

    try:
        while not stop_event.is_set():
            try:
                event = event_queue.get(timeout=0.5)
            except queue.Empty:
                now = time.monotonic()
                if now - last_status_time >= 2.0:
                    print(f"[{now:.1f}s] Frames: {frame_count}, Chunks: {chunk_count}")
                    last_status_time = now
                continue

            if event.type == "vad_status":
                frame_count = event.payload.get("frames", 0)
                speech_frames = event.payload.get("speech_frames", 0)
                triggered = event.payload.get("triggered", False)
                chunks = event.payload.get("chunks_emitted", 0)

                speech_pct = (speech_frames / frame_count * 100) if frame_count else 0
                state = "🟢 SPEAKING" if triggered else "⚫ Silent"

                print(
                    f"[VAD] {state} | Frames: {frame_count} ({speech_pct:.1f}% speech) | Chunks: {chunks}"
                )

            elif event.type == "chunk_emitted":
                chunk_count = event.payload.get("chunk_id", 0)
                duration = event.payload.get("duration", 0)
                frames = event.payload.get("frames", 0)

                print(
                    f"[CHUNK #{chunk_count}] Duration: {duration:.2f}s, Frames: {frames}"
                )

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        stop_event.set()
        audio_stream.stop()
        audio_stream.close()
        vad_chunker.join(timeout=2)

    print("\nFinal stats:")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Total chunks emitted: {chunk_count}")

    if chunk_count == 0:
        print("\n⚠️  WARNING: No chunks were emitted!")
        print("Possible issues:")
        print("  - No audio detected (check microphone/audio source)")
        print("  - Energy threshold too high")
        print("  - VAD sample rate mismatch")
        print("  - Frames not reaching VAD chunker")

    return 0


if __name__ == "__main__":
    exit(main())
