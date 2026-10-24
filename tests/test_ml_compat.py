"""Dependency-compatibility checks for the local ML stack (skipped when it isn't installed)."""

import struct
import wave

import pytest


def test_faster_whisper_can_decode_audio_with_installed_pyav(tmp_path):
    # Regression: PyAV 19 dropped an argument faster-whisper passes, breaking every transcription.
    audio = pytest.importorskip("faster_whisper.audio")
    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<16000h", *([0] * 16000)))

    samples = audio.decode_audio(str(path), sampling_rate=16000)
    assert len(samples) == 16000
