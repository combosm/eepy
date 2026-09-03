"""Best-effort local audible alarm with no network or service dependency."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
import math
import sys
from typing import Any


SAMPLE_RATE = 44_100
TONE_FREQUENCY_HZ = 880.0
TONE_SECONDS = 0.18
GAP_SECONDS = 0.08
TONE_REPETITIONS = 3
TONE_VOLUME = 0.65


@dataclass(frozen=True)
class AlarmResult:
    delivered: bool
    error: str | None = None


def _alarm_pcm() -> bytes:
    """Build a short three-pulse signed 16-bit mono alarm."""
    peak = int(32767 * TONE_VOLUME)
    tone_samples = int(SAMPLE_RATE * TONE_SECONDS)
    gap_samples = int(SAMPLE_RATE * GAP_SECONDS)
    samples = array("h")

    for repetition in range(TONE_REPETITIONS):
        samples.extend(
            int(
                peak
                * math.sin(2.0 * math.pi * TONE_FREQUENCY_HZ * index / SAMPLE_RATE)
            )
            for index in range(tone_samples)
        )
        if repetition < TONE_REPETITIONS - 1:
            samples.extend([0] * gap_samples)

    if sys.byteorder != "little":
        samples.byteswap()
    return samples.tobytes()


class LocalAlarm:
    """Prepare one local sound and start playback without waiting for completion."""

    def __init__(self, mixer: Any | None = None) -> None:
        self._sound = None
        self._error: str | None = None
        try:
            if mixer is None:
                import pygame

                mixer = pygame.mixer
            if mixer.get_init() is None:
                mixer.init(
                    frequency=SAMPLE_RATE,
                    size=-16,
                    channels=1,
                    buffer=512,
                )
            self._sound = mixer.Sound(buffer=_alarm_pcm())
        except Exception as error:
            self._error = str(error) or type(error).__name__

    @property
    def available(self) -> bool:
        return self._sound is not None

    @property
    def error(self) -> str | None:
        return self._error

    def play(self) -> AlarmResult:
        if self._sound is None:
            # The terminal bell is only a last-resort local attempt. The dashboard's
            # visual warning remains the deterministic fallback communicated to users.
            print("\a", end="", flush=True)
            return AlarmResult(False, self._error or "local audio unavailable")
        try:
            channel = self._sound.play()
            if channel is None:
                return AlarmResult(False, "no audio channel available")
            return AlarmResult(True)
        except Exception as error:
            self._error = str(error) or type(error).__name__
            return AlarmResult(False, self._error)
