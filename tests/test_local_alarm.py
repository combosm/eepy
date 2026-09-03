from __future__ import annotations

import unittest

from vision.local_alarm import LocalAlarm


class FakeSound:
    def __init__(self, channel=object(), error: Exception | None = None) -> None:
        self.channel = channel
        self.error = error
        self.play_count = 0

    def play(self):
        self.play_count += 1
        if self.error is not None:
            raise self.error
        return self.channel


class FakeMixer:
    def __init__(self, *, initialised: bool = False) -> None:
        self.initialised = initialised
        self.init_arguments = None
        self.sound = FakeSound()
        self.buffer = None

    def get_init(self):
        return (44_100, -16, 1) if self.initialised else None

    def init(self, **kwargs) -> None:
        self.initialised = True
        self.init_arguments = kwargs

    def Sound(self, *, buffer):
        self.buffer = buffer
        return self.sound


class BrokenMixer:
    def get_init(self):
        raise RuntimeError("audio device unavailable")


class LocalAlarmTests(unittest.TestCase):
    def test_initialises_local_mixer_and_prepares_nonempty_sound(self) -> None:
        mixer = FakeMixer()

        alarm = LocalAlarm(mixer)

        self.assertTrue(alarm.available)
        self.assertEqual(mixer.init_arguments["frequency"], 44_100)
        self.assertGreater(len(mixer.buffer), 0)

    def test_play_reports_success_when_channel_is_allocated(self) -> None:
        mixer = FakeMixer(initialised=True)
        alarm = LocalAlarm(mixer)

        result = alarm.play()

        self.assertTrue(result.delivered)
        self.assertEqual(mixer.sound.play_count, 1)

    def test_no_available_channel_is_reported_as_failure(self) -> None:
        mixer = FakeMixer(initialised=True)
        mixer.sound.channel = None
        alarm = LocalAlarm(mixer)

        result = alarm.play()

        self.assertFalse(result.delivered)
        self.assertEqual(result.error, "no audio channel available")

    def test_initialisation_failure_preserves_visual_fallback_state(self) -> None:
        alarm = LocalAlarm(BrokenMixer())

        result = alarm.play()

        self.assertFalse(alarm.available)
        self.assertFalse(result.delivered)
        self.assertEqual(result.error, "audio device unavailable")

    def test_playback_exception_is_contained(self) -> None:
        mixer = FakeMixer(initialised=True)
        mixer.sound.error = RuntimeError("playback failed")
        alarm = LocalAlarm(mixer)

        result = alarm.play()

        self.assertFalse(result.delivered)
        self.assertEqual(result.error, "playback failed")


if __name__ == "__main__":
    unittest.main()
