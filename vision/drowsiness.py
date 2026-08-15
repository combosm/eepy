"""Time-based fatigue state for the driver-monitoring pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class FatigueUpdate:
    eye_evidence: float
    mouth_evidence: float
    is_drowsy: bool
    just_confirmed: bool
    just_recovered: bool


class FatigueState:
    """Combine sustained eye closure with secondary recent mouth evidence.

    Eye evidence is an elapsed-time evidence budget. After the blink grace
    period, a severity of 1.0 fills the budget in ``eye_ramp_seconds``;
    lower severities take proportionally longer. Mouth evidence is retained as
    a time-bounded average and can corroborate, but cannot confirm by itself.
    """

    def __init__(
        self,
        eye_grace_seconds: float,
        eye_ramp_seconds: float,
        mouth_window_seconds: float,
        eye_weight: float,
        mouth_weight: float,
    ) -> None:
        self.eye_grace_seconds = eye_grace_seconds
        self.eye_ramp_seconds = eye_ramp_seconds
        self.mouth_window_seconds = mouth_window_seconds
        self.eye_weight = eye_weight
        self.mouth_weight = mouth_weight
        self.eye_closure_started_at: float | None = None
        self.mouth_samples: deque[tuple[float, float]] = deque()
        self.is_drowsy = False

    def _recent_mouth_evidence(self, mouth_severity: float, now: float) -> float:
        self.mouth_samples.append((now, mouth_severity))
        cutoff = now - self.mouth_window_seconds
        while self.mouth_samples and self.mouth_samples[0][0] < cutoff:
            self.mouth_samples.popleft()
        return sum(score for _, score in self.mouth_samples) / len(
            self.mouth_samples
        )

    def update(
        self,
        eye_severity: float,
        eye_closed: bool,
        mouth_severity: float,
        now: float,
    ) -> FatigueUpdate:
        if eye_closed:
            if self.eye_closure_started_at is None:
                self.eye_closure_started_at = now
            sustained_seconds = max(
                0.0,
                now - self.eye_closure_started_at - self.eye_grace_seconds,
            )
            eye_evidence = (
                eye_severity * sustained_seconds / self.eye_ramp_seconds
            )
        else:
            self.eye_closure_started_at = None
            eye_evidence = 0.0

        mouth_evidence = self._recent_mouth_evidence(mouth_severity, now)

        # The threshold equals the eye weight so eye evidence of 1.0 always
        # confirms directly. Mouth evidence can only corroborate an active eye
        # closure because its maximum weighted contribution is 0.3.
        combined_evidence = (
            self.eye_weight * eye_evidence
            + self.mouth_weight * mouth_evidence
        )
        # Callers may supply NumPy scalar measurements. Coerce the result so
        # monitoring state remains safe to expose through Flask's JSON API.
        is_drowsy = bool(eye_closed and combined_evidence >= self.eye_weight)
        just_confirmed = is_drowsy and not self.is_drowsy
        just_recovered = self.is_drowsy and not is_drowsy
        self.is_drowsy = is_drowsy

        return FatigueUpdate(
            eye_evidence=eye_evidence,
            mouth_evidence=mouth_evidence,
            is_drowsy=is_drowsy,
            just_confirmed=just_confirmed,
            just_recovered=just_recovered,
        )

    def face_missing(self) -> FatigueUpdate:
        """Discard state when the tracked driver face is no longer present."""
        was_drowsy = self.is_drowsy
        self.eye_closure_started_at = None
        self.mouth_samples.clear()
        self.is_drowsy = False
        return FatigueUpdate(0.0, 0.0, False, False, was_drowsy)
