"""Deterministic rolling awake-state eligibility for passive calibration.

The gate consumes Step 1A frame-quality results and recent EAR/MAR measurements.
It never changes the global fatigue decision. Its only authority is to withhold
measurements from future personalised calibration when wakefulness is uncertain.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import statistics

from vision.calibration_eligibility import FrameEligibility, FrameQuality


class AwakeGateReason(str, Enum):
    INSUFFICIENT_HISTORY = "insufficient_history"
    INSUFFICIENT_USABLE_EVIDENCE = "insufficient_usable_evidence"
    RECENT_USABLE_GAP = "recent_usable_gap"
    EYE_CLOSURE_IN_PROGRESS = "eye_closure_in_progress"
    PROLONGED_EYE_CLOSURE = "prolonged_eye_closure"
    SUSPICIOUS_MOUTH_BEHAVIOUR = "suspicious_mouth_behaviour"
    UNSTABLE_EAR = "unstable_ear"
    DOWNWARD_EAR_TREND = "downward_ear_trend"
    SUSPECTED_FATIGUE = "suspected_fatigue"
    RECOVERY_FREEZE = "recovery_freeze"
    NON_MONOTONIC_TIME = "non_monotonic_time"


@dataclass(frozen=True)
class AwakeGatePolicy:
    """Initial conservative policy values requiring representative validation."""

    window_seconds: float = 10.0
    minimum_history_seconds: float = 8.0
    minimum_weighted_evidence_seconds: float = 6.0
    maximum_usable_gap_seconds: float = 1.0
    maximum_interval_credit_seconds: float = 0.5
    degraded_frame_weight: float = 0.5
    maximum_blink_seconds: float = 0.5
    suspicious_mouth_seconds: float = 1.5
    repeated_mouth_episode_limit: int = 3
    maximum_ear_coefficient_of_variation: float = 0.12
    maximum_ear_downward_slope_per_second: float = 0.004
    quarantine_seconds: float = 2.0
    recovery_freeze_seconds: float = 5.0
    eye_closed_threshold: float = 0.3
    suspicious_mouth_threshold: float = 0.6


@dataclass(frozen=True)
class AwakeGateUpdate:
    eligible: bool
    current_sample_approved: bool
    reasons: tuple[AwakeGateReason, ...]
    weighted_evidence_seconds: float
    history_seconds: float
    quarantined_since: float | None = None


@dataclass(frozen=True)
class _Observation:
    timestamp: float
    quality: FrameQuality
    ear: float | None
    mar: float | None
    awake_candidate: bool


DEFAULT_AWAKE_GATE_POLICY = AwakeGatePolicy()


class AwakeStateGate:
    """Continuously decide whether recent measurements look safely awake."""

    def __init__(self, policy: AwakeGatePolicy = DEFAULT_AWAKE_GATE_POLICY) -> None:
        self.policy = policy
        self._observations: deque[_Observation] = deque()
        self._last_timestamp: float | None = None
        self._eye_closure_started_at: float | None = None
        self._mouth_open_started_at: float | None = None
        self._mouth_episode_starts: deque[float] = deque()
        self._freeze_until: float | None = None

    def _reset_temporal_state(self) -> None:
        self._observations.clear()
        self._eye_closure_started_at = None
        self._mouth_open_started_at = None
        self._mouth_episode_starts.clear()
        self._freeze_until = None

    def _prune(self, now: float) -> None:
        cutoff = now - self.policy.window_seconds
        while self._observations and self._observations[0].timestamp < cutoff:
            self._observations.popleft()
        while self._mouth_episode_starts and self._mouth_episode_starts[0] < cutoff:
            self._mouth_episode_starts.popleft()

    def _quarantine(self, now: float) -> float:
        quarantined_since = now - self.policy.quarantine_seconds
        self._observations = deque(
            item
            for item in self._observations
            if item.timestamp < quarantined_since
        )
        self._freeze_until = now + self.policy.recovery_freeze_seconds
        return quarantined_since

    def _weighted_evidence(self) -> float:
        candidates = [item for item in self._observations if item.awake_candidate]
        evidence = 0.0
        for previous, current in zip(candidates, candidates[1:]):
            interval = current.timestamp - previous.timestamp
            if interval <= self.policy.maximum_usable_gap_seconds:
                weight = (
                    1.0
                    if current.quality is FrameQuality.VALID
                    else self.policy.degraded_frame_weight
                )
                evidence += min(
                    interval,
                    self.policy.maximum_interval_credit_seconds,
                ) * weight
        return evidence

    @staticmethod
    def _ear_slope(observations: list[_Observation]) -> float:
        if len(observations) < 2:
            return 0.0
        origin = observations[0].timestamp
        times = [item.timestamp - origin for item in observations]
        ears = [float(item.ear) for item in observations if item.ear is not None]
        mean_time = statistics.fmean(times)
        mean_ear = statistics.fmean(ears)
        denominator = sum((value - mean_time) ** 2 for value in times)
        if denominator == 0.0:
            return 0.0
        numerator = sum(
            (time - mean_time) * (ear - mean_ear)
            for time, ear in zip(times, ears)
        )
        return numerator / denominator

    def observe(
        self,
        *,
        frame: FrameEligibility,
        ear: float | None,
        mar: float | None,
        fatigue_suspected: bool,
        now: float,
    ) -> AwakeGateUpdate:
        """Add one timestamped result and return the current gate decision."""
        if not math.isfinite(now) or (
            self._last_timestamp is not None and now < self._last_timestamp
        ):
            self._reset_temporal_state()
            self._last_timestamp = now if math.isfinite(now) else None
            return AwakeGateUpdate(
                False,
                False,
                (AwakeGateReason.NON_MONOTONIC_TIME,),
                0.0,
                0.0,
            )

        self._last_timestamp = now
        measurements_valid = (
            frame.may_enter_rolling_gate
            and ear is not None
            and mar is not None
            and math.isfinite(ear)
            and math.isfinite(mar)
        )
        eye_closed = bool(measurements_valid and ear < self.policy.eye_closed_threshold)
        mouth_suspicious = bool(
            measurements_valid and mar > self.policy.suspicious_mouth_threshold
        )
        awake_candidate = bool(
            measurements_valid and not eye_closed and not mouth_suspicious
        )
        self._observations.append(
            _Observation(now, frame.quality, ear, mar, awake_candidate)
        )
        self._prune(now)

        immediate_reasons: list[AwakeGateReason] = []
        quarantined_since: float | None = None

        if eye_closed:
            if self._eye_closure_started_at is None:
                self._eye_closure_started_at = now
            immediate_reasons.append(AwakeGateReason.EYE_CLOSURE_IN_PROGRESS)
            if now - self._eye_closure_started_at >= self.policy.maximum_blink_seconds:
                immediate_reasons.append(AwakeGateReason.PROLONGED_EYE_CLOSURE)
                quarantined_since = self._quarantine(now)
        else:
            self._eye_closure_started_at = None

        if mouth_suspicious:
            if self._mouth_open_started_at is None:
                self._mouth_open_started_at = now
                self._mouth_episode_starts.append(now)
            if (
                now - self._mouth_open_started_at
                >= self.policy.suspicious_mouth_seconds
                or len(self._mouth_episode_starts)
                >= self.policy.repeated_mouth_episode_limit
            ):
                immediate_reasons.append(AwakeGateReason.SUSPICIOUS_MOUTH_BEHAVIOUR)
                quarantined_since = self._quarantine(now)
        else:
            self._mouth_open_started_at = None

        if fatigue_suspected:
            immediate_reasons.append(AwakeGateReason.SUSPECTED_FATIGUE)
            quarantined_since = self._quarantine(now)

        self._prune(now)
        reasons = list(immediate_reasons)
        history_seconds = (
            now - self._observations[0].timestamp if self._observations else 0.0
        )
        candidates = [item for item in self._observations if item.awake_candidate]
        evidence = self._weighted_evidence()

        if history_seconds < self.policy.minimum_history_seconds:
            reasons.append(AwakeGateReason.INSUFFICIENT_HISTORY)
        if evidence < self.policy.minimum_weighted_evidence_seconds:
            reasons.append(AwakeGateReason.INSUFFICIENT_USABLE_EVIDENCE)
        if (
            not candidates
            or now - candidates[-1].timestamp
            > self.policy.maximum_usable_gap_seconds
        ):
            reasons.append(AwakeGateReason.RECENT_USABLE_GAP)

        if len(candidates) >= 2:
            ears = [float(item.ear) for item in candidates if item.ear is not None]
            mean_ear = statistics.fmean(ears)
            if mean_ear <= 0.0 or (
                statistics.pstdev(ears) / mean_ear
                > self.policy.maximum_ear_coefficient_of_variation
            ):
                reasons.append(AwakeGateReason.UNSTABLE_EAR)
            if (
                self._ear_slope(candidates)
                < -self.policy.maximum_ear_downward_slope_per_second
            ):
                reasons.append(AwakeGateReason.DOWNWARD_EAR_TREND)

        if self._freeze_until is not None and now < self._freeze_until:
            reasons.append(AwakeGateReason.RECOVERY_FREEZE)

        unique_reasons = tuple(dict.fromkeys(reasons))
        eligible = not unique_reasons
        return AwakeGateUpdate(
            eligible=eligible,
            current_sample_approved=bool(eligible and awake_candidate),
            reasons=unique_reasons,
            weighted_evidence_seconds=round(evidence, 3),
            history_seconds=round(history_seconds, 3),
            quarantined_since=quarantined_since,
        )
