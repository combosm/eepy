"""Deterministic fatigue-episode and intervention state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class InterventionAction(str, Enum):
    INITIAL_ALERT = "initial_alert"
    REPEAT_ALERT = "repeat_alert"
    RECOVERED = "recovered"


@dataclass(frozen=True)
class InterventionPolicy:
    """Timing policy for repeating a local alert during persistent fatigue."""

    repeat_cooldown_seconds: float = 15.0


@dataclass(frozen=True)
class InterventionUpdate:
    active: bool
    episode_id: int | None
    escalation_level: int
    action: InterventionAction | None

    @property
    def should_alert(self) -> bool:
        return self.action in {
            InterventionAction.INITIAL_ALERT,
            InterventionAction.REPEAT_ALERT,
        }


DEFAULT_INTERVENTION_POLICY = InterventionPolicy()


class InterventionController:
    """Create one episode per fatigue period and rate-limit repeat alerts."""

    def __init__(
        self,
        policy: InterventionPolicy = DEFAULT_INTERVENTION_POLICY,
    ) -> None:
        if policy.repeat_cooldown_seconds <= 0.0:
            raise ValueError("repeat cooldown must be positive")
        self.policy = policy
        self._next_episode_id = 1
        self._active_episode_id: int | None = None
        self._last_alert_at: float | None = None
        self._last_update_at: float | None = None
        self._escalation_level = 0

    def update(
        self,
        *,
        is_drowsy: bool | None,
        now: float,
    ) -> InterventionUpdate:
        """Advance state; ``None`` means recovery cannot currently be observed."""
        if not math.isfinite(now):
            raise ValueError("intervention timestamp must be finite")
        if self._last_update_at is not None and now < self._last_update_at:
            raise ValueError("intervention timestamps must be monotonic")
        self._last_update_at = now

        if is_drowsy and self._active_episode_id is None:
            self._active_episode_id = self._next_episode_id
            self._next_episode_id += 1
            self._last_alert_at = now
            self._escalation_level = 1
            return self._snapshot(InterventionAction.INITIAL_ALERT)

        if is_drowsy or (is_drowsy is None and self._active_episode_id is not None):
            action = None
            if (
                self._last_alert_at is not None
                and now - self._last_alert_at
                >= self.policy.repeat_cooldown_seconds
            ):
                self._last_alert_at = now
                self._escalation_level += 1
                action = InterventionAction.REPEAT_ALERT
            return self._snapshot(action)

        if is_drowsy is False and self._active_episode_id is not None:
            update = InterventionUpdate(
                active=False,
                episode_id=self._active_episode_id,
                escalation_level=self._escalation_level,
                action=InterventionAction.RECOVERED,
            )
            self._active_episode_id = None
            self._last_alert_at = None
            self._escalation_level = 0
            return update

        return self._snapshot(None)

    def _snapshot(
        self,
        action: InterventionAction | None,
    ) -> InterventionUpdate:
        return InterventionUpdate(
            active=self._active_episode_id is not None,
            episode_id=self._active_episode_id,
            escalation_level=self._escalation_level,
            action=action,
        )
