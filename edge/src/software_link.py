"""Shared application-level link emulator for PipeSD software mode."""

import collections
import threading
import time
from typing import Any, Dict, Optional, Sequence, Tuple


class SoftwareLink:
    """Full-duplex FIFO link shared by all logical HTTP senders.

    Uplink and downlink have independent queues. The fixed startup cost occupies
    the corresponding resource together with byte serialization, matching the
    paper's per-batch ``alpha + beta * n`` communication model.
    """

    VERSION = "shared-fifo-v1"

    def __init__(
        self,
        uplink_MBps: float,
        downlink_MBps: float,
        uplink_startup_seconds: float = 0.0,
        downlink_startup_seconds: float = 0.0,
        bandwidth_profile: Optional[Sequence[Tuple[float, float]]] = None,
        profile_interval_seconds: float = 20.0,
        profile_started_at: Optional[float] = None,
        history_size: int = 100,
    ) -> None:
        self._lock = threading.Lock()
        self._uplink_MBps = self._validate_rate(uplink_MBps, "uplink")
        self._downlink_MBps = self._validate_rate(downlink_MBps, "downlink")
        self._uplink_startup = max(0.0, float(uplink_startup_seconds))
        self._downlink_startup = max(0.0, float(downlink_startup_seconds))
        self._next_free = {"uplink": 0.0, "downlink": 0.0}
        self._history = collections.deque(maxlen=max(1, int(history_size)))
        self._totals = {
            "uplink": {"transfers": 0, "bytes": 0, "queue_wait_seconds": 0.0, "service_seconds": 0.0},
            "downlink": {"transfers": 0, "bytes": 0, "queue_wait_seconds": 0.0, "service_seconds": 0.0},
        }
        self._profile = [
            (self._validate_rate(up, "profile uplink"), self._validate_rate(down, "profile downlink"))
            for up, down in list(bandwidth_profile or [])
        ]
        self._profile_interval = float(profile_interval_seconds)
        if self._profile and self._profile_interval <= 0:
            raise ValueError("profile_interval_seconds must be positive")
        self._profile_started_at = (
            time.monotonic() if profile_started_at is None else float(profile_started_at)
        )

    @staticmethod
    def _validate_rate(value: float, label: str) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError(f"{label} bandwidth must be positive")
        return value

    def _active_rates_unlocked(self, now: float):
        if not self._profile:
            return self._uplink_MBps, self._downlink_MBps, None
        elapsed = max(0.0, now - self._profile_started_at)
        index = int(elapsed // self._profile_interval) % len(self._profile)
        uplink, downlink = self._profile[index]
        return uplink, downlink, index

    def set_bandwidths(self, uplink_MBps: float, downlink_MBps: Optional[float] = None) -> None:
        """Update future transfers; already-reserved transfers remain stable."""
        with self._lock:
            self._uplink_MBps = self._validate_rate(uplink_MBps, "uplink")
            if downlink_MBps is not None:
                self._downlink_MBps = self._validate_rate(downlink_MBps, "downlink")

    def transmit(self, direction: str, byte_count: int) -> Dict[str, Any]:
        if direction not in self._next_free:
            raise ValueError(f"unknown link direction: {direction}")
        byte_count = max(0, int(byte_count))
        queued_at = time.monotonic()
        with self._lock:
            uplink, downlink, profile_index = self._active_rates_unlocked(queued_at)
            rate_MBps = uplink if direction == "uplink" else downlink
            startup = self._uplink_startup if direction == "uplink" else self._downlink_startup
            started_at = max(queued_at, self._next_free[direction])
            service_seconds = startup + byte_count / (rate_MBps * 1_000_000.0)
            finished_at = started_at + service_seconds
            self._next_free[direction] = finished_at
            record = {
                "direction": direction,
                "bytes": byte_count,
                "bandwidth_MBps": rate_MBps,
                "startup_seconds": startup,
                "profile_index": profile_index,
                "queue_wait_seconds": max(0.0, started_at - queued_at),
                "service_seconds": service_seconds,
            }
            totals = self._totals[direction]
            totals["transfers"] += 1
            totals["bytes"] += byte_count
            totals["queue_wait_seconds"] += record["queue_wait_seconds"]
            totals["service_seconds"] += service_seconds
            self._history.append(dict(record))
        remaining = finished_at - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        return record

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            uplink, downlink, profile_index = self._active_rates_unlocked(time.monotonic())
            return {
                "emulator_version": self.VERSION,
                "queue_policy": "shared-fifo-per-direction",
                "uplink_bandwidth_MBps": uplink,
                "downlink_bandwidth_MBps": downlink,
                "uplink_startup_seconds": self._uplink_startup,
                "downlink_startup_seconds": self._downlink_startup,
                "profile": [list(item) for item in self._profile],
                "profile_interval_seconds": self._profile_interval if self._profile else None,
                "profile_index": profile_index,
                "totals": {key: dict(value) for key, value in self._totals.items()},
                "recent_transfers": list(self._history),
            }
