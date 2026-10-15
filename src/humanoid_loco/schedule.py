"""Velocity-command schedules shared by the MuJoCo rollout, the Isaac evaluator and the C++ loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Segment:
    duration: float
    command: tuple[float, float, float]
    push: tuple[float, float, float] | None = None  # world-frame force on the pelvis [N]


@dataclass(frozen=True)
class Schedule:
    name: str
    segments: tuple[Segment, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> Schedule:
        raw = yaml.safe_load(Path(path).read_text())
        return cls(raw["name"], tuple(Segment(**s) for s in raw["segments"]))

    def to_dict(self) -> dict:
        return {"name": self.name, "segments": [asdict(s) for s in self.segments]}

    @property
    def duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def at(self, t: float) -> Segment:
        for seg in self.segments:
            if t < seg.duration:
                return seg
            t -= seg.duration
        return self.segments[-1]
