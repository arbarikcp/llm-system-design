"""Model registry: versioning, lineage, promotion, rollback (Chapter 43).

The control-plane source of truth for which model version is in production, how
it was produced (lineage), and how to roll back -- the backbone of model
lifecycle management.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelVersion:
    name: str
    version: str
    base_model: str = ""
    training_data: str = ""          # dataset snapshot id (lineage)
    config: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    stage: str = "staging"           # "staging" | "production" | "archived"
    created_at: float = field(default_factory=time.time)

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"


class ModelRegistry:
    def __init__(self) -> None:
        self._versions: Dict[str, ModelVersion] = {}
        self._history: Dict[str, List[str]] = {}     # name -> production version history

    def register(self, mv: ModelVersion) -> None:
        self._versions[mv.key] = mv

    def get(self, name: str, version: str) -> Optional[ModelVersion]:
        return self._versions.get(f"{name}:{version}")

    def production(self, name: str) -> Optional[ModelVersion]:
        for mv in self._versions.values():
            if mv.name == name and mv.stage == "production":
                return mv
        return None

    def promote(self, name: str, version: str) -> ModelVersion:
        """Promote a version to production; demote the current prod to archived."""
        target = self._versions[f"{name}:{version}"]
        current = self.production(name)
        if current and current.key != target.key:
            current.stage = "archived"
        target.stage = "production"
        self._history.setdefault(name, []).append(version)
        return target

    def rollback(self, name: str) -> Optional[ModelVersion]:
        """Roll back to the previous production version."""
        hist = self._history.get(name, [])
        if len(hist) < 2:
            return None
        previous = hist[-2]
        return self.promote(name, previous)

    def lineage(self, name: str, version: str) -> dict:
        mv = self._versions[f"{name}:{version}"]
        return {"model": mv.key, "base_model": mv.base_model,
                "training_data": mv.training_data, "config": mv.config}

    def list(self, name: Optional[str] = None) -> List[ModelVersion]:
        return [mv for mv in self._versions.values() if name is None or mv.name == name]
