"""Teacher alpha and distillation strength g are deliberately separate."""

import json
from pathlib import Path


class Router:
    def __init__(self, cfg):
        self.routes = cfg["routing"]["tasks"]
        self.dimension = cfg["data"]["dimension"]
        self.quality = {}
        if cfg["routing"].get("calibration_file"):
            report = json.loads(Path(cfg["routing"]["calibration_file"]).read_text())
            if report.get("dimension") != self.dimension:
                raise ValueError("Calibration dimension differs from this training run")
            self.quality = report["routes"]

    def resolve(self, row):
        task = row["task_id"]
        if task not in self.routes:
            raise ValueError(f"No teacher route for task {task}; cross-task fallback is forbidden")
        route = self.routes[task]
        raw = dict(route["teachers"])
        strength = float(route.get("strength", 1.0)) * row.get("distill_strength", 1.0)
        if self.quality:
            if task not in self.quality:
                raise ValueError(f"Calibration has no result for task {task}")
            calibrated = self.quality[task]
            raw = {k: v * calibrated["gains"].get(k, 0.0) for k, v in raw.items()}
            strength *= calibrated["strength"]
        total = sum(raw.values())
        if total <= 0 or strength == 0:
            return {}, 0.0
        return {k: v/total for k, v in raw.items() if v > 0}, strength
