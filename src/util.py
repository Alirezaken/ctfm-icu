"""Small shared utilities: logging and stage checkpoint/resume markers.

Every stage must checkpoint and resume, because jobs sit in a queue (§0). A stage
writes a JSON marker under paths.checkpoints when a unit of work completes; on
re-run it skips units whose marker already exists unless --force is given.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def die(msg: str, code: int = 1):
    print(f"ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


class Checkpoint:
    """Filesystem resume markers for a stage, e.g.

        ckpt = Checkpoint(cfg, "extract_embeddings")
        if ckpt.done("images_p10") and not force:
            return
        ... do work ...
        ckpt.mark("images_p10", n_vectors=12345)
    """

    def __init__(self, cfg, stage: str):
        self.dir = cfg.storage("checkpoints", stage)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _f(self, unit: str) -> Path:
        return self.dir / f"{unit}.json"

    def done(self, unit: str) -> bool:
        return self._f(unit).exists()

    def mark(self, unit: str, **meta):
        meta["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._f(unit).write_text(json.dumps(meta, indent=2))

    def clear(self, unit: str | None = None):
        targets = [self._f(unit)] if unit else list(self.dir.glob("*.json"))
        for t in targets:
            t.unlink(missing_ok=True)
