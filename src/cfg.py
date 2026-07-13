"""Load and resolve config.yaml -- the single source of truth (§0).

The whole pipeline reads its configuration through `load()`. Path strings may use
`${ENV}` and `${ENV:default}` interpolation so the same config.yaml is portable
across machines without editing source. Derived storage subdirectories are
resolved to absolute paths under `paths.storage_root`.

`Config` is a thin dotted-access wrapper over the parsed dict; `require()` enforces
the "no fabrication" rule -- a stage asks for the values it needs and fails loudly
(naming the config key) if any are still null, rather than inventing a default.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "config.yaml"

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::([^}]*))?\}")


def _interp(value):
    """Recursively expand ${VAR} / ${VAR:default} in all string leaves."""
    if isinstance(value, str):
        def sub(m):
            var, default = m.group(1), m.group(2)
            return os.environ.get(var, default if default is not None else "")
        return _ENV_RE.sub(sub, value)
    if isinstance(value, dict):
        return {k: _interp(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp(v) for v in value]
    return value


class Config:
    """Dotted, read-only access over the parsed config with null-checking."""

    def __init__(self, data: dict, path: Path):
        self._d = data
        self.path = path
        self.reduction = self.get("estimator.embedding_reduction", "none")
        self.pca_components = int(self.get("estimator.pca_components", 30))
        self.folds = int(self.get("estimator.cross_fitting_folds", 5))
        self.seed = int(self.get("run.seed", 42))
        self.trim = float(self.get("estimator.trim", 0.01))
        self.nboot = int(self.get("bootstrap.n_resamples", 2000))

    def __getitem__(self, key):
        return self._d[key]

    def get(self, dotted: str, default=None):
        """cfg.get('estimator.crossfit_folds') -> 5; missing/null -> default."""
        node = self._d
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return default if node is None else node

    def require(self, *dotted_keys):
        """Return the listed values; raise if any is missing or null (§0).

        Used at the top of each stage so a run aborts with a precise message
        ("config key X is null -- Soroosh must supply it") instead of fabricating.
        """
        missing, out = [], []
        for k in dotted_keys:
            v = self.get(k, None)
            if v is None:
                missing.append(k)
            out.append(v)
        if missing:
            raise ValueError(
                "config.yaml is missing required value(s): "
                + ", ".join(missing)
                + " -- supply them before running this stage (no fabrication, §0)."
            )
        return out[0] if len(out) == 1 else out

    # --- storage path helpers -------------------------------------------------
    @property
    def storage_root(self) -> Path:
        return Path(self.get("paths.storage_root"))

    def storage(self, subkey: str, *parts) -> Path:
        """Absolute path under storage_root for a named subdir, e.g.
        cfg.storage('results', 'effects.csv')."""
        sub = self.get(f"paths.{subkey}", subkey)
        return self.storage_root.joinpath(sub, *parts)

    def input(self, subkey: str) -> Path:
        return Path(self.get(f"paths.{subkey}"))


def load(path: str | os.PathLike | None = None) -> Config:
    p = Path(path) if path else _DEFAULT_CONFIG
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    with open(p) as fh:
        data = _interp(yaml.safe_load(fh))
    return Config(data, p)
