#!/usr/bin/env python3
"""main.py -- THE single entry point.

    python main.py --list
    python main.py --stage emulate
    python main.py --stage estimate --intervention fluids_sepsis
    python main.py --stage all                    # every stage, every intervention
    python main.py --stage estimate --force

BY DEFAULT EVERY PER-INTERVENTION STAGE RUNS EVERY INTERVENTION.
The previous pipeline defaulted each stage to a single hard-coded intervention, so
`--stage all` quietly produced a one-intervention study that looked like a
four-intervention one. Pass --intervention to narrow it deliberately; otherwise the full
set runs.

Stages checkpoint and resume, because these jobs sit in a SLURM queue and get preempted.
Results, embeddings and manifests are written under paths.storage_root, never beside the
code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import cfg as cfg_mod
from src import stages
from src.util import log, die


def _parse(argv=None):
    p = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", help="stage name, or 'all'")
    p.add_argument("--config", default=None)
    p.add_argument("--intervention", default=None,
                   help="restrict a per-intervention stage to one intervention "
                        "(default: ALL of them)")
    p.add_argument("--force", action="store_true", help="ignore checkpoints")
    p.add_argument("--list", action="store_true")
    return p.parse_args(argv)


def _run(name, cfg, args):
    mod = stages.get(name)
    kw = {"force": args.force}
    if name in stages.PER_INTERVENTION:
        kw["intervention"] = args.intervention        # None => all interventions
    log(f"{'='*70}")
    log(f"=== STAGE: {name}")
    log(f"{'='*70}")
    mod.run(cfg, **kw)


def main(argv=None):
    args = _parse(argv)

    if args.list or not args.stage:
        print("Stages (dependency order):")
        for i, s in enumerate(stages.STAGES, 1):
            tag = " [per-intervention]" if s in stages.PER_INTERVENTION else ""
            print(f"  {i:>2}. {s}{tag}")
        if not args.stage:
            print("\nNothing to do: pass --stage <name> or --stage all.")
        return 0

    cfg = cfg_mod.load(args.config)
    log(f"config:       {cfg.path}")
    log(f"storage_root: {cfg.storage_root}")
    log(f"seed={cfg.seed} folds={cfg.folds} boot={cfg.nboot} reduction={cfg.reduction}")

    if args.stage == "all":
        for s in stages.STAGES:
            _run(s, cfg, args)
        return 0

    if args.stage not in stages.STAGES:
        die(f"unknown stage '{args.stage}'. Known: {', '.join(stages.STAGES)}")
    _run(args.stage, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
