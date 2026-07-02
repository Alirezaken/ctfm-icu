#!/usr/bin/env python3
"""main_causal.py -- THE single entry point for the causal pipeline (§0).

Everything runs from here, dispatched by a stage flag. Per-stage logic lives in
src/stages/. There is no other entry point.

  python main_causal.py --list
  python main_causal.py --stage link
  python main_causal.py --stage emulate --intervention fluids_sepsis
  python main_causal.py --stage all                 # run every stage in §9 order
  python main_causal.py --stage estimate --force    # ignore checkpoints

Each stage checkpoints and resumes (jobs sit in a queue). Results, embeddings and
patient-derived tables are written under paths.storage_root, never beside the code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# make `import src.*` work regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import cfg as cfg_mod
from src import stages
from src.util import log, die


def _parse_args(argv=None):
    p = argparse.ArgumentParser(prog="main_causal.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", help="stage to run, or 'all' for the full §9 order")
    p.add_argument("--config", default=None, help="path to config.yaml (default: repo root)")
    p.add_argument("--intervention", default=None,
                   help="intervention key for per-intervention stages (default: fluids_sepsis)")
    p.add_argument("--variant", default=None,
                   help="embedding-fix variant from config_variants.yaml (e.g. pscore); "
                        "writes to its own results dir, leaving config.yaml + results/ untouched")
    p.add_argument("--force", action="store_true", help="ignore checkpoints / recompute")
    p.add_argument("--list", action="store_true", help="list stages and exit")
    return p.parse_args(argv)


def _run_stage(name: str, cfg, args):
    mod = stages.get(name)
    kwargs = {"force": args.force}
    # per-intervention stages accept --intervention
    if args.intervention is not None and "intervention" in mod.run.__code__.co_varnames:
        kwargs["intervention"] = args.intervention
    log(f"=== stage: {name} ===")
    mod.run(cfg, **kwargs)


def main(argv=None):
    args = _parse_args(argv)

    if args.list or not args.stage:
        print("Stages (run order, §9):")
        for i, s in enumerate(stages.STAGES, 1):
            print(f"  {i:>2}. {s}")
        if not args.stage:
            print("\nNothing to do: pass --stage <name> or --stage all.")
        return 0

    cfg = cfg_mod.load(args.config)
    log(f"config: {cfg.path}")
    log(f"storage_root: {cfg.storage_root}")

    if args.variant:
        import yaml
        vpath = Path(__file__).resolve().parent / "config_variants.yaml"
        vdefs = (yaml.safe_load(open(vpath)) or {}).get("variants", {})
        if args.variant not in vdefs:
            die(f"unknown variant '{args.variant}'. Known: {', '.join(vdefs)}")
        v = vdefs[args.variant]
        cfg.apply_variant(v["results_dir"], v["reduction"], v.get("pca_components", 30))
        log(f"variant: {args.variant}  reduction={cfg.reduction}  results -> {v['results_dir']}/")

    if args.stage == "all":
        for s in stages.STAGES:
            _run_stage(s, cfg, args)
        return 0

    if args.stage not in stages.STAGES:
        die(f"unknown stage '{args.stage}'. Known: {', '.join(stages.STAGES)}")
    _run_stage(args.stage, cfg, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
