#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import itertools
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

# utils.config

from utils.config import config as base_config  # dict】
from utils.data_loader import load_data
from optimiser.GAOptimiser import GAOptimiser

# -----------------------------
# Tools
# -----------------------------

def time_stamp():
    return time.strftime("%Y%m%d-%H%M%S")

def _to_list_or_single(v):
    return v if isinstance(v, list) else [v]

def _pct_to_float(x) -> float:
    """Accepts 0.3, '0.3', '30%', 30 and returns 0.30 as float."""
    if x is None:
        return 1.0
    if isinstance(x, (int, float)):
        xf = float(x)
        return xf/100.0 if xf > 1 else xf
    s = str(x).strip()
    if s.endswith('%'):
        return float(s[:-1]) / 100.0
    xf = float(s)
    return xf/100.0 if xf > 1 else xf

def _as_scalar(val, default=None):
    if isinstance(val, list):
        return val[0] if val else default
    return val if val is not None else default

def build_configs_for_sweep(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand config into a list of configs by sweeping:
      - method_mode
      - gene_space_top_pct (accepts 0.2, "0.2", 20, "20%")
      - stop_criteria (accepts "saturate_100", 100, "300", etc.)
    Each emitted config uses a *single* stop criterion as a list: ["saturate_X"].
    """
    def _to_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        # allow comma-separated strings
        if isinstance(v, str) and "," in v:
            return [x.strip() for x in v.split(",") if x.strip()]
        return [v]

    def _norm_stop(x) -> str:
        # normalize to "saturate_<N>"
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("saturate_"):
                return s
            if s.isdigit():
                return f"saturate_{int(s)}"
            return s
        if isinstance(x, (int, float)):
            return f"saturate_{int(x)}"
        # fallback
        return str(x)

    modes  = _to_list(cfg.get("method_mode", "mixed"))
    tops   = _to_list(cfg.get("gene_space_top_pct", 0.20))
    stops  = _to_list(cfg.get("stop_criteria", ["saturate_100"]))

    out: List[Dict[str, Any]] = []
    for mode, top, stop in itertools.product(modes, tops, stops):
        c = dict(cfg)  # shallow copy
        c["method_mode"] = mode
        c["gene_space_top_pct"] = _pct_to_float(top)  # -> float in [0,1]
        c["stop_criteria"] = [_norm_stop(stop)]       # keep type as list for GA
        out.append(c)
    return out

def run_one(cfg: Dict[str, Any]) -> Path:
    data = load_data()
    xy_all = data["xy_all"]
    incident_freq = data["incident_freq"]
    time_matrix = data["time_matrix"]

    gene_space = np.flatnonzero(incident_freq.ravel() > 0).astype(int)

    cfg = dict(cfg)  # do not mutate caller's dict

    # --- normalize the two key inputs to scalars (already done in build_configs_for_sweep, but keep defensive) ---
    cfg["method_mode"] = _as_scalar(cfg.get("method_mode"), "mixed")
    # NEW: gene_space_top_pct (float in [0,1])
    cfg["gene_space_top_pct"] = _pct_to_float(_as_scalar(cfg.get("gene_space_top_pct"), 0.20))
    top_pct = float(cfg["gene_space_top_pct"])
    init_mode = cfg["method_mode"]

    # start optimiser
    opt = GAOptimiser(data=data, config=cfg, gene_space=gene_space)

    # optional baseline
    start_layout = None
    start_layout_path = cfg.get("start_layout_path")
    if start_layout_path and Path(start_layout_path).exists():
        start_layout = np.load(start_layout_path)

    # record init params into opt.config (saved later with results)
    opt.config["init_mode"] = init_mode
    opt.config["init_top_pct"] = top_pct

    # build initial population (use *top_pct* from gene_space_top_pct)
    init_pop = opt.build_initial_population(
        base_layout=start_layout,
        pop_size=int(cfg["sol_per_pop"]),
        mode=init_mode,   # "mixed" | "single_swap" | "random"
        #min_dist=float(cfg.get("min_station_spacing", 3000.0)),
        min_dist=None,
        enforce_spacing=None,
        n_single_swap_from_base=int(cfg.get("n_single_swap_seeds", 60)),
        alpha=float(cfg.get("seed_alpha", 1.0)),
        uniform_mix_ratio=float(cfg.get("seed_uniform_mix_ratio", 0.1)),
        top_pct=top_pct,
        rng=opt.rng,      # persistent RNG for reproducibility
    )

    best_solution, best_incidents, best_pct, ga = opt.run_single(
        start_layout=start_layout,
        initial_population=init_pop,
        plot=True,
        verbose=True,
    )

    out_dir_root = cfg.get("out_dir", "outputs")
    out_dir = opt.save_results(
        run_dir_root=out_dir_root,
        run_label=None,
        best_solution=best_solution,
        best_incidents=best_incidents,
        best_pct=best_pct,
        ga=ga,
        start_layout=start_layout,
        detail=None,
        plot_map=True,
    )
    print(f"[DONE] Saved to: {out_dir}")
    return out_dir

# -----------------------------
# main()
# -----------------------------
def main():
    cfg_all = dict(base_config)
    # parameters loop: False —— single run
    #                  True —— sweep
    sweep = bool(cfg_all.get("sweep_parameters", True))

    if sweep:
        cfg_list = build_configs_for_sweep(cfg_all)
        print(f"[INFO] Sweeping {len(cfg_list)} configurations...")
    else:
        cfg = dict(cfg_all)
        if isinstance(cfg.get("method_mode"), list):
            cfg["method_mode"] = cfg["method_mode"][0]
        if isinstance(cfg.get("gene_space_top_pct"), list):
            cfg["gene_space_top_pct"] = cfg["gene_space_top_pct"][0]
        cfg_list = [cfg]
        print("[INFO] Single run...")

    for i, cfg in enumerate(cfg_list, 1):
        mode = cfg.get('method_mode')
        top  = _pct_to_float(cfg.get('gene_space_top_pct'))
        print(f"\n=== Run {i}/{len(cfg_list)} | mode={mode} | top_pct={top:.2%} ===")
        run_one(cfg)


if __name__ == "__main__":
    main()