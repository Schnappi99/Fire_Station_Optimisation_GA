# feasibility_study/methods/best_k.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm
import time

def demand_weighted_single_swap(
    *,
    evaluator,
    current_layout_idx: np.ndarray,
    iterations: int,
    candidate_cells: np.ndarray,
    MIN_DIST: float = 3000.0,
    K_best: int = 30,
    epsilon: float = 1e-6,
    alpha: float = 1.0,
    uniform_mix_ratio: float = 0.0,
    mutual_exclusion: bool = True,
    random_state: Optional[int] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    candidate_cells = np.asarray(candidate_cells, dtype=int)
    layout = np.asarray(current_layout_idx, dtype=int).copy()
    k = layout.size

    # 基于候选格的需求权重
    incident_freq = evaluator.incident_freq.ravel().astype(float)
    w = np.maximum(incident_freq[candidate_cells], epsilon) ** float(alpha)
    if uniform_mix_ratio > 0.0:
        lam = 1.0 - float(uniform_mix_ratio)
        w = lam * w + (1.0 - lam) * 1.0
    base_prob = w / w.sum()

    pos_in_candidates = {int(c): i for i, c in enumerate(candidate_cells)}

    baseline_fitness, _, _ = evaluator.evaluate_layout(layout)
    records: list[dict] = []

    xy_all = evaluator.xy_all  # (N, 2)
    t0 = time.time()

    # tqdm time progress
    iterator = range(1, int(iterations) + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Single-swap random walk", ncols=80)

    for it in iterator:
        s = int(rng.integers(0, k))
        old_cell = int(layout[s])
        allowed = np.ones(candidate_cells.size, dtype=bool)

        if mutual_exclusion:
            occ = set(int(x) for x in layout.tolist())
            occ.discard(old_cell)
            for oc in occ:
                j = pos_in_candidates.get(oc)
                if j is not None:
                    allowed[j] = False

        if k > 1:
            others = np.delete(layout, s)
            if others.size > 0:
                cand_xy = xy_all[candidate_cells]
                others_xy = xy_all[others]
                dmin = np.min(
                    np.linalg.norm(cand_xy[:, None, :] - others_xy[None, :, :], axis=2),
                    axis=1
                )
                allowed &= (dmin >= float(MIN_DIST))

        j_old = pos_in_candidates.get(old_cell)
        if j_old is not None:
            allowed[j_old] = False

        allowed_count = int(allowed.sum())
        if allowed_count == 0:
            records.append({
                "iter": it, "fitness": np.nan,
                "moved_station": s, "old_cell": old_cell,
                "new_cell": None, "layout": layout.copy(),
                "allowed_count": 0, "note": "skip_no_feasible_target",
            })
            continue

        new_cell = _choose_best_of_k(
            evaluator=evaluator,
            layout=layout, station_idx=s,
            candidate_cells=candidate_cells,
            allowed_mask=allowed,
            base_prob=base_prob,
            K=int(K_best), rng=rng
        )
        if new_cell == -1:
            records.append({
                "iter": it, "fitness": np.nan,
                "moved_station": s, "old_cell": old_cell,
                "new_cell": None, "layout": layout.copy(),
                "allowed_count": allowed_count, "note": "skip_sampling_failed",
            })
            continue

        layout[s] = new_cell
        f, _, _ = evaluator.evaluate_layout(layout)
        records.append({
            "iter": it, "fitness": float(f),
            "moved_station": s, "old_cell": old_cell,
            "new_cell": int(new_cell), "layout": layout.copy(),
            "allowed_count": allowed_count,
        })

    total_time = time.time() - t0
    print(f"\nCompleted {iterations} iterations in {total_time:.2f} seconds "
          f"(avg {total_time/iterations:.3f} s/iter)")

    df = pd.DataFrame.from_records(records)
    df.attrs["baseline_fitness"] = float(baseline_fitness)
    return df


def _choose_best_of_k(
    *,
    evaluator,
    layout: np.ndarray,
    station_idx: int,
    candidate_cells: np.ndarray,
    allowed_mask: np.ndarray,
    base_prob: np.ndarray,
    K: int,
    rng: np.random.Generator
) -> int:
    probs = np.zeros_like(base_prob)
    probs[allowed_mask] = base_prob[allowed_mask]
    s = float(probs.sum())
    if s <= 0:
        return -1
    probs = probs / s

    allowed_idx = np.flatnonzero(allowed_mask)
    if allowed_idx.size == 0:
        return -1

    K_eff = min(int(K), allowed_idx.size)
    sampled_pos = rng.choice(allowed_idx, size=K_eff, replace=False, p=probs[allowed_idx])

    best_cell = -1
    best_f = -np.inf
    old_cell = int(layout[station_idx])

    for pos in sampled_pos:
        cand_cell = int(candidate_cells[pos])
        layout[station_idx] = cand_cell
        f, _, _ = evaluator.evaluate_layout(layout)
        if f > best_f:
            best_f = f
            best_cell = cand_cell

    layout[station_idx] = old_cell
    return best_cell