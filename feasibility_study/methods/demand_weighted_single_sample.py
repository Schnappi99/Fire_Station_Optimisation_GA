from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from tqdm import tqdm
import time

def demand_weighted_single_sample(
    *,
    evaluator,                              # utils.evaluator.Evaluator
    current_layout_idx: np.ndarray,         # baseline layout (k,)
    iterations: int,
    candidate_cells: np.ndarray,            # feasible cells (e.g. incident_freq > 0)
    MIN_DIST: float = 3000.0,               # (b) minimal spacing between stations
    epsilon: float = 1e-6,
    alpha: float = 1.0,                     # demand weighting exponent
    uniform_mix_ratio: float = 0.0,         # mixture ratio with uniform distribution
    mutual_exclusion: bool = True,          # (c) disallow overlap with other stations
    accept_if_better: bool = False,         # accept move only if fitness improves
    random_state: Optional[int] = None,
    show_progress: bool = True,
    max_resample: int = 5,                  # maximum resampling attempts if very few candidates
) -> pd.DataFrame:

    """
    Demand-weighted single-sample random walk under constraints:
      (a) At each step, only one station is changed while others remain fixed;
      (b) New station must keep >= MIN_DIST distance from all other stations;
      (c) New station cannot be at the same location as its old position,
          and cannot overlap with any other stations (if mutual_exclusion=True).

    Process (per iteration):
      1) Randomly pick one station s
      2) Build a feasible candidate set under (b) and (c)
      3) Sample exactly 1 new cell from the candidate set according to demand weights
      4) Evaluate fitness:
         - If accept_if_better=True: accept only if fitness improves
         - Else: always accept the move

    Returns
    -------
    pd.DataFrame with columns:
      iter, fitness, moved_station, old_cell, new_cell, layout, allowed_count
    DataFrame.attrs["baseline_fitness"] = incidents_served of the baseline layout
    """
    rng = np.random.default_rng(random_state)
    candidate_cells = np.asarray(candidate_cells, dtype=int)
    layout = np.asarray(current_layout_idx, dtype=int).copy()
    k = layout.size

    # Demand weights (with optional uniform mixture)
    inc = evaluator.incident_freq.ravel().astype(float)
    w = np.maximum(inc[candidate_cells], epsilon) ** float(alpha)
    if uniform_mix_ratio > 0.0:
        lam = 1.0 - float(uniform_mix_ratio)
        w = lam * w + (1.0 - lam) * 1.0
    base_prob = w / w.sum()

    # Map from cell -> index position in candidate_cells
    pos_in_candidates = {int(c): i for i, c in enumerate(candidate_cells)}
    baseline_fitness, _, _ = evaluator.evaluate_layout(layout)

    xy_all = evaluator.xy_all
    records: list[dict] = []
    t0 = time.time()

    iterator = range(1, int(iterations) + 1)
    if show_progress:
        iterator = tqdm(iterator, desc="Single-sample (demand-weighted)", ncols=80)

    for it in iterator:
        s = int(rng.integers(0, k))
        old_cell = int(layout[s])

        # Initial feasible set: all candidates allowed
        allowed = np.ones(candidate_cells.size, dtype=bool)

        # Mutual exclusion: forbid cells already occupied by other stations
        if mutual_exclusion:
            occ = set(int(x) for x in layout.tolist())
            occ.discard(old_cell)
            for oc in occ:
                j = pos_in_candidates.get(oc)
                if j is not None:
                    allowed[j] = False

        # Distance constraint: enforce min spacing to all other stations
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

        # Disallow staying at the same location
        j_old = pos_in_candidates.get(old_cell)
        if j_old is not None:
            allowed[j_old] = False

        allowed_count = int(allowed.sum())
        if allowed_count == 0:
            records.append({
                "iter": it, "fitness": np.nan,
                "moved_station": s, "old_cell": old_cell, "new_cell": None,
                "layout": layout.copy(), "allowed_count": 0,
                "note": "skip_no_feasible_target",
            })
            continue

        # Sample exactly 1 new cell from allowed candidates
        new_cell = -1
        allowed_idx = np.flatnonzero(allowed)
        probs = np.zeros_like(base_prob)
        probs[allowed] = base_prob[allowed]
        sprob = float(probs.sum())
        if sprob > 0:
            probs = probs / sprob
            for _ in range(max_resample):
                pos = int(rng.choice(allowed_idx, replace=False, p=probs[allowed_idx]))
                cand_cell = int(candidate_cells[pos])
                # cand_cell 已满足 allowed 约束，无需再次判断
                new_cell = cand_cell
                break

        if new_cell == -1:
            records.append({
                "iter": it, "fitness": np.nan,
                "moved_station": s, "old_cell": old_cell, "new_cell": None,
                "layout": layout.copy(), "allowed_count": allowed_count,
                "note": "skip_sampling_failed",
            })
            continue

        # Acceptance rule
        if accept_if_better:
            # Evaluate old vs new fitness and accept only if improved
            old_f, _, _ = evaluator.evaluate_layout(layout)
            layout[s] = new_cell
            new_f, _, _ = evaluator.evaluate_layout(layout)
            if new_f < old_f:
                # Revert if worse
                layout[s] = old_cell
                f_used = old_f
                records.append({
                    "iter": it, "fitness": float(f_used),
                    "moved_station": s, "old_cell": old_cell,
                    "new_cell": None, "layout": layout.copy(),
                    "allowed_count": allowed_count, "note": "reject_worse",
                })
                continue
            else:
                f_used = new_f
        else:
            # Always move regardless of improvement
            layout[s] = new_cell
            f_used, _, _ = evaluator.evaluate_layout(layout)

        records.append({
            "iter": it, "fitness": float(f_used),
            "moved_station": s, "old_cell": old_cell,
            "new_cell": int(new_cell), "layout": layout.copy(),
            "allowed_count": allowed_count,
        })

    total_time = time.time() - t0
    print(f"\nCompleted {iterations} iterations in {total_time:.2f}s "
          f"(avg {total_time/iterations:.4f}s/iter)")

    df = pd.DataFrame.from_records(records)
    df.attrs["baseline_fitness"] = float(baseline_fitness)
    return df