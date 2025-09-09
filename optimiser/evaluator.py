# optimiser/weighted.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


def compute_station_probs(layout_idx: np.ndarray) -> np.ndarray:
    """
    ownership 权重: 哪个站负责的 demand 多 → 更容易被抽到移动。
    layout_idx: shape=(k,), 每个元素是 cell_id（即在 _time_matrix 的列/索引空间能对应上）
    """
    # 取出这些站对应的时间子矩阵：sel = (M, k)
    # 这里假设 _time_matrix 的列索引能按 cell_id 直接取到；若你的列是“按全部cells顺序”，
    # 就做一次 pos 映射把 layout_idx 转为列号。
    sel = _time_matrix[:, layout_idx]  # (M, k)
    arg = np.argmin(sel, axis=1)       # 每个需求格最近的站（0..k-1）
    w_st = np.bincount(arg, weights=_incident_freq, minlength=layout_idx.size).astype(float)  # 各站负责的需求权重和
    if w_st.sum() <= 0:
        w_st = np.ones_like(w_st, dtype=float)
    return w_st / w_st.sum()


def build_allowed_mask(layout: np.ndarray,
                       s: int,
                       candidate_cells: np.ndarray,
                       pos_in_candidates: dict[int, int],
                       mutual_exclusion: bool,
                       MIN_DIST: float,
                       _xy_all: np.ndarray) -> np.ndarray:
    """返回候选格的可行性布尔掩码（满足互斥、最小间距、不原地）"""
    allowed = np.ones(candidate_cells.shape[0], dtype=bool)

    # 互斥（除开该站当前占用的 old_cell）
    old_cell = layout[s]
    if mutual_exclusion:
        occ = set(layout.tolist()); occ.discard(old_cell)
        for oc in occ:
            j = pos_in_candidates.get(oc)
            if j is not None:
                allowed[j] = False

    # 最小间距
    if layout.size > 1:
        others = np.delete(layout, s)
        if others.size > 0:
            cand_xy = _xy_all[candidate_cells]   # (C, 2)
            others_xy = _xy_all[others]          # (k-1, 2)
            # 每个候选点到其他站的最近距离
            dmin = np.min(np.linalg.norm(cand_xy[:, None, :] - others_xy[None, :, :], axis=2), axis=1)
            allowed &= (dmin >= float(MIN_DIST))

    # 不允许留在原地
    j_old = pos_in_candidates.get(old_cell)
    if j_old is not None:
        allowed[j_old] = False

    return allowed


def sample_new_cell(candidate_cells: np.ndarray,
                    allowed_mask: np.ndarray,
                    base_prob: np.ndarray,
                    rng: np.random.Generator) -> int | None:
    """
    在 allowed 的候选里，按 base_prob 加权抽样新 cell_id。
    兜底：若权重和为 0，则在 allowed 里均匀抽；若 allowed 全 False，返回 None。
    """
    if not allowed_mask.any():
        return None

    probs = np.zeros_like(base_prob, dtype=float)
    probs[allowed_mask] = base_prob[allowed_mask]
    ssum = probs.sum()
    if ssum <= 0:
        # 回退：均匀抽 allowed 区域
        idxs = np.flatnonzero(allowed_mask)
        j = rng.choice(idxs)
        return int(candidate_cells[j])
    else:
        probs = probs / ssum
        j = rng.choice(np.arange(candidate_cells.shape[0]), p=probs)
        return int(candidate_cells[j])