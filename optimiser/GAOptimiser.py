# optimiser/GAOptimiser.py
from __future__ import annotations
import time
from pathlib import Path
from typing import Optional, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygad
import json
from datetime import datetime

from utils.evaluator import Evaluator

def _parse_percent(val, default=None) -> float | None:
    """Parse values like '20%' or 0.2 into a float in [0,1]; None stays None."""
    if val is None:
        return default
    if isinstance(val, str):
        v = val.strip()
        if v.endswith("%"):
            return float(v[:-1]) / 100.0
    return float(val)

def _euclid_min_dist(xy: np.ndarray, chosen: np.ndarray, cand_idx: int) -> float:
    if chosen.size == 0:
        return np.inf
    d = np.linalg.norm(xy[chosen] - xy[cand_idx], axis=1)
    return float(d.min())


class GAOptimiser:
    """Fire station layout optimisation (Genetic Algorithm)."""

    feature_names_default = [
        "nearest_station_travel_time",
        "neighbour_frequency_per_month",
        "Agriculture - mainly crops",
        "Deciduous woodland",
        "station_count",
    ]

    def __init__(
        self,
        data: dict,
        config: dict,
        gene_space: Iterable[int],
        feature_names: Optional[list[str]] = None,
    ):
        # input data
        self.xy_all: np.ndarray = data["xy_all"]
        self.time_matrix: np.ndarray = data["time_matrix"]
        self.A_matrix: np.ndarray = data["A_matrix"]
        self.partial_features: np.ndarray = data["partial_features"]
        self.incident_freq: np.ndarray = np.asarray(data["incident_freq"], float).ravel()
        self.rf_model = data["rf_model"]

        self.config = dict(config)
        self.debug = bool(self.config.get("debug", True))
        self.gene_space = np.array(sorted({int(i) for i in gene_space}), dtype=int)
        self.feature_names = feature_names or self.feature_names_default

        # Random Number Generator: to control each random choice during the process
        self.rng = np.random.default_rng(self.config.get("random_seed", None))

        # evaluator
        self.evaluator = Evaluator(
            xy_all=self.xy_all,
            time_matrix=self.time_matrix,
            A_matrix=self.A_matrix,
            partial_features=self.partial_features,
            incident_freq=self.incident_freq,
            rf_model=self.rf_model,
            feature_names=self.feature_names,
        )

        # quick sanity
        N = self.xy_all.shape[0]
        assert self.time_matrix.shape[0] == N
        assert self.partial_features.shape[0] == N
        assert self.incident_freq.shape[0] == N
        assert self.partial_features.shape[1] == len(self.feature_names) - 2

        # logging
        self.log_dir = Path(self.config.get("log_dir", "log"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = None
        self._log: list[tuple[float, float]] = []

    # public helpers
    def evaluate_layout(self, solution: np.ndarray):
        """Proxy to Evaluator"""
        return self.evaluator.evaluate_layout(solution)

    # build initial population
    def build_initial_population(
        self,
        base_layout: Optional[np.ndarray],
        pop_size: int,
        mode: str = "mixed",                 # "mixed" | "single_swap" | "random"
        # dist constriant
        min_dist: Optional[float] = None,    # None;
        enforce_spacing: Optional[bool] = None,  # 若不为 None，则覆盖 min_dist 的开关
        n_single_swap_from_base: int = 0,
        alpha: float = 1.0,
        uniform_mix_ratio: float = 0.0,
        top_pct: float = 0.10,
        rng=None,
    ) -> np.ndarray:

        """
        Initial population builder with TOGGLED spacing constraint.

        - mixed: baseline + n_single_swap neighbors + demand-weighted random
        - single_swap: only single-swap neighbors around baseline (plus baseline if given)
        - random: demand-weighted random

        Spacing control:
          * If enforce_spacing is None:
                spacing_enabled = (min_dist is not None)
            else:
                spacing_enabled = bool(enforce_spacing)
                if spacing_enabled and min_dist is None:
                    # 从 config 兜底
                    min_dist = float(self.config.get("min_dist", 3000.0))
        """
        rng = rng or self.rng
        k = int(self.config["num_stations"])
        feasible = np.asarray(self.gene_space, dtype=int)
        freq = np.asarray(self.incident_freq, dtype=float)
        xy = np.asarray(self.xy_all)

        if feasible.size < k:
            raise ValueError(f"gene_space too small: {feasible.size} < num_stations={k}")

        # ---- 解析 spacing 开关 ----
        if enforce_spacing is None:
            spacing_enabled = (min_dist is not None)
        else:
            spacing_enabled = bool(enforce_spacing)
            if spacing_enabled and min_dist is None:
                min_dist = float(self.config.get("min_dist", 3000.0))

        # ---------- helpers ----------
        def _weights(ids: np.ndarray) -> np.ndarray:
            # demand^alpha, optionally mixed with uniform
            w = np.maximum(freq[ids], 1e-12) ** float(alpha)
            if uniform_mix_ratio > 0.0:
                lam = 1.0 - float(uniform_mix_ratio)
                w = lam * w + (1.0 - lam)
            s = w.sum()
            return w / s if s > 0 else np.ones(ids.size) / ids.size

        def _top_pool(pool: np.ndarray) -> np.ndarray:
            if top_pct is None or top_pct <= 0.0 or top_pct >= 1.0:
                return pool
            vals = freq[pool]
            q = np.quantile(vals, 1.0 - float(top_pct))
            top = pool[vals >= q]
            return top if top.size > 0 else pool

        # ---- 无间距：一次性“加权不放回抽样” ----
        def sample_layout_no_spacing() -> np.ndarray:
            pool = _top_pool(feasible)
            p = _weights(pool)
            chosen = rng.choice(pool, size=k, replace=False, p=p)
            return np.asarray(chosen, dtype=int)

        # ---- 有间距：逐位贪心 + 动态剔除近邻候选----
        def sample_layout_with_spacing() -> np.ndarray:
            chosen: list[int] = []
            # 初始池：可选 top-p 过滤
            pool = _top_pool(feasible)
            # 动态概率（按池中剩余候选更新）
            while len(chosen) < k:
                if pool.size == 0:
                    # 如果空间约束太紧导致选不满，则回退到“只保证不重复”补齐
                    remaining = np.setdiff1d(feasible, np.asarray(chosen, int), assume_unique=False)
                    p_remain = _weights(remaining)
                    need = k - len(chosen)
                    extra = rng.choice(remaining, size=need, replace=False, p=p_remain)
                    chosen.extend(list(map(int, extra)))
                    break

                p = _weights(pool)
                cand = int(rng.choice(pool, p=p))

                # 距离检查：与已选最近距离 >= min_dist 才收下
                if len(chosen) == 0 or _euclid_min_dist(xy, np.asarray(chosen, int), cand) >= float(min_dist):
                    chosen.append(cand)
                    # 从池中剔除 cand 以及所有与 cand 过近的点（加速后续成功率）
                    if min_dist is not None and min_dist > 0:
                        # 计算 cand 到池中所有点的距离并剔除过近者
                        dist_to_cand = np.linalg.norm(xy[pool] - xy[cand], axis=1)
                        pool = pool[dist_to_cand >= float(min_dist)]
                    # 同时也要剔除已选（避免重复）
                    if pool.size > 0:
                        pool = pool[~np.isin(pool, np.asarray(chosen, int))]
                else:
                    # cand 太近：从池中移除它，继续
                    pool = pool[pool != cand]

            return np.asarray(chosen, dtype=int)

        def single_swap_from(layout: np.ndarray) -> np.ndarray:
            """
            Change exactly one station; ensure no-dup; spacing optional.
            """
            child = np.asarray(layout, dtype=int).copy()
            i = int(rng.integers(0, k))

            # 候选 = 不在 child 的所有点（可选 top-p）
            pool = feasible[~np.isin(feasible, child)]
            pool = _top_pool(pool)
            if pool.size == 0:
                return child  # 没有可替换的候选，直接返回

            if not spacing_enabled:
                p = _weights(pool)
                child[i] = int(rng.choice(pool, p=p))
                return child

            # spacing 启用：先过滤出满足 min_dist 的候选
            keep_mask = []
            others = np.delete(child, i)  # 除了位置 i 之外的所有已选
            for cand in pool:
                if _euclid_min_dist(xy, others, cand) >= float(min_dist):
                    keep_mask.append(True)
                else:
                    keep_mask.append(False)
            pool_ok = pool[np.array(keep_mask, dtype=bool)]

            # 若没有满足 spacing 的候选，则回退到“仅不重复”
            pool_final = pool_ok if pool_ok.size > 0 else pool
            p = _weights(pool_final)
            child[i] = int(rng.choice(pool_final, p=p))
            return child

        # generate the population
        pop: list[np.ndarray] = []

        # include baseline if applicable
        if mode in ("mixed", "single_swap") and base_layout is not None:
            base_layout = np.asarray(base_layout, dtype=int)
            if base_layout.size != k:
                raise ValueError(f"base_layout length {base_layout.size} != k={k}")
            if len(np.unique(base_layout)) != k:
                raise ValueError("base_layout contains duplicate indices.")
            # 若启用 spacing，保证 baseline 自身也满足（否则提示或容忍）
            if spacing_enabled:
                ok = True
                for a in range(k):
                    others = np.delete(base_layout, a)
                    if _euclid_min_dist(xy, others, base_layout[a]) < float(min_dist):
                        ok = False
                        break
                if not ok:
                    # 这里选择“容忍并放入”，也可以改为 raise 或自动修正
                    pass
            pop.append(base_layout)

        if mode in ("mixed", "single_swap") and base_layout is not None and n_single_swap_from_base > 0:
            for _ in range(int(n_single_swap_from_base)):
                pop.append(single_swap_from(base_layout))

        sampler = sample_layout_with_spacing if spacing_enabled else sample_layout_no_spacing

        if mode in ("mixed", "random"):
            while len(pop) < pop_size:
                pop.append(sampler())

        # fill (safety)
        while len(pop) < pop_size:
            pop.append(sampler())

        return np.asarray(pop[:pop_size], dtype=int)

    # callbacks
    def _on_start(self, ga: pygad.GA):
        """Triggered once before the GA starts running."""
        import time
        self._t0 = time.time()
        self._last_gen_t = self._t0
        self._log.clear()
        self._per_gen = []

        print(f"Optimising with {ga.num_genes} stations from {len(self.gene_space)} feasible locations...")

        # Optional: print whether the spacing constraint is enabled
        spacing_enabled = getattr(self, "spacing_enabled", None)
        min_dist = getattr(self, "min_dist", None)
        if spacing_enabled is not None:
            print(f"[INIT_POP] Spacing constraint: {'ON' if spacing_enabled else 'OFF'} (min_dist={min_dist})")

    def _on_generation(self, ga: pygad.GA):
        """Triggered once after each generation is completed."""
        import time
        gen = int(getattr(ga, "generations_completed", 0))

        # Measure time taken for this generation
        now = time.time()
        dt = now - (self._last_gen_t or now)
        self._last_gen_t = now

        # Retrieve best fitness in this generation
        try:
            best_fit = ga.best_solution()[1]
        except Exception:
            best_fit = float("nan")

        # Estimate number of offspring created (via crossover + mutation)
        offspring = 0
        try:
            if hasattr(ga,
                       "last_generation_offspring_crossover") and ga.last_generation_offspring_crossover is not None:
                offspring += len(ga.last_generation_offspring_crossover)
            if hasattr(ga, "last_generation_offspring_mutation") and ga.last_generation_offspring_mutation is not None:
                offspring += len(ga.last_generation_offspring_mutation)
        except Exception:
            pass

        # Record per-generation data
        self._per_gen.append({
            "gen": gen,
            "dt_sec": float(dt),
            "best_fitness": float(best_fit) if best_fit == best_fit else None,  # NaN → None
            "offspring_count": int(offspring),
        })

        # Log every 10 generations and print progress
        if gen % 10 == 0:
            try:
                elapsed = time.time() - (self._t0 or time.time())
                self._log.append((elapsed, float(best_fit)))
                print(f"Gen {gen:>4} | Best {best_fit:,.4f}")
            except Exception:
                pass

    def _on_stop(self, ga: pygad.GA, last_pop_fitness):
        """Triggered once when the GA finishes or meets the stop criteria."""
        import time, json, csv, pandas as pd, matplotlib.pyplot as plt

        # Get final best fitness
        try:
            final_best = ga.best_solution()[1]
        except Exception:
            final_best = float("nan")

        # Compute total runtime and average time per generation
        total_time = time.time() - (self._t0 or time.time())
        gens = max(1, int(getattr(ga, "generations_completed", 0)))
        avg_time = total_time / gens

        # Identify reason for stopping
        stop_reason = "completed_all_generations"
        if getattr(ga, "stop_criteria", None):
            stop_reason = "stopped_by_criteria"

        # === Save time–fitness log (existing behaviour) ===
        if self._log:
            log_df = pd.DataFrame(self._log, columns=["Time_s", "Best_fitness"])
            log_df.to_csv(self.log_dir / "log.csv", index=False)

            plt.figure(figsize=(8, 4.5))
            plt.plot(log_df["Time_s"], log_df["Best_fitness"], marker="o")
            plt.xlabel("Time (s)")
            plt.ylabel("Best Fitness")
            plt.title("GA Fitness Over Time")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(self.log_dir / "fitness_curve.png", dpi=140)
            plt.close()

        # === Save per-generation performance ===
        if self._per_gen:
            with open(self.log_dir / "per_generation.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["gen", "dt_sec", "best_fitness", "offspring_count"])
                writer.writeheader()
                writer.writerows(self._per_gen)

        # === Save overall timing summary ===
        summary = {
            "generations_completed": gens,
            "sol_per_pop": int(getattr(ga, "sol_per_pop", 0)),
            "keep_elitism": int(getattr(ga, "keep_elitism", 0)),
            "keep_parents": int(getattr(ga, "keep_parents", -1)),
            "num_parents_mating": int(getattr(ga, "num_parents_mating", 0)),
            "total_time_sec": float(total_time),
            "avg_time_per_gen_sec": float(avg_time),
            "stop_reason": stop_reason,
            "final_best_fitness": None if final_best != final_best else float(final_best),
        }

        with open(self.log_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_generation": self._per_gen},
                      f, ensure_ascii=False, indent=2)

        # Final console output
        print(f"GA stopped. Final best: {final_best:,.4f}")
        print(f"[GA] Total {gens} generations, total_time={total_time:.2f}s, avg/gen={avg_time:.3f}s")
        print(f"[GA] Wrote timing.json, per_generation.csv, and log.csv to: {self.log_dir}")


    # mutation: single-swap + spacing
    def _mutation_single_swap_dist(self, offspring, ga_instance=None):
        """
        Each offspring mutates EXACTLY ONE gene;
        ensure ≥ min_dist & no duplicates;
        demand-weighted Top-p%.
        """
        rng = getattr(self, "rng", None)
        if rng is None:
            rng = np.random.default_rng(self.config.get("random_seed", None))
            self.rng = rng
        #rng = np.random.default_rng(self.config.get("random_seed", None))
        k = offspring.shape[1]

        # spacing control (toggle)
        enforce_spacing = self.config.get("enforce_spacing", None)
        min_dist = self.config.get("min_station_spacing", None)
        if enforce_spacing is None:
            spacing_enabled = (min_dist is not None)
        else:
            spacing_enabled = bool(enforce_spacing)
        if spacing_enabled and min_dist is None:
            min_dist = 3000.0
        min_dist = None if min_dist is None else float(min_dist)

        # top-p control (supports "20%")
        top_pct = _parse_percent(self.config.get("gene_space_top_pct", "20%"), default=0.2)
        if top_pct is None or top_pct <= 0.0 or top_pct >= 1.0:
            top_pct = 0.0  # treat as no filtering

        mut_prob = float(self.config.get("mutation_probability", 1.0))
        xy, freq, feasible = self.xy_all, self.incident_freq, self.gene_space

        def too_close_arr(cand: int, arr: np.ndarray) -> bool:
            if not spacing_enabled or arr.size == 0:
                return False
            d = np.linalg.norm(xy[arr] - xy[cand], axis=1)
            return float(d.min()) < float(min_dist)

        def p_of(ids: np.ndarray):
            w = freq[ids].astype(float)
            s = w.sum()
            return (w / s) if s > 0 else None

        # precompute Top-p
        pool_all = feasible
        if top_pct > 0:
            vals = freq[pool_all]
            q = np.quantile(vals, 1.0 - top_pct)
            top_ids = pool_all[vals >= q]
            if top_ids.size < max(1, min(10, k)):
                top_ids = pool_all
        else:
            top_ids = pool_all

        for r in range(offspring.shape[0]):
            if rng.random() >= mut_prob:
                continue
            i = int(rng.integers(0, k))
            pool = top_ids[~np.isin(top_ids, offspring[r])]
            if pool.size == 0:
                pool = top_ids
            p = p_of(pool)  # or your weighted version

            old_val = int(offspring[r, i])
            for _ in range(200):
                cand = int(rng.choice(pool, p=p))
                ok_spacing = (not spacing_enabled) or (not too_close_arr(cand, np.delete(offspring[r], i)))
                if cand != old_val and ok_spacing:
                    offspring[r, i] = cand
                    break
        return offspring

    def _make_run_id(self, extra_label: str | None = None) -> str:
        """
        Compose a readable run_id from key params + timestamp (robust to list values).
        """
        ts = datetime.now().strftime("%d-%H%M%S")

        pieces = []

        # top_pct
        top_val = self.config.get("gene_space_top_pct", None)
        pieces.append(f"top{_parse_percent(top_val)}")

        # distance
        if self.config.get("min_station_spacing") is not None:
            pieces.append(f"dist{int(self.config['min_station_spacing'])}")

        # mutation probability
        mut_val = self.config.get("mutation_probability", None)
        if mut_val is not None:
            try:
                pieces.append(f"mut{float(mut_val):.2f}")
            except Exception:
                pass

        # extra label if provided
        if extra_label:
            pieces.append(extra_label)

        pieces.append(ts)
        return "_".join(pieces)

    def save_results(
            self,
            *,
            run_dir_root: str | Path = "outputs",
            run_label: str | None = None,
            best_solution: np.ndarray,
            best_incidents: float,
            best_pct: float,
            ga: "pygad.GA",
            start_layout: np.ndarray | None = None,
            detail: pd.DataFrame | None = None,
            plot_map: bool = True,
    ) -> Path:
        """
        Save minimal GA summary and key config parameters in a simple JSON file.
        """
        run_dir_root = Path(run_dir_root)
        run_id = self._make_run_id(run_label)
        out_dir = run_dir_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Compute timing info
        total_time_sec = float(getattr(self, "_total_time", 0.0))
        if hasattr(self, "_t0"):
            total_time_sec = time.time() - self._t0
        generations_completed = int(getattr(ga, "generations_completed", 0))
        avg_time_per_gen = total_time_sec / generations_completed if generations_completed > 0 else np.nan

        # Safely extract final best fitness
        final_best_fitness = float("nan")
        try:
            if hasattr(ga, "best_solution"):
                bs = ga.best_solution()  # (solution, fitness, index)
                if isinstance(bs, (list, tuple)) and len(bs) >= 2:
                    final_best_fitness = float(bs[1])
        except Exception:
            pass

        # Build concise summary JSON
        summary = {
            "summary": {
                "generations_completed": generations_completed,
                "sol_per_pop": int(getattr(ga, "sol_per_pop", self.config.get("sol_per_pop", 0))),
                "keep_elitism": int(self.config.get("keep_elitism", 0)),
                "keep_parents": int(self.config.get("keep_parents", 0)),
                "num_parents_mating": int(self.config.get("num_parents_mating", 0)),
                "total_time_sec": total_time_sec,
                "avg_time_per_gen_sec": avg_time_per_gen,
                "stop_reason": str(getattr(ga, "stop_reason", "stopped_by_criteria")),
                "final_best_fitness": final_best_fitness,
            },
            "parameters": {
                "method_mode": self.config.get("init_mode", self.config.get("method_mode")),
                "gene_space_top_pct": self.config.get("init_top_pct", self.config.get("gene_space_top_pct")),
                "stop_criteria": self.config.get("stop_criteria", ["saturate_100"]),
            },
        }

        # Save concise JSON
        out_json = out_dir / f"summary_{run_id}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        # # Optional map plot
        # if plot_map:
        #     try:
        #         fig, ax = plt.subplots(figsize=(8, 8))
        #         ax.scatter(self.xy_all[:, 0], self.xy_all[:, 1], s=2, alpha=0.2, label="Grid centroids")
        #         sel_xy = self.xy_all[np.asarray(best_solution, int)]
        #         ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=60, marker="*", label="Selected stations")
        #         ax.set_aspect("equal");
        #         ax.legend()
        #         ax.set_title("Optimised Fire Station Locations")
        #         fig.tight_layout()
        #         fig.savefig(out_dir / f"map_{run_id}.png", dpi=150)
        #         plt.close(fig)
        #     except Exception as e:
        #         print(f"[WARN] plot_map failed: {e}")
        #
        # print(f"[SAVE] Summary saved to: {out_json.resolve()}")
        return out_dir

    def save_results_old(
            self,
            *,
            run_dir_root: str | Path = "outputs",
            run_label: str | None = None,
            best_solution: np.ndarray,
            best_incidents: float,
            best_pct: float,
            ga: "pygad.GA",
            start_layout: np.ndarray | None = None,
            detail: pd.DataFrame | None = None,
            plot_map: bool = True,
    ) -> Path:
        """
        Save all artifacts for this run under outputs/<run_id>/.
        Returns the created run directory Path.
        """
        run_dir_root = Path(run_dir_root)
        run_id = self._make_run_id(run_label)
        out_dir = run_dir_root / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # config
        with open(out_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

        # save best solution
        np.save(out_dir / "best_solution.npy", np.asarray(best_solution, int))
        pd.Series(best_solution, name="candidate_index").to_csv(out_dir / "best_solution.csv", index=False)

        # save details
        if detail is None:
            incidents_served, pct, detail = self.evaluate_layout(np.asarray(best_solution, int))
        else:
            incidents_served, pct = float(best_incidents), float(best_pct)

        # save summary
        total_incidents = float(self.incident_freq.sum())
        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "num_stations": int(len(best_solution)),
            "best_incidents": float(incidents_served),
            "best_pct": float(pct),
            "total_incidents": float(total_incidents),
        }

        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        detail.to_csv(out_dir / "cell_detail.csv", index=False)

        # GA fitness
        if getattr(ga, "best_solutions_fitness", None) is not None:
            pd.Series(ga.best_solutions_fitness, name="best_fitness_each_gen").to_csv(
                out_dir / "best_fitness_each_gen.csv", index=False
            )

        # save log
        # _on_stop: self.log_dir points to: out_dir/"log"
        log_dir = out_dir / "log"
        log_dir.mkdir(exist_ok=True)

        if getattr(self, "_log", None):
            log_df = pd.DataFrame(self._log, columns=["Time_s", "Best_fitness"])
            log_df.to_csv(log_dir / "log.csv", index=False)
            plt.figure(figsize=(8, 4.5))
            plt.plot(log_df["Time_s"], log_df["Best_fitness"], marker="o")
            plt.xlabel("Time (s)");
            plt.ylabel("Best Fitness")
            plt.title("GA Fitness Over Time");
            plt.grid(True);
            plt.tight_layout()
            plt.savefig(log_dir / "fitness_curve.png", dpi=140);
            plt.close()

        # save map
        if plot_map:
            try:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.scatter(self.xy_all[:, 0], self.xy_all[:, 1], s=2, alpha=0.2, label="Grid centroids")
                sel_xy = self.xy_all[np.asarray(best_solution, int)]
                ax.scatter(sel_xy[:, 0], sel_xy[:, 1], s=60, marker="*", label="Selected stations")
                ax.set_aspect("equal");
                ax.legend()
                ax.set_title("Optimised Fire Station Locations")
                fig.tight_layout()
                fig.savefig(out_dir / "optimised_layout_map.png", dpi=150)
                plt.close(fig)
            except Exception as e:
                print(f"[WARN] plot_map failed: {e}")

        # baseline
        if start_layout is not None:
            base_inc, base_pct, _ = self.evaluate_layout(np.asarray(start_layout, int))
            with open(out_dir / "baseline_compare.txt", "w", encoding="utf-8") as f:
                f.write(f"Baseline:  {base_inc:,.2f} ({base_pct:.2%})\n")
                f.write(f"Optimised: {incidents_served:,.2f} ({pct:.2%})\n")
                f.write(f"Delta: +{incidents_served - base_inc:,.2f} | Δ{(pct - base_pct):.2%}\n")

        print(f"[SAVE] Results saved to: {out_dir.resolve()}")
        return out_dir

    # main run
    def run_single(
            self,
            *,
            start_layout: Optional[np.ndarray] = None,
            initial_population: Optional[np.ndarray] = None,
            plot: bool = True,
            verbose: bool = True,
    ):
        k = int(self.config["num_stations"])
        stop_criteria = tuple(self.config.get("stop_criteria", ()))

        if initial_population is not None:
            initial_population = np.asarray(initial_population, int)
            assert initial_population.ndim == 2 and initial_population.shape[1] == k, \
                "init_pop must be (sol_per_pop, num_stations)"
            # check
            if not self.config.get("allow_duplicate_genes", False):
                if any(np.unique(row).size != row.size for row in initial_population):
                    raise ValueError("init_pop has duplicates but allow_duplicate_genes=False.")
            bad = np.setdiff1d(initial_population, self.gene_space)
            if bad.size > 0:
                raise ValueError(f"init_pop contains indices outside gene_space: {bad[:10]}...")

        tmp_run_id = self._make_run_id(extra_label="running")
        self.log_dir = Path(self.config.get("out_dir", "outputs")) / tmp_run_id / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ga = pygad.GA(
            num_generations=int(self.config["generations"]),
            sol_per_pop=int(self.config["sol_per_pop"]),
            num_parents_mating=int(self.config["num_parents_mating"]),
            num_genes=k,
            gene_type=int,
            gene_space=list(self.gene_space),
            initial_population=initial_population,
            fitness_func=self.evaluator.fitness_pygad_with_ga,
            parent_selection_type=self.config["parent_selection_type"],
            K_tournament=int(self.config.get("K_tournament", 3)),
            crossover_type=self.config["crossover_type"],
            crossover_probability=float(self.config.get("crossover_probability", 0.0)),  # no crossover
            mutation_type="random",
            mutation_probability=float(self.config.get("mutation_probability", 1.0)),
            keep_elitism=int(self.config.get("keep_elitism", 2)),
            keep_parents=int(self.config.get("keep_parents", 0)),
            stop_criteria=stop_criteria,
            random_seed=self.config.get("random_seed", None),
            allow_duplicate_genes=False,
            on_start=self._on_start,
            on_generation=self._on_generation,
            on_stop=self._on_stop,
        )

        #  single-swap + spacing
        ga.mutation = self._mutation_single_swap_dist.__get__(self, self.__class__)
        ga.run()

        best_solution_arr, best_fitness, _ = ga.best_solution()
        best_solution = np.asarray(best_solution_arr, int)
        total_incidents = float(self.incident_freq.sum())
        best_pct = (best_fitness / total_incidents) if total_incidents > 0 else float("nan")

        if plot:
            try:
                ga.plot_fitness(title="GA (population best) over generations")
            except Exception:
                if verbose:
                    print("Plot failed.")

        return best_solution, float(best_fitness), float(best_pct), ga