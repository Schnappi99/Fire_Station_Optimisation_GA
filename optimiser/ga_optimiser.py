# optimiser/ga_runner.py
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import geopandas as gpd
import pandas as pd
import pygad
from joblib import load
import numpy as np

from optimiser.config import config  # dict
from optimiser.data_loader import load_data

import time
import matplotlib.pyplot as plt
import utils.osrm_utils as tools
from shapely.geometry import Point



def make_single_swap_seeds(base_layout: np.ndarray,
                           all_candidates: np.ndarray,
                           n_single_swap_seeds: int,
                           pop_size: int,
                           rng: np.random.Generator) -> np.ndarray:
    """
    Construct an initial population for GA that emphasizes local search
    around the current (base) layout.
    """
    base_layout = np.asarray(base_layout, dtype=int)
    k = base_layout.size
    pool_all = np.asarray(all_candidates, dtype=int)

    seeds = [base_layout.copy()]

    locked = set(base_layout.tolist())
    free_choices = [c for c in pool_all if c not in locked]
    if not free_choices:
        free_choices = pool_all.tolist()

    for _ in range(n_single_swap_seeds):
        child = base_layout.copy()
        i_pos = rng.integers(0, k)
        new_id = int(rng.choice([c for c in free_choices if c != child[i_pos]]))
        child[i_pos] = new_id
        seeds.append(child)

    while len(seeds) < pop_size:
        seeds.append(rng.choice(pool_all, size=k, replace=False))

    return np.array(seeds[:pop_size], dtype=int)


@dataclass

class GAOptimiser:
    """
    Fire station layout optimisation using Genetic Algorithm (PyGAD).

    Fitness (to maximize) = expected number of efficiently served incidents

        incidents_served = sum_i( efficiency_i * incident_freq_i )

    where:
    - efficiency_i: predicted efficiency (0~1) from RF model
    - incident_freq_i: historical frequency of incidents in cell i
    其中：
    - efficiency_i：Efficiency of random forest model prediction (0~1)
    - incident_freq_i：The frequency of historical fire alarm events in grid i
    """

    # Default feature names (must match RF training)
    feature_names_default = [
        "nearest_station_travel_time",
        "neighbour_frequency_per_month",
        "Agriculture - mainly crops",
        "Deciduous woodland",
        "station_count",
    ]

    def __init__(
        self,
        data: Dict,
        config: Dict,
        gene_space: Iterable[int],
        feature_names: List[str] | None = None,
    ):
        """
        Parameters
        ----------
        data: dict
              - "xy_all": (N,2) grid centroids in EPSG:27700
              - "time_matrix": (N,N) travel times from each grid to candidate stations
              - "A_matrix": (N,N) distance matrix in meters OR binary reachability matrix   binary(0,1)
              - "partial_features": (N,3) other features besides travel time & station_count
              - "incident_freq": (N,1) incident frequencies (weights)
              - "rf_model": trained sklearn RF model with .predict
              - "total_incidents": float, sum of incident_freq
        config: GAConfig
            GA parameters
        gene_space: Iterable[int]
             candidate station indices
        feature_names: list[str] | None
            Full feature names in correct order (must match RF training)
        """
        # ---- data binding ----
        self.xy_all: np.ndarray = data["xy_all"]
        self.time_matrix: np.ndarray = data["time_matrix"]
        self.A_matrix: np.ndarray = data["A_matrix"]
        self.partial_features: np.ndarray = data["partial_features"]
        self.incident_freq: np.ndarray = np.asarray(data["incident_freq"], dtype=float)
        self.rf_model = data["rf_model"]
        self.total_incidents: float = float(data["total_incidents"])

        self.config = dict(config)
        self.gene_space = list(sorted(set(int(i) for i in gene_space)))  # 去重排序 / dedup & sort

        # ---- feature names ----
        self.feature_names = feature_names or self.feature_names_default

        # ---- sanity check ----
        N = self.xy_all.shape[0]
        assert self.time_matrix.shape[0] == N, "time_matrix rows must match xy_all"
        assert self.partial_features.shape[0] == N, "partial_features rows must match xy_all"
        assert self.incident_freq.shape[0] == N, "incident_freq length must match xy_all"

        expected_P = len(self.feature_names) - 2
        assert self.partial_features.shape[1] == expected_P, (
            f"partial_features must have {expected_P} columns. Feature names: {self.feature_names}"
        )

        # ---- logging ----
        self.log_dir = Path(self.config.get("log_dir", "log"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._t0 = None
        self._log = []  # (elapsed_s, best_fitness)
        self._gen_times = []  # 记录每代的耗时


    def _station_count_all(self, solution: np.ndarray) -> np.ndarray:
        """
        给定布局（station列索引），返回所有格网的 station_count（长度 N 的 1D 数组）。
        A_matrix 为二值(0/1)时：直接按列求和；若未来用距离/时间矩阵，可改为阈值判断后再 sum。
        """
        solution = np.asarray(solution, dtype=int)
        A_sub = self.A_matrix[:, solution]                            # (N, k)
        # 稀疏/稠密都可：sum(axis=1) 可能产生 (N,1)，强制扁平 1D:
        counts = np.asarray(A_sub.sum(axis=1)).ravel().astype(int)    # (N,)
        return counts

    # =====================#
    #    Public API for evaluating the layout
    # =====================#
    def evaluate_layout(self, solution: np.ndarray):
        """
        Independently evaluate any given layout (for feasibility study).

        Parameters
        ----------
        solution : array-like[int] station indices (no duplicates)

        Returns
        -------
        incidents_served : float     efficiently served incidents (absolute count)
        eff_pct : float              Percentage of total incidents served efficiently
        detail : pd.DataFrame        Per-cell details: nearest_time, station_count_10km, incident_freq, efficiency, expected_served
        """
        solution = np.asarray(solution, dtype=int)
        if len(np.unique(solution)) != len(solution):
            raise ValueError("solution has duplicate indices; must be unique")

        # nearest station travel time
        selected_times = self.time_matrix[:, solution]
        selected_times = self.time_matrix[:, solution]
        nearest_times = selected_times.min(axis=1)
        # station_count from A_matrix
        station_count = self._station_count_all(solution)

        # build RF input in correct column order
        X = np.column_stack([nearest_times, self.partial_features, station_count])
        X_df = pd.DataFrame(X, columns=self.feature_names)

        # RF predict and aggregate
        eff = np.clip(self.rf_model.predict(X_df), 0.0, 1.0)
        expected_served = eff * self.incident_freq

        incidents_served = float(expected_served.sum())
        eff_pct = incidents_served / self.total_incidents if self.total_incidents > 0 else 0.0

        detail = pd.DataFrame({
            "nearest_time": nearest_times,
            "station_count": station_count,
            "incident_freq": self.incident_freq,
            "efficiency": eff,
            "expected_served": expected_served,
        })
        return incidents_served, eff_pct, detail


    # =====================#
    # GA callbacks
    # =====================#
    def _on_start(self, ga: pygad.GA):
        self._t0 = time.time()
        self._log.clear()
        self._gen_times.clear()
        print(f"Optimising with {ga.num_genes} stations from {len(self.gene_space)} feasible locations...")

    def _on_generation(self, ga: pygad.GA):
        # 已用时
        now = time.time()
        elapsed = now - (self._t0 or now)

        # 记录 elapsed 序列（用于算每代耗时的移动平均）
        if not hasattr(self, "_gen_elapsed"):
            self._gen_elapsed = []
        self._gen_elapsed.append(elapsed)

        # 估算“每代耗时”（移动平均）
        if len(self._gen_elapsed) >= 2:
            per_gen_inst = self._gen_elapsed[-1] - self._gen_elapsed[-2]
        else:
            per_gen_inst = elapsed / max(1, ga.generations_completed)

        ma_window = 20
        if len(self._gen_elapsed) >= ma_window:
            per_gen_ma = (self._gen_elapsed[-1] - self._gen_elapsed[-ma_window]) / (ma_window - 1)
        else:
            per_gen_ma = per_gen_inst

        # 预计剩余时间 ETA
        remaining_gens = max(0, ga.num_generations - ga.generations_completed)
        eta_sec = remaining_gens * max(per_gen_ma, 1e-6)

        # 当前这代的“历史最佳适应度”
        # 注意：PyGAD 没有 ga.best_solution_fitness；应使用 ga.best_solutions_fitness[-1] 或 ga.best_solution()[1]
        try:
            best_fit = ga.best_solutions_fitness[-1]
        except Exception:
            # 保险起见（极早期代），用 best_solution()
            _, best_fit, _ = ga.best_solution()

        # 每 N 代打印一次
        if ga.generations_completed % 10 == 0:
            print(
                f"Gen {ga.generations_completed:>4} | "
                f"Elapsed: {elapsed:6.1f}s | "
                f"ETA: {eta_sec:6.1f}s | "
                f"Best fitness: {best_fit:,.2f}"
            )
            # 记录到内存日志（后续在 _on_stop 里落盘）
            self._log.append((elapsed, float(best_fit)))

    def _on_stop(self, ga: pygad.GA, last_pop_fitness):
        """在停止时落盘日志并绘制曲线。"""
        # 取得最终最佳适应度（用于打印）
        try:
            final_best = ga.best_solutions_fitness[-1]
        except Exception:
            _, final_best, _ = ga.best_solution()

        # 若无日志记录，直接打印并返回
        if not self._log:
            print(f"GA stopped. Final best fitness: {final_best:,.2f}")
            return

        # 保存 CSV & 曲线
        log_df = pd.DataFrame(self._log, columns=["Time_s", "Best_fitness_incidents"])
        log_csv = self.log_dir / "log.csv"
        log_png = self.log_dir / "fitness_curve.png"
        log_df.to_csv(log_csv, index=False)

        plt.figure(figsize=(9, 5.5))
        plt.plot(log_df["Time_s"].values, log_df["Best_fitness_incidents"].values, marker="o")
        plt.xlabel("Elapsed Time (s)")
        plt.ylabel("Best Fitness (incidents served efficiently)")
        plt.title("GA Fitness Over Time")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(log_png, dpi=150)
        plt.close()

        print(f"Saved logs to: {log_csv} and {log_png} | Final best: {final_best:,.2f}")

    def _fitness(self, ga_instance: pygad.GA, solution: np.ndarray, solution_idx: int) -> float:
        """
        GA 的 fitness_func（返回 incidents_served）
        GA fitness function (returns incidents_served as float)
        """
        incidents_served, _, _ = self.evaluate_layout(solution)
        return float(incidents_served)

    # =====================#
    # 运行 GA / Run GA
    # =====================#
    def run_single(
            self,
            plot: bool = True,
            verbose: bool = True,
            start_layout: np.ndarray | None = None,  # 旧：单个起点布局（可选）
            n_seeds: int = 0,  # 旧：基于 start_layout 生成的邻域种子数
            seed_replace_rate: float = 0.2,  # 旧：邻域种子替换比例
            save_dir: str | None = None,
            initial_population: np.ndarray | None = None,  # 新增：直接传入自定义初始种群
    ):
        t0 = time.time()

        if verbose:
            print(
                f"Optimising with {self.config['num_stations']} stations from {len(self.gene_space)} feasible locations...")

        stop_criteria = tuple(self.config.get("stop_criteria", ()))
        k = self.config["num_stations"]
        rng = np.random.default_rng(self.config.get("random_seed", None))

        # ---------- 1) 如果显式给了 initial_population，就直接用它 ----------
        # 形状检查： (sol_per_pop, k)
        if initial_population is not None:
            initial_population = np.asarray(initial_population, dtype=int)
            assert initial_population.ndim == 2 and initial_population.shape[1] == k, \
                f"initial_population must be (sol_per_pop, num_stations) with num_stations={k}"

        # ---------- 2) 否则，基于 start_layout 自动生成 seeds ----------
        elif start_layout is not None:
            start_layout = np.asarray(start_layout, dtype=int)
            assert start_layout.size == k, "start_layout length must equal num_stations"
            assert np.unique(start_layout).size == k, "start_layout must have unique indices"
            assert set(start_layout).issubset(set(self.gene_space)), "start_layout must be subset of gene_space"

            seeds = [start_layout.copy()]
            pool_all = np.array(self.gene_space, dtype=int)

            # 邻域扰动种子
            for _ in range(max(0, n_seeds)):
                child = start_layout.copy()
                m = max(1, int(round(seed_replace_rate * k)))
                pos = rng.choice(k, size=m, replace=False)
                used = set(child.tolist())
                for p in pos:
                    choices = np.array([c for c in pool_all if c not in used], dtype=int)
                    if choices.size == 0:
                        # 极端兜底
                        choices = pool_all
                    new_val = int(rng.choice(choices))
                    used.discard(child[p])
                    child[p] = new_val
                    used.add(new_val)
                seeds.append(child)

            # 补到种群大小
            pop_size = self.config["sol_per_pop"]
            while len(seeds) < pop_size:
                seeds.append(rng.choice(pool_all, size=k, replace=False))
            initial_population = np.array(seeds[:pop_size], dtype=int)

        # ---------- 3) 构造 GA（把 initial_population 传进去） ----------
        ga = pygad.GA(
            num_generations=self.config["generations"],
            sol_per_pop=self.config["sol_per_pop"],
            num_parents_mating=self.config["num_parents_mating"],
            num_genes=self.config["num_stations"],
            gene_type=int,
            gene_space=self.gene_space,
            initial_population=initial_population,  # ★ 关键行
            fitness_func=self._fitness,
            on_start=self._on_start,
            on_generation=self._on_generation,
            on_stop=self._on_stop,
            parent_selection_type=self.config["parent_selection_type"],
            crossover_type=self.config["crossover_type"],
            crossover_probability=self.config["crossover_probability"],
            mutation_type=self.config["mutation_type"],
            mutation_probability=self.config["mutation_probability"],
            keep_elitism=self.config["keep_elitism"],
            keep_parents=self.config["keep_parents"],
            stop_criteria=stop_criteria,
            random_seed=self.config["random_seed"],
            allow_duplicate_genes=False,
        )

        ga.run()

        # —— 后面保持你原来的 best_solution / 打印 / 返回 ——
        best_solution_arr, best_fitness, _ = ga.best_solution()
        best_solution = np.asarray(best_solution_arr, dtype=int)
        best_incidents = float(best_fitness)
        best_pct = best_incidents / self.total_incidents if self.total_incidents > 0 else float("nan")

        if plot:
            try:
                ga.plot_fitness(title="GA (population best) over generations")
            except Exception as e:
                if verbose: print("Plot failed:", e)

        if save_dir:
            self.save_results(save_dir, best_solution, ga=ga, extra_metrics={
                "generations": self.config["generations"],
                "sol_per_pop": self.config["sol_per_pop"],
            })

        if verbose:
            total_runtime = time.time() - t0
            print("-" * 56)
            print("GA finished!")
            print(f"Best fitness (incidents): {best_incidents:,.4f} / {self.total_incidents:,.4f}")
            print(f"Efficiency: {best_pct:.2%}")
            print(f"Best layout candidate indices: {best_solution.tolist()}")
            print(f"Total runtime: {total_runtime:,.1f}s")

        return best_solution, best_incidents, best_pct, ga

    def run(
            self,
            plot: bool = True,
            verbose: bool = True,
            start_layout: np.ndarray | None = None,  # 起始布局（可选）
            n_seeds: int = 0,  # 生成多少个邻域扰动种子（0=不用）
            seed_replace_rate: float = 0.2,  # 种子替换比例
            save_dir: str | None = None,  # 可选：自动落盘目录
    ):
        t0 = time.time()

        if verbose:
            print(
                f"Optimising with {self.config['num_stations']} stations from {len(self.gene_space)} feasible locations...")

        stop_criteria = tuple(self.config.get("stop_criteria", ()))
        k = self.config["num_stations"]
        rng = np.random.default_rng(self.config.get("random_seed", None))

        # ---------- initial population（以当前布局为起点，非必需） ----------
        initial_population = None
        if start_layout is not None:
            start_layout = np.asarray(start_layout, dtype=int)
            assert start_layout.size == k, "start_layout length must equal num_stations"
            assert np.unique(start_layout).size == k, "start_layout must have unique indices"
            assert set(start_layout).issubset(set(self.gene_space)), "start_layout must be subset of gene_space"

            seeds = [start_layout.copy()]
            # 邻域扰动
            if n_seeds > 1:
                pool_all = np.array(self.gene_space, dtype=int)
                for _ in range(n_seeds - 1):
                    child = start_layout.copy()
                    m = max(1, int(round(seed_replace_rate * k)))
                    pos = rng.choice(k, size=m, replace=False)
                    # 从未使用的候选里抽
                    used = set(child.tolist())
                    choices = np.array([c for c in pool_all if c not in used], dtype=int)
                    if choices.size >= m:
                        new_vals = rng.choice(choices, size=m, replace=False)
                    else:
                        # 兜底：允许从 gene_space 里挑不冲突的
                        new_vals = []
                        used = set(child.tolist())
                        for _ in range(m):
                            cands = np.array([c for c in pool_all if c not in used], dtype=int)
                            new = rng.choice(cands)
                            used.add(new)
                            new_vals.append(new)
                        new_vals = np.array(new_vals, dtype=int)
                    child[pos] = new_vals
                    seeds.append(child)

            # 填满种群
            while len(seeds) < self.config["sol_per_pop"]:
                seeds.append(rng.choice(self.gene_space, size=k, replace=False))
            initial_population = np.array(seeds[: self.config["sol_per_pop"]], dtype=int)

        # ---------- GA ----------
        ga = pygad.GA(
            num_generations=self.config["generations"],
            sol_per_pop=self.config["sol_per_pop"],
            num_parents_mating=self.config["num_parents_mating"],
            num_genes=self.config["num_stations"],
            gene_type=int,
            gene_space=self.gene_space,  # 允许全局搜索
            initial_population=initial_population,  # 用起始布局做种子（可选）
            fitness_func=self._fitness,
            on_start=self._on_start,
            on_generation=self._on_generation,
            on_stop=self._on_stop,
            parent_selection_type=self.config["parent_selection_type"],
            crossover_type=self.config["crossover_type"],
            crossover_probability=self.config["crossover_probability"],
            mutation_type=self.config["mutation_type"],
            mutation_probability=self.config["mutation_probability"],
            keep_elitism=self.config["keep_elitism"],
            keep_parents=self.config["keep_parents"],
            stop_criteria=stop_criteria,
            random_seed=self.config["random_seed"],
            allow_duplicate_genes=False,
        )

        ga.run()

        # ---------- 统一产出 ----------
        try:
            best_solution_arr, best_fitness, _ = ga.best_solution()
        except Exception:
            # 极少数情况下 best_solution() 在早期代可能异常；兜底从历史序列取最后一个
            best_fitness = float(ga.best_solutions_fitness[-1])
            # 这里没有直接拿到解本身，如需更严谨可在 _fitness 里缓存，但一般不会走到这里
            best_solution_arr = ga.population[np.argmax(ga.last_generation_fitness)]

        best_solution = np.asarray(best_solution_arr, dtype=int)
        # 注意：你的 fitness 定义如果是“比例”而不是“绝对事件数”，这里需要相应解释
        best_incidents = float(best_fitness)
        best_pct = best_incidents / self.total_incidents if self.total_incidents > 0 else float("nan")

        # 保存曲线（可视化）
        if plot:
            try:
                ga.plot_fitness(title="GA (population best) over generations")
            except Exception as e:
                if verbose:
                    print("Plot failed:", e)

        # 自动保存（可选）
        if save_dir:
            self.save_results(save_dir, best_solution, ga=ga, extra_metrics={
                "generations": self.config["generations"],
                "sol_per_pop": self.config["sol_per_pop"],
                "num_parents_mating": self.config["num_parents_mating"],
            })

        if verbose:
            total_runtime = time.time() - t0
            print("-" * 56)
            print("GA finished!")
            print(f"Best fitness (incidents): {best_incidents:,.2f} / {self.total_incidents:,.0f}")
            print(f"Efficiency (% incidents served well): {best_pct:.2%}")
            print(f"Best layout candidate indices: {best_solution.tolist()}")
            print(f"Total runtime: {total_runtime:,.1f}s")

        # <<< 关键：无论如何都 return 四元组
        return best_solution, best_incidents, best_pct, ga

