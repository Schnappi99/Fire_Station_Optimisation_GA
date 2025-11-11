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

    # evaluate the efficiency
    def evaluate_layout(self, solution: np.ndarray):
        """Proxy to Evaluator"""
        return self.evaluator.evaluate_layout(solution)

    # build initial population
    def build_initial_population(
            self,
            base_layout: Optional[np.ndarray],     # Baseline layout used as the starting point (if provided).
            pop_size: int,                         # Number of individuals to generate in the initial population.
            mode: str,                             # Population generation mode: "balanced_init", "local_init", or "random_init".
            n_single_swap_from_base: int,          # Number of single-swap neighbours generated from the baseline layout.
            alpha: float,                          # Exponent controlling demand weight sensitivity (freq^alpha).
            uniform_mix_ratio: float,              # Mixing ratio between demand-weighted and uniform random sampling.
            top_pct: float,                        # Select candidate cells from the top p% highest-demand areas.
            rng=None,                              # Random number generator for reproducibility.
    ) -> np.ndarray:
        """
        Build the initial population for GA

        Modes:

            - "balanced_init": use the current layout as a start point (if given),
                               create several single-swap neighbours (n_single_swap_from_base),
                               and fill the rest with demand-weighted random layouts.

            - "local_init": use the current layout as a start point (if given),
                            create single-swap neighbours (n_single_swap_from_base).

            - "random_init": fully random layouts based on demand weights.
                             No baseline used.

        All layouts:
            - have exactly k = num_stations genes;
            - have no duplicate station indices;
            - can be biased towards high-demand cells using top_pct.
        """
        rng = self.rng
        k = int(self.config["num_stations"])
        feasible = np.asarray(self.gene_space, dtype=int)
        freq = np.asarray(self.incident_freq, dtype=float)

        if feasible.size < k:
            raise ValueError(f"gene_space too small: {feasible.size} < num_stations={k}")

        # weighted demand function
        def _weights(ids: np.ndarray) -> np.ndarray:
            """
            Demand-weighted sampling function:

            1): each location's weight = freq^alpha

            2): optional mixed with a uniform distribution (controlled by uniform_mix_ratio)
            to make the demand-weighted more random and diverse.

            """

            # weighted demand based on incident_freq
            w = np.maximum(freq[ids], 1e-12) ** float(alpha)

            # mixed with a uniform distribution
            if uniform_mix_ratio > 0.0:
                lam = 1.0 - float(uniform_mix_ratio)
                w = lam * w + (1.0 - lam)
            s = w.sum()
            return w / s if s > 0 else np.ones(ids.size) / ids.size

        def _top_pool(pool: np.ndarray) -> np.ndarray:
            """
            Select candidate cells from the top p% with the highest demand.

            If top_pct is not set or outside (0, 1), the full pool is returned.

            """
            if top_pct is None or top_pct <= 0.0 or top_pct >= 1.0:
                return pool
            vals = freq[pool]
            q = np.quantile(vals, 1.0 - float(top_pct))
            top = pool[vals >= q]
            return top if top.size > 0 else pool

        def sample_layout() -> np.ndarray:
            """
            Sample one layout of size k (new layout from scratch) :
            - no duplicates;
            - demand-weighted;
            - optionally restricted to top p% high-demand cells.
            """
            pool = _top_pool(feasible)
            if pool.size < k:
                raise ValueError(f"top-pool size {pool.size} < k={k}; relax top_pct or gene_space.")
            p = _weights(pool)
            chosen = rng.choice(pool, size=k, replace=False, p=p)
            return np.asarray(chosen, dtype=int)

        def single_swap_from(layout: np.ndarray) -> np.ndarray:
            """
            Change exactly one station index in `layout`(change exactly one station index).
            - No duplicates;
            - Optionally restricted to top p% high-demand cells.
            """
            child = np.asarray(layout, dtype=int).copy()
            if child.size != k:
                raise ValueError(f"layout length {child.size} != k={k}")

            # candidate pool: all feasible indices not already in the layout
            pool = feasible[~np.isin(feasible, child)]
            pool = _top_pool(pool)
            if pool.size == 0:
                # no alternative candidate, return unchanged
                return child
            # randomly pick a location index i
            i = int(rng.integers(0, k))
            p = _weights(pool)
            child[i] = int(rng.choice(pool, p=p))
            return child

        # build population
        # create an empty list named `pop` to store each layout (i.e., each individual instance).
        pop: list[np.ndarray] = []

        # add baseline into `pop` list if start_layout is not None
        if mode in ("balanced_init", "local_init") and base_layout is not None:
            base_layout = np.asarray(base_layout, dtype=int)
            if base_layout.size != k:
                raise ValueError(f"base_layout length {base_layout.size} != k={k}")
            if len(np.unique(base_layout)) != k:
                raise ValueError("base_layout contains duplicate indices.")
            pop.append(base_layout)

        # generate several single-swap neighbours (determined by n_single_swap_from_base)
        #  n_single_swap_from_base : the number of the layout which generated from single swap
        if mode in ("balanced_init", "local_init") and base_layout is not None and n_single_swap_from_base > 0:
            for _ in range(int(n_single_swap_from_base)):
                pop.append(single_swap_from(base_layout))

        # fill the rest of the layouts with random demand-weighted method
        if mode in ("balanced_init", "random_init"):
            while len(pop) < pop_size:
                pop.append(sample_layout())

        # safety fill in case
        while len(pop) < pop_size:
            pop.append(sample_layout())

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

        # Identify reason for stopping:
        # - If completed all planned generations → "completed_all_generations"
        # - Otherwise → stopped early due to stop criteria (e.g. saturation)
        max_gens = int(self.config.get("generations"))
        if gens >= max_gens:
            stop_reason = "maximum_generations_hit"
        else:
            stop_reason = "saturation"

        #  Save minimal summary (delegate to save_results)
        self.save_results(
            run_dir_root=self.config.get("out_dir", "outputs"),
            run_label="auto",
            ga=ga,
            stop_reason=stop_reason,
            total_gens=gens,
            total_time=total_time,
            avg_time_per_gen=avg_time,
            final_best_fitness=float(final_best),
        )


        # Save time–fitness log (existing behaviour)
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

        # Save per-generation performance
        if self._per_gen:
            with open(self.log_dir / "per_generation.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["gen", "dt_sec", "best_fitness", "offspring_count"])
                writer.writeheader()
                writer.writerows(self._per_gen)

        # Save overall timing summary
        summary = {
            "stop_reason": stop_reason,               # reason : "maximum_generations_hit" or "saturation"
            "total_generations": int(gens),           # real gens
            "max_generations": int(max_gens),         # max gens
            "total_time_sec": float(total_time),      # total time
            "avg_time_per_gen_sec": float(avg_time),  # avg time
            "final_best_fitness": (
                None if final_best != final_best else float(final_best)
            ),
        }

        with open(self.log_dir / "timing.json", "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_generation": self._per_gen},
                      f, ensure_ascii=False, indent=2)

        # Final console output
        print(f"GA stopped. Final best: {final_best:,.4f}")
        print(f"[GA] Total {gens} generations, total_time={total_time:.2f}s, avg/gen={avg_time:.3f}s")
        print(f"[GA] Wrote timing.json, per_generation.csv, and log.csv to: {self.log_dir}")


    def _mutation_single_swap(self, offspring, ga_instance=None):
        """
        Mutation: single-swap.
        Each offspring mutates EXACTLY ONE gene,
        sampling from Top-p% highest-demand cells (demand-weighted).
        """
        # --- RNG setup ---
        rng = getattr(self, "rng", None)
        if rng is None:
            rng = np.random.default_rng(self.config.get("random_seed", None))
            self.rng = rng

        k = offspring.shape[1]

        # --- Top-p% control ---
        top_pct = _parse_percent(self.config.get("gene_space_top_pct", "20%"), default=0.2)
        if top_pct is None or top_pct <= 0.0 or top_pct >= 1.0:
            top_pct = 0.0  # treat as no filtering

        mut_prob = float(self.config.get("mutation_probability", 1.0))
        xy, freq, feasible = self.xy_all, self.incident_freq, self.gene_space

        # --- helper: demand-weighted probability ---
        def p_of(ids: np.ndarray):
            w = freq[ids].astype(float)
            s = w.sum()
            return (w / s) if s > 0 else None

        # --- precompute Top-p% subset ---
        pool_all = feasible
        if top_pct > 0:
            vals = freq[pool_all]
            q = np.quantile(vals, 1.0 - top_pct)
            top_ids = pool_all[vals >= q]
            if top_ids.size < max(1, min(10, k)):
                top_ids = pool_all
        else:
            top_ids = pool_all

        # --- mutate each offspring ---
        for r in range(offspring.shape[0]):
            if rng.random() >= mut_prob:
                continue

            # pick which gene to mutate
            i = int(rng.integers(0, k))

            # candidate pool: all top_ids not already in this solution
            pool = top_ids[~np.isin(top_ids, offspring[r])]
            if pool.size == 0:
                pool = top_ids

            p = p_of(pool)
            old_val = int(offspring[r, i])

            # choose new location different from the old one
            for _ in range(100):
                cand = int(rng.choice(pool, p=p))
                if cand != old_val:
                    offspring[r, i] = cand
                    break

        return offspring

    def save_results(
            self,
            *,
            run_dir_root: str | Path = "outputs",
            run_label: str | None = None,
            ga=None,
            stop_reason: str,
            total_gens: int,
            total_time: float,
            avg_time_per_gen: float,
            final_best_fitness: float,
    ) -> Path:
        """
        Save minimal GA summary (stop reason, generations, time, fitness).
        """
        import json
        from datetime import datetime
        from pathlib import Path

        # Directory based only on timestamp
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(run_dir_root) / (f"{ts}_{run_label}" if run_label else ts)
        run_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "stop_reason": stop_reason,  # "maximum_generations_hit" / "saturation"
            "total_generations": int(total_gens),
            "total_time_sec": float(total_time),
            "avg_time_per_gen_sec": float(avg_time_per_gen),
            "final_best_fitness": float(final_best_fitness),
        }

        out_json = run_dir / "summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[SAVE] Summary saved to: {out_json.resolve()}")
        return run_dir

    # main run
    def run_single(
            self,
            *,
            start_layout,
            initial_population,
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
            K_tournament=int(self.config.get("K_tournament")),
            crossover_type=self.config["crossover_type"],
            crossover_probability=float(self.config.get("crossover_probability")),  # no crossover
            mutation_type="random",
            mutation_probability=float(self.config.get("mutation_probability")),
            keep_elitism=int(self.config.get("keep_elitism")),
            keep_parents=int(self.config.get("keep_parents")),
            stop_criteria=stop_criteria,
            random_seed=self.config.get("random_seed"),
            allow_duplicate_genes=False,
            on_start=self._on_start,
            on_generation=self._on_generation,
            on_stop=self._on_stop,
        )

        #  mutation
        ga.mutation = self._mutation_single_swap.__get__(self, self.__class__)
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