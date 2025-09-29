import os
import time
from pathlib import Path
from typing import Dict, List, Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pygad
import numpy as np
from typing import Iterable

# Generate an init_pop with a "single-swap" around a given layout base_layout (allow none).
def make_single_swap_seeds_weighted(
    base_layout: np.ndarray | None,                 #  allow None
    incident_freq: np.ndarray,                      # (N,)
    feasible_indices: np.ndarray,                   # preferably incident>0 or Top-p set
    n_single_swap_seeds: int,
    pop_size: int,
    rng: np.random.Generator,
    *,
    k: int,                                         # chromosome length (num_stations)
    top_pct: float = 0.10,                          # Top-p% in feasible set
    alpha: float = 1.0,                             # demand weighting exponent
    uniform_mix_ratio: float = 0.0,                 # mix in uniform weight to increase exploration
    epsilon: float = 1e-12                          # avoid all-zeros
) -> np.ndarray:
    """
    Build a single-swap seeding population around base_layout.
    If base_layout is None, the very first seed (seed[0]) is sampled from feasible_indices
    with demand-weighted, without-replacement sampling of length k.

    Returns
    -------
    np.ndarray of shape (pop_size, k)
    """
    incident_freq = np.asarray(incident_freq).ravel()
    feasible_indices = np.asarray(feasible_indices, dtype=int)
    if feasible_indices.size == 0:
        raise ValueError("feasible_indices is empty.")
    if k <= 0:
        raise ValueError("k (num_stations) must be positive.")

    # demand weighted
    def weight_of(ids: np.ndarray) -> np.ndarray:
        """Demand^alpha mixed with uniform by uniform_mix_ratio, normalized to sum=1."""
        if ids.size == 0:
            return np.array([], dtype=float)
        w = np.maximum(incident_freq[ids].astype(float), epsilon) ** float(alpha)
        if uniform_mix_ratio > 0.0:
            lam = 1.0 - float(uniform_mix_ratio)
            w = lam * w + (1.0 - lam) * 1.0
        s = float(w.sum())
        return w / s if s > 0 else np.full(ids.shape[0], 1.0 / ids.shape[0], dtype=float)

    def weighted_choice_without_replacement(pool: np.ndarray, size: int) -> np.ndarray:
        """Compatibility for older NumPy: draw without replacement using dynamic reweighting."""
        pool = np.asarray(pool, dtype=int)
        out: list[int] = []
        cand = pool.copy()
        for _ in range(min(size, cand.size)):
            p_now = weight_of(cand)
            idx = int(rng.choice(np.arange(cand.size), p=p_now))
            out.append(int(cand[idx]))
            cand = np.delete(cand, idx)
        if len(out) < size:
            # fallback: allow repeats if pool smaller than size
            p_all = weight_of(pool)
            rest = rng.choice(pool, size=size - len(out), replace=True, p=p_all)
            out.extend([int(x) for x in rest])
        return np.asarray(out, dtype=int)

    # --- seed[0] ---
    if base_layout is None:                                         # [ADDED]
        # Seed 0: demand-weighted sampling from feasible_indices
        if feasible_indices.size >= k:
            p0 = weight_of(feasible_indices)
            try:
                base_layout = rng.choice(feasible_indices, size=k, replace=False, p=p0)
            except TypeError:
                base_layout = weighted_choice_without_replacement(feasible_indices, k)
        else:
            # Not enough feasible cells → allow repeats
            p0 = weight_of(feasible_indices)
            base_layout = rng.choice(feasible_indices, size=k, replace=True, p=p0)
    else:
        base_layout = np.asarray(base_layout, dtype=int)
        if base_layout.size != k:
            raise ValueError(f"base_layout length {base_layout.size} != k={k}")

    # --- compute Top-p% subset over feasible_indices ---
    freq_feasible = incident_freq[feasible_indices].astype(float)
    q = np.quantile(freq_feasible, 1.0 - top_pct) if top_pct > 0 else -np.inf
    top_mask = freq_feasible >= q
    top_candidates = feasible_indices[top_mask]

    # Fallback: if Top set too small, use full feasible set
    MIN_TOP = max(1, min(10, k))
    if top_candidates.size < MIN_TOP:
        top_candidates = feasible_indices

    seeds = [base_layout.copy()]

    # single-swap seeds: change exactly one position; new value sampled from Top set ---
    for _ in range(n_single_swap_seeds):
        child = base_layout.copy()
        i_pos = int(rng.integers(0, k))

        free_top = top_candidates[~np.isin(top_candidates, child)]
        if free_top.size == 0:
            free_top = feasible_indices[~np.isin(feasible_indices, child)]
            if free_top.size == 0:
                free_top = feasible_indices  # extreme fallback: allow duplicates

        p = weight_of(free_top)
        new_id = int(rng.choice(free_top, p=p))
        child[i_pos] = new_id
        seeds.append(child)

    # fill up to pop_size with demand-weighted random rows from feasible_indices ---
    while len(seeds) < pop_size:
        pool = feasible_indices
        if pool.size >= k:
            p = weight_of(pool)
            try:
                row = rng.choice(pool, size=k, replace=False, p=p)
            except TypeError:
                row = weighted_choice_without_replacement(pool, k)
        else:
            p = weight_of(pool)
            row = rng.choice(pool, size=k, replace=True, p=p)
        seeds.append(row)

    return np.asarray(seeds[:pop_size], dtype=int)


class GAOptimiser:
    """
    Fire station layout optimisation using a Genetic Algorithm (PyGAD).

    Fitness (to maximize) = expected number of efficiently served incidents:
        incidents_served = sum_i( efficiency_i * incident_freq_i )

    Where:
    - efficiency_i (0..1): from the RF model
    - incident_freq_i: historical incident frequency of cell i

    Data shapes:
    - xy_all: (N, 2)
    - time_matrix: (N, M)  travel time from each grid (N) to candidate stations (M)  [CHANGED doc]
    - A_matrix: (N, M)     0/1 reachability or count kernel                      [CHANGED doc]
    - partial_features: (N, P)  features excluding [nearest_time, station_count]
    - incident_freq: (N,)
    """

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
        # ---- bind data ----
        self.xy_all: np.ndarray = data["xy_all"]
        self.time_matrix: np.ndarray = data["time_matrix"]
        self.A_matrix: np.ndarray = data["A_matrix"]
        self.partial_features: np.ndarray = data["partial_features"]
        self.incident_freq: np.ndarray = np.asarray(data["incident_freq"], dtype=float).ravel()
        self.rf_model = data["rf_model"]
        self.total_incidents: float = float(data["total_incidents"])

        # (Optional) lighter dtypes can speed up a lot
        # self.time_matrix = self.time_matrix.astype(np.float32, copy=False)   # [OPTIONAL]
        # self.A_matrix = self.A_matrix.astype(np.int32, copy=False)           # [OPTIONAL]

        self.config = dict(config)
        self.gene_space = sorted({int(i) for i in gene_space})  # dedup + sort

        # ---- feature names ----
        self.feature_names = feature_names or self.feature_names_default

        # ---- sanity checks ----
        N = self.xy_all.shape[0]
        assert self.time_matrix.shape[0] == N, "time_matrix rows must match xy_all"
        assert self.partial_features.shape[0] == N, "partial_features rows must match xy_all"
        assert self.incident_freq.shape[0] == N, "incident_freq length must match xy_all"

        expected_P = len(self.feature_names) - 2  # exclude nearest_time & station_count
        assert self.partial_features.shape[1] == expected_P, (
            f"partial_features must have {expected_P} columns. Feature names: {self.feature_names}"
        )

        # ---- logging ----
        self.log_dir = Path(self.config.get("log_dir", "log"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._t0 = None
        self._log: list[tuple[float, float]] = []   # (elapsed_s, best_fitness)
        self._gen_times: list[float] = []

    # ---- helpers ----
    def _station_count_all(self, solution: np.ndarray) -> np.ndarray:
        """
        Given a layout (station column indices), return per-cell station_count (length N).
        If A_matrix is binary, sum along columns; otherwise threshold first.
        """
        solution = np.asarray(solution, dtype=int)
        A_sub = self.A_matrix[:, solution]                               # (N, k)
        counts = np.asarray(A_sub.sum(axis=1)).ravel().astype(int)       # (N,)
        return counts

    # ---- public: evaluate any layout ----
    def evaluate_layout(self, solution: np.ndarray):
        """
        Evaluate a given layout.

        Returns
        -------
        incidents_served : float
        eff_pct          : float
        detail           : pd.DataFrame with columns:
                           [nearest_time, station_count, incident_freq, efficiency, expected_served]
        """
        solution = np.asarray(solution, dtype=int)
        if np.unique(solution).size != solution.size:
            raise ValueError("solution has duplicate indices; must be unique")

        # nearest station travel time
        selected_times = self.time_matrix[:, solution]                    # [FIXED] removed duplicate line
        nearest_times = selected_times.min(axis=1)                        # (N,)

        # station count from A_matrix
        station_count = self._station_count_all(solution)                 # (N,)

        # build RF input in the correct order
        X = np.column_stack([nearest_times, self.partial_features, station_count])
        X_df = pd.DataFrame(X, columns=self.feature_names)

        # RF predict and aggregate
        eff = np.clip(self.rf_model.predict(X_df), 0.0, 1.0)             # (N,)
        expected_served = eff * self.incident_freq                        # (N,)

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

    def _on_generation(self, ga_instance: pygad.GA):
        """
        Called at the end of each generation:
          1) Anneal: every 50 gens, decrease gene_space_top_pct by 0.05 (min 0.05).
          2) Log progress & diversity every 10 gens.
          3) Append (elapsed, best) into memory log for saving at _on_stop.
        """
        # gen index
        try:
            gen = int(ga_instance.generations_completed)
        except Exception:
            gen = None

        # (1) anneal top ratio
        try:
            if gen is not None and gen > 0 and (gen % 50 == 0):
                cur = float(self.config.get("gene_space_top_pct", 0.20))
                new = max(0.05, cur - 0.05)
                if new != cur:
                    self.config["gene_space_top_pct"] = new
                    print(f" Gen {gen}: gene_space_top_pct {cur:.2f} → {new:.2f}")
        except Exception as e:
            print("on_generation/anneal error:", e)

        # (2) progress & diversity print
        try:
            if gen is not None and gen % 10 == 0:
                best_sol, best_fit, _ = ga_instance.best_solution()
                # population mean
                try:
                    mean_fit = float(np.mean(np.asarray(ga_instance.population_fitness, dtype=float))) \
                               if getattr(ga_instance, "population_fitness", None) is not None else float("nan")
                except Exception:
                    mean_fit = float("nan")

                # diversity: unique rows ratio
                uniq_ratio = None
                try:
                    pop = ga_instance.population  # (sol_per_pop, k)
                    if pop is not None:
                        uniq = np.unique(np.asarray(pop, dtype=int), axis=0).shape[0]
                        uniq_ratio = uniq / max(1, pop.shape[0])
                except Exception:
                    pass

                msg = f"Gen {gen:>5} | Best: {best_fit:.4f} | Mean: {mean_fit:.4f}"
                if uniq_ratio is not None:
                    msg += f" | Unique%: {uniq_ratio:.2%}"
                cur_top = float(self.config.get("gene_space_top_pct", 0.20))
                msg += f" | TopPct: {cur_top:.2f}"
                print(msg)
        except Exception as e:
            print("on_generation/log error:", e)

        # (3) append in-memory log (elapsed, best)
        try:
            best_fit = ga_instance.best_solution()[1]
            elapsed = time.time() - (self._t0 or time.time())
            self._log.append((elapsed, float(best_fit)))
        except Exception:
            pass

    def _on_stop(self, ga: pygad.GA, last_pop_fitness):
        """Flush logs to disk and draw the curve when GA stops."""
        # final best
        try:
            final_best = ga.best_solutions_fitness[-1]
        except Exception:
            _, final_best, _ = ga.best_solution()

        if not self._log:
            print(f"GA stopped. Final best fitness: {final_best:,.2f}")
            return

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
        """PyGAD fitness function: return incidents_served (maximize)."""
        incidents_served, _, _ = self.evaluate_layout(solution)
        return float(incidents_served)

    # =====================#
    # Run GA
    # =====================#
    def run_single(
        self,
        plot: bool = True,
        verbose: bool = True,
        start_layout: np.ndarray | None = None,  # only for logging/evaluation
        n_seeds: int = 0,
        seed_replace_rate: float = 0.2,
        save_dir: str | None = None,
        initial_population: np.ndarray | None = None,
    ):
        t0 = time.time()
        k = int(self.config["num_stations"])
        stop_criteria = tuple(self.config.get("stop_criteria", ()))

        pool_all = np.asarray(self.gene_space, dtype=int)
        if pool_all.size < k:
            raise ValueError(f"gene_space too small: {pool_all.size} < num_stations={k}")

        rng = np.random.default_rng(self.config.get("random_seed", None))
        incident_freq_arr = np.asarray(self.incident_freq).ravel()

        def _illegal(arr: np.ndarray, feasible: np.ndarray) -> np.ndarray:
            arr = np.asarray(arr, dtype=int)
            feasible = np.asarray(feasible, dtype=int)
            return np.setdiff1d(arr, feasible)

        # start_layout: warn only
        if start_layout is not None and verbose:
            bad = _illegal(start_layout, pool_all)
            if bad.size > 0:
                print(" start_layout has cells outside gene_space (warning only):", bad)

        # initial_population: legality + no-dup check
        if initial_population is not None:
            initial_population = np.asarray(initial_population, dtype=int)
            assert initial_population.ndim == 2 and initial_population.shape[1] == k, \
                f"initial_population must be (sol_per_pop, num_stations) with num_stations={k}"

            has_bad = False
            for i, sol in enumerate(initial_population):
                bad = _illegal(sol, pool_all)
                if bad.size > 0:
                    has_bad = True
                    if verbose:
                        print(f" initial seed {i} has cells outside gene_space:", bad)
            if has_bad:
                raise ValueError("initial_population contains genes outside gene_space.")

            if not self.config.get("allow_duplicate_genes", False):
                if any(np.unique(row).size != row.size for row in initial_population):
                    raise ValueError("initial_population has duplicates but allow_duplicate_genes=False.")
        else:
            initial_population = None  # let PyGAD sample from gene_space

        if verbose:
            print(f"Optimising with {k} stations from {pool_all.size} feasible locations (pre-filtered)...")

        # ---------- custom mutation: mutate into Top-P% only ----------
        def _mutation_top_p(self_inner, offspring, ga_instance=None):   # [ADDED]
            """Custom mutation: when a gene mutates, new value is sampled from Top-P% over gene_space."""
            top_pct = float(self.config.get("gene_space_top_pct", 0.10))
            vals = incident_freq_arr[pool_all].astype(float)

            if top_pct <= 0.0:
                top_ids = pool_all.copy()
            else:
                q = np.quantile(vals, 1.0 - top_pct)
                top_mask = vals >= q
                top_ids = pool_all[top_mask]
                min_top = max(1, min(10, k))
                if top_ids.size < min_top:
                    top_ids = pool_all.copy()

            mut_prob = float(self.config["mutation_probability"])
            avoid_dups = not bool(self.config.get("allow_duplicate_genes", False))

            # simple demand weights (no uniform mixing here—keep fast)
            def _p(ids: np.ndarray):
                w = incident_freq_arr[ids].astype(float)
                s = w.sum()
                return w / s if s > 0 else None

            for r in range(offspring.shape[0]):
                for c in range(offspring.shape[1]):
                    if rng.random() < mut_prob:
                        if avoid_dups:
                            pool = top_ids[~np.isin(top_ids, offspring[r])]
                            if pool.size == 0:
                                pool = top_ids
                        else:
                            pool = top_ids
                        p = _p(pool)
                        offspring[r, c] = int(rng.choice(pool, p=p))
            return offspring

        # ---------- build and run GA ----------
        ga = pygad.GA(
            num_generations=int(self.config["generations"]),
            sol_per_pop=int(self.config["sol_per_pop"]),
            num_parents_mating=int(self.config["num_parents_mating"]),
            num_genes=k,
            gene_type=int,
            gene_space=self.gene_space,                 # wide domain (incident>0 or user set)
            initial_population=initial_population,      # None → PyGAD samples; or pass your init_pop
            fitness_func=self._fitness,
            on_start=self._on_start,
            on_generation=self._on_generation,
            on_stop=self._on_stop,
            parent_selection_type=self.config["parent_selection_type"],
            K_tournament=int(self.config.get("K_tournament", 3)),
            crossover_type=self.config["crossover_type"],
            crossover_probability=float(self.config["crossover_probability"]),
            mutation_type="random",                     # will be replaced by our custom function below
            mutation_probability=float(self.config["mutation_probability"]),
            keep_elitism=int(self.config["keep_elitism"]),
            keep_parents=int(self.config["keep_parents"]),
            stop_criteria=stop_criteria,
            random_seed=self.config["random_seed"],
            allow_duplicate_genes=bool(self.config.get("allow_duplicate_genes", False)),
        )

        # Top-P% rule is enforced during mutation.
        ga.mutation = _mutation_top_p.__get__(self, self.__class__)
        ga.run()

        # ---------- results ----------
        best_solution_arr, best_fitness, _ = ga.best_solution()
        best_solution = np.asarray(best_solution_arr, dtype=int)
        best_incidents = float(best_fitness)
        best_pct = best_incidents / self.total_incidents if self.total_incidents > 0 else float("nan")

        if plot:
            try:
                ga.plot_fitness(title="GA (population best) over generations")
            except Exception as e:
                if verbose:
                    print("Plot failed:", e)

        if save_dir:
            self.save_results(save_dir, best_solution, ga=ga, extra_metrics={
                "generations": int(self.config["generations"]),
                "sol_per_pop": int(self.config["sol_per_pop"]),
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