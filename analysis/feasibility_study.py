import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore
from optimiser import ga_optimiser
from optimiser.data_loader import load_data




def evaluate_random_layouts(n_samples: int, n_station: int, feasible_cells: np.ndarray) -> pd.DataFrame:
    """
    随机采样若干个 layout，计算其 fitness，用于和优化结果对比
    """
    results = []

    print(f"Evaluating {n_samples} random layouts with {n_station} stations...")

    for _ in tqdm(range(n_samples)):
        random_layout = np.random.choice(feasible_cells, size=n_station, replace=False)
        fitness = ga_optimiser.fitness_function(None, random_layout, 0)  # 忽略 GA 参数
        results.append((random_layout, fitness))

    # 保存为 DataFrame
    df = pd.DataFrame(results, columns=["layout", "fitness"])
    return df


if __name__ == "__main__":
    # === 加载数据，并写入 optimiser 的全局变量中 ===
    data = load_data()
    ga_optimiser._xy_all = data["xy_all"]
    ga_optimiser._incident_xy = data["incident_xy"]
    ga_optimiser._incident_freq = data["incident_freq"]
    ga_optimiser._incident_grid_idx = data["incident_grid_idx"]
    ga_optimiser._features = data["features"]
    ga_optimiser._rf_model = data["rf_model"]
    ga_optimiser._total_incidents = data["total_incidents"]

    # 参数设置
    n_samples = 1000
    n_station = 40
    feasible_cells = np.arange(ga_optimiser._xy_all.shape[0])  # 所有 cell 都可行，也可读取 mask

    # 运行随机 layout 评估
    df_random = evaluate_random_layouts(n_samples, n_station, feasible_cells)

    # 保存
    out_path = "/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_layouts.csv"
    df_random.to_csv(out_path, index=False)
    print(f"Saved random layout results to: {out_path}")

    # ===== 可视化 Fitness 分布 =====
    plt.figure(figsize=(10, 6))
    plt.hist(df_random["fitness"], bins=30, color="skyblue", edgecolor="black", alpha=0.8)
    plt.title("Fitness Distribution of 1000 Random Layouts")
    plt.xlabel("Fitness")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()

    # 保存 & 展示图
    plt.savefig("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_layout_hist.png")
    plt.show()

    # ===== 当前布局效率的百分位计算（例如 28493） =====
    current_fitness =  49.981  # <-- 你可以替换成你的真实 layout 得分
    percentile = percentileofscore(df_random["fitness"], current_fitness)
    print(f"Current layout fitness: {current_fitness}")
    print(f"Percentile in random layouts: {percentile:.2f}%")

    plt.savefig("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/analysis/random_fitness_histogram.png", dpi=300)