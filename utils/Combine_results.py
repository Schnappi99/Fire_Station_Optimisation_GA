import json
import pandas as pd
from pathlib import Path

# 设置 results 文件夹路径
results_dir = Path("optimiser/outputs")

# 匹配所有以 summary_ 开头、.json 结尾的文件
json_files = list(results_dir.rglob("summary_*.json"))

records = []
for f in json_files:
    with open(f, "r", encoding="utf-8") as file:
        data = json.load(file)

    summary = data.get("summary", {})
    params = data.get("parameters", {})

    record = {
        "file_name": f.name,                # JSON 文件名
        "run_folder": f.parent.name,        # 文件夹名（如 top0.1_dist3000_mut0.20_31-063059）
        "method_mode": params.get("method_mode"),
        "gene_space_top_pct": params.get("gene_space_top_pct"),
        "stop_criteria": ", ".join(params.get("stop_criteria", [])),
        "generations_completed": summary.get("generations_completed"),
        "final_best_fitness": summary.get("final_best_fitness"),
        "total_time_sec": summary.get("total_time_sec"),
        "avg_time_per_gen_sec": summary.get("avg_time_per_gen_sec"),
        "stop_reason": summary.get("stop_reason"),
    }

    records.append(record)

# 转为 DataFrame
df = pd.DataFrame(records)

# 可选排序
df = df.sort_values(by=["gene_space_top_pct", "final_best_fitness"], ascending=[False, False])

# 保存 CSV 文件
out_path = results_dir / "results_summary.csv"
df.to_csv(out_path, index=False)

print(f"✅ Combined results saved to: {out_path.resolve()}")
print(df.head())