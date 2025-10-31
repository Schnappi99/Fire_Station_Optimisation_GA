import json
import re
from pathlib import Path
import pandas as pd

RESULTS_DIR = Path("/Users/zhaoyuxin/Repos/fire_station_optimisation_ga/optimiser/outputs")  # 改成你的输出根目录

# 允许两种命名
json_files = list(RESULTS_DIR.rglob("summary.json")) + list(RESULTS_DIR.rglob("summary_*.json"))

def _pct_to_float(x):
    """将 0.2 / '0.2' / 20 / '20%' 统一转为 0.2。无法解析返回 None。"""
    if x is None:
        return None
    try:
        if isinstance(x, (int, float)):
            v = float(x)
            return v/100.0 if v > 1 else v
        s = str(x).strip()
        if s.endswith("%"):
            return float(s[:-1]) / 100.0
        v = float(s)
        return v/100.0 if v > 1 else v
    except Exception:
        return None

def _infer_top_pct_from_name(name: str):
    """
    从 run_folder 或文件名中解析: top0.2 / top0.20 / top20 / top20%
    返回 [0,1] 的 float 或 None
    """
    # top 后面跟数字与可选的 % 号
    m = re.search(r"top(\d+(\.\d+)?)(%)?", name.lower())
    if not m:
        return None
    num = float(m.group(1))
    has_pct = bool(m.group(3))
    if has_pct or num > 1:
        return num / 100.0
    return num

records = []
for f in json_files:
    try:
        data = json.loads(Path(f).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to read {f}: {e}")
        continue

    summary = data.get("summary", {}) or {}
    params  = data.get("parameters", {}) or {}

    # 1) JSON 正常字段
    top_pct = _pct_to_float(params.get("gene_space_top_pct"))
    # 2) 兼容旧字段
    if top_pct is None:
        top_pct = _pct_to_float(params.get("init_top_pct"))
    # 3) 从文件/目录名推断
    if top_pct is None:
        top_pct = _infer_top_pct_from_name(f.name) or _infer_top_pct_from_name(f.parent.name)

    # 记录
    record = {
        "file_name": f.name,
        "run_folder": f.parent.name,
        "method_mode": params.get("method_mode") or params.get("init_mode"),
        "gene_space_top_pct": top_pct,  # 标准化为 0~1
        "stop_criteria": ", ".join(params.get("stop_criteria", [])),
        "generations_completed": summary.get("generations_completed"),
        "final_best_fitness": summary.get("final_best_fitness"),
        "total_time_sec": summary.get("total_time_sec"),
        "avg_time_per_gen_sec": summary.get("avg_time_per_gen_sec"),
        "stop_reason": summary.get("stop_reason"),
    }
    records.append(record)

# 生成 DataFrame
df = pd.DataFrame(records)

# 若列缺失，补上方便排序
for col in ["gene_space_top_pct", "final_best_fitness"]:
    if col not in df.columns:
        df[col] = None


df = df.sort_values(
    by=["gene_space_top_pct", "final_best_fitness"],
    ascending=[False, False],
    na_position="last"
)

out_csv = RESULTS_DIR / "results_summary.csv"
df.to_csv(out_csv, index=False)
print(f" Combined results saved to: {out_csv.resolve()}")
print(df.head(10))