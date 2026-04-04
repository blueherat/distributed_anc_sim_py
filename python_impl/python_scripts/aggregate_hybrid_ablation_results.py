from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Hybrid ablation run results (multiseed).")
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def safe_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("1", "true", "yes", "y"):
            return True
        if s in ("0", "false", "no", "n"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_std(arr: list[float]) -> tuple[float, float]:
    if len(arr) == 0:
        return float("nan"), float("nan")
    vals = np.asarray(arr, dtype=np.float64)
    return float(np.mean(vals)), float(np.std(vals, ddof=1) if vals.size > 1 else 0.0)


def ci95(std: float, n: int) -> float:
    if n <= 0 or not np.isfinite(std):
        return float("nan")
    return float(1.96 * std / np.sqrt(float(n)))


def flatten_fail_reasons(rows: list[dict[str, Any]]) -> Counter:
    c: Counter = Counter()
    for r in rows:
        for reason in r.get("fail_reasons", []):
            c[str(reason)] += 1
    return c


def collect_seed_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cfg_dir in sorted([p for p in results_root.iterdir() if p.is_dir()]):
        for seed_dir in sorted([p for p in cfg_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")]):
            train_summary = seed_dir / "train" / "summary.json"
            eval_summary = seed_dir / "eval" / "summary.json"
            row: dict[str, Any] = {
                "config": str(cfg_dir.name),
                "seed": int(seed_dir.name.replace("seed_", "")) if seed_dir.name.replace("seed_", "").isdigit() else seed_dir.name,
                "train_summary_exists": train_summary.exists(),
                "eval_summary_exists": eval_summary.exists(),
            }

            if train_summary.exists():
                tr = read_json(train_summary)
                row.update(
                    {
                        "ablation_tag": tr.get("ablation_tag", ""),
                        "feature_encoding": tr.get("feature_encoding", ""),
                        "fusion_mode": tr.get("fusion_mode", ""),
                        "disable_feature_b": tr.get("disable_feature_b", False),
                        "loss_domain": tr.get("loss_domain", ""),
                        "lambda_reg": safe_float(tr.get("lambda_reg", float("nan"))),
                        "final_epoch": safe_float(tr.get("final_epoch", float("nan"))),
                    }
                )
                stages = tr.get("stage_summaries", [])
                if isinstance(stages, list):
                    for s in stages:
                        level = int(s.get("level", -1))
                        row[f"best_val_nr_l{level}"] = safe_float(s.get("best_val_nr_db", float("nan")))

            if eval_summary.exists():
                ev = read_json(eval_summary)
                imp_gate = ev.get("improvement_gate") or {}
                half_gate = ev.get("half_target_gate") or {}
                row.update(
                    {
                        "gate_status": str(ev.get("gate_status", "unknown")),
                        "mean_nr_db": safe_float(ev.get("mean_nr_db", float("nan"))),
                        "warmstart_ratio": safe_float(
                            (ev.get("warmstart_metrics") or {}).get("convergence_step_ratio_mean", float("nan"))
                        ),
                        "warmstart_early_gain_db": safe_float(
                            (ev.get("warmstart_metrics") or {}).get("early_gain_db_mean", float("nan"))
                        ),
                        "warmstart_init_nr_db_mean": safe_float(
                            (ev.get("warmstart_metrics") or {}).get("init_nr_db_mean", float("nan"))
                        ),
                        "improvement_pass": safe_bool(imp_gate.get("pass", False)),
                        "improvement_enabled": safe_bool(imp_gate.get("enabled", False)),
                        "improvement_gap_db": safe_float(
                            imp_gate.get("gap_db", (ev.get("warmstart_metrics") or {}).get("improvement_gap_db", float("nan")))
                        ),
                        "improvement_threshold_db": safe_float(
                            imp_gate.get("threshold_db", (ev.get("warmstart_metrics") or {}).get("min_improvement_db", float("nan")))
                        ),
                        "improvement_early_gain_db_mean": safe_float(
                            imp_gate.get("early_gain_db_mean", (ev.get("warmstart_metrics") or {}).get("early_gain_db_mean", float("nan")))
                        ),
                        "improvement_sample_pass_rate": safe_float(
                            imp_gate.get("sample_pass_rate", (ev.get("warmstart_metrics") or {}).get("sample_improvement_pass_rate", float("nan")))
                        ),
                        "sample_6db_pass_rate": safe_float(
                            imp_gate.get("sample_6db_pass_rate", (ev.get("warmstart_metrics") or {}).get("sample_6db_pass_rate", float("nan")))
                        ),
                        "half_target_pass": safe_bool(half_gate.get("pass", False)),
                        "half_target_enabled": safe_bool(half_gate.get("enabled", False)),
                        "half_target_gap_db": safe_float(half_gate.get("gap_db", float("nan"))),
                        "half_target_threshold_db": safe_float(half_gate.get("threshold_db", float("nan"))),
                        "half_target_target_nr_db_mean": safe_float(half_gate.get("target_nr_db_mean", float("nan"))),
                        "half_target_init_nr_db_mean": safe_float(half_gate.get("init_nr_db_mean", float("nan"))),
                        "half_target_sample_pass_rate": safe_float(half_gate.get("sample_pass_rate", float("nan"))),
                        "fail_reasons": ev.get("fail_reasons", []),
                    }
                )
                level_results = ev.get("level_results", {})
                for level_key in ("level_1", "level_2", "level_3"):
                    lv = level_results.get(level_key, {}) if isinstance(level_results, dict) else {}
                    row[f"{level_key}_nr_db"] = safe_float(lv.get("nr_db", float("nan")))
            else:
                row["gate_status"] = "missing_eval"
                row["improvement_pass"] = False
                row["half_target_pass"] = False
                row["fail_reasons"] = ["missing_eval_summary"]

            rows.append(row)
    return rows


def aggregate_by_config(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cfg: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in seed_rows:
        by_cfg[str(r.get("config", "unknown"))].append(r)

    out: list[dict[str, Any]] = []
    for cfg, rows in sorted(by_cfg.items()):
        gate_pass = sum(1 for r in rows if str(r.get("gate_status", "")) == "passed")
        improvement_pass = sum(1 for r in rows if safe_bool(r.get("improvement_pass", False)))
        half_target_pass = sum(1 for r in rows if safe_bool(r.get("half_target_pass", False)))
        n = len(rows)

        mean_nr_vals = [safe_float(r.get("mean_nr_db", float("nan"))) for r in rows]
        ws_ratio_vals = [safe_float(r.get("warmstart_ratio", float("nan"))) for r in rows]
        imp_gap_vals = [safe_float(r.get("improvement_gap_db", float("nan"))) for r in rows]
        imp_early_gain_vals = [safe_float(r.get("improvement_early_gain_db_mean", float("nan"))) for r in rows]
        imp_sample_pass_vals = [safe_float(r.get("improvement_sample_pass_rate", float("nan"))) for r in rows]
        sample_6db_pass_vals = [safe_float(r.get("sample_6db_pass_rate", float("nan"))) for r in rows]
        imp_threshold_vals = [safe_float(r.get("improvement_threshold_db", float("nan"))) for r in rows]
        half_target_gap_vals = [safe_float(r.get("half_target_gap_db", float("nan"))) for r in rows]
        init_nr_vals = [safe_float(r.get("half_target_init_nr_db_mean", float("nan"))) for r in rows]
        target_nr_vals = [safe_float(r.get("half_target_target_nr_db_mean", float("nan"))) for r in rows]
        sample_half_pass_vals = [safe_float(r.get("half_target_sample_pass_rate", float("nan"))) for r in rows]
        l1_vals = [safe_float(r.get("level_1_nr_db", float("nan"))) for r in rows]
        l2_vals = [safe_float(r.get("level_2_nr_db", float("nan"))) for r in rows]
        l3_vals = [safe_float(r.get("level_3_nr_db", float("nan"))) for r in rows]

        def finite_list(vs: list[float]) -> list[float]:
            return [v for v in vs if np.isfinite(v)]

        mean_nr_m, mean_nr_s = mean_std(finite_list(mean_nr_vals))
        ws_m, ws_s = mean_std(finite_list(ws_ratio_vals))
        imp_gap_m, imp_gap_s = mean_std(finite_list(imp_gap_vals))
        imp_early_gain_m, imp_early_gain_s = mean_std(finite_list(imp_early_gain_vals))
        imp_sample_pass_m, _ = mean_std(finite_list(imp_sample_pass_vals))
        sample_6db_pass_m, _ = mean_std(finite_list(sample_6db_pass_vals))
        imp_threshold_m, _ = mean_std(finite_list(imp_threshold_vals))
        half_gap_m, half_gap_s = mean_std(finite_list(half_target_gap_vals))
        init_nr_m, _ = mean_std(finite_list(init_nr_vals))
        target_nr_m, _ = mean_std(finite_list(target_nr_vals))
        sample_half_pass_m, _ = mean_std(finite_list(sample_half_pass_vals))
        l1_m, l1_s = mean_std(finite_list(l1_vals))
        l2_m, l2_s = mean_std(finite_list(l2_vals))
        l3_m, l3_s = mean_std(finite_list(l3_vals))

        reasons = flatten_fail_reasons(rows)
        top_reasons = [f"{k}:{v}" for k, v in reasons.most_common(6)]

        sample = rows[0]
        out.append(
            {
                "config": cfg,
                "ablation_tag": sample.get("ablation_tag", ""),
                "feature_encoding": sample.get("feature_encoding", ""),
                "fusion_mode": sample.get("fusion_mode", ""),
                "disable_feature_b": sample.get("disable_feature_b", False),
                "loss_domain": sample.get("loss_domain", ""),
                "lambda_reg": safe_float(sample.get("lambda_reg", float("nan"))),
                "num_seeds": n,
                "gate_pass_count": gate_pass,
                "gate_pass_rate": float(gate_pass / n) if n > 0 else float("nan"),
                "improvement_enabled": safe_bool(sample.get("improvement_enabled", False)),
                "improvement_threshold_db": imp_threshold_m,
                "improvement_pass_count": improvement_pass,
                "improvement_pass_rate": float(improvement_pass / n) if n > 0 else float("nan"),
                "improvement_gap_db_mean": imp_gap_m,
                "improvement_gap_db_std": imp_gap_s,
                "improvement_early_gain_db_mean": imp_early_gain_m,
                "improvement_early_gain_db_std": imp_early_gain_s,
                "improvement_sample_pass_rate_mean": imp_sample_pass_m,
                "sample_6db_pass_rate_mean": sample_6db_pass_m,
                "half_target_pass_count": half_target_pass,
                "half_target_pass_rate": float(half_target_pass / n) if n > 0 else float("nan"),
                "half_target_gap_db_mean": half_gap_m,
                "half_target_gap_db_std": half_gap_s,
                "half_target_sample_pass_rate_mean": sample_half_pass_m,
                "init_nr_db_mean": init_nr_m,
                "target_nr_db_mean": target_nr_m,
                "mean_nr_db_mean": mean_nr_m,
                "mean_nr_db_std": mean_nr_s,
                "mean_nr_db_ci95": ci95(mean_nr_s, len(finite_list(mean_nr_vals))),
                "warmstart_ratio_mean": ws_m,
                "warmstart_ratio_std": ws_s,
                "warmstart_ratio_ci95": ci95(ws_s, len(finite_list(ws_ratio_vals))),
                "level_1_nr_db_mean": l1_m,
                "level_1_nr_db_std": l1_s,
                "level_2_nr_db_mean": l2_m,
                "level_2_nr_db_std": l2_s,
                "level_3_nr_db_mean": l3_m,
                "level_3_nr_db_std": l3_s,
                "top_fail_reasons": " | ".join(top_reasons),
            }
        )

    def _sortable(v: Any) -> float:
        fv = safe_float(v, default=float("-inf"))
        return fv if np.isfinite(fv) else float("-inf")

    out.sort(
        key=lambda r: (
            _sortable(r.get("improvement_pass_rate", float("-inf"))),
            _sortable(r.get("improvement_gap_db_mean", float("-inf"))),
            _sortable(r.get("improvement_sample_pass_rate_mean", float("-inf"))),
            _sortable(r.get("half_target_pass_rate", float("-inf"))),
            _sortable(r.get("gate_pass_rate", float("-inf"))),
            _sortable(r.get("mean_nr_db_mean", float("-inf"))),
            _sortable(r.get("warmstart_ratio_mean", float("-inf"))),
        ),
        reverse=True,
    )
    for idx, r in enumerate(out, start=1):
        r["rank"] = idx
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    args = parse_args()
    results_root = Path(args.results_root)
    out_dir = Path(args.output_dir) if args.output_dir else results_root / "_aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_rows = collect_seed_rows(results_root)
    config_rows = aggregate_by_config(seed_rows)

    seed_csv = out_dir / "seed_level_metrics.csv"
    group_csv = out_dir / "group_summary_ranked.csv"
    summary_json = out_dir / "summary.json"

    write_csv(seed_csv, seed_rows)
    write_csv(group_csv, config_rows)

    payload = {
        "results_root": str(results_root),
        "num_seed_rows": len(seed_rows),
        "num_groups": len(config_rows),
        "top_group": config_rows[0] if config_rows else None,
        "files": {
            "seed_level_metrics_csv": str(seed_csv),
            "group_summary_ranked_csv": str(group_csv),
        },
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
