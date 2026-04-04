from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def run_cmd(cmd: list[str], cwd: Path) -> tuple[int, str]:
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t0
    header = f"[exit={proc.returncode} elapsed_s={elapsed:.3f}]"
    return int(proc.returncode), header + "\n" + proc.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative 2+1 rounds of Hybrid ablation optimization: coarse, coarse-focus, full."
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--results-root", type=str, required=True)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--curriculum-levels", type=str, default="1,2,3")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--nr-margin-weight", type=float, default=0.0)
    parser.add_argument("--nr-target-ratio", type=float, default=0.5)
    parser.add_argument("--nr-margin-warmup-epochs", type=int, default=0)
    parser.add_argument("--nr-margin-mode", choices=("power", "db"), default="power")
    parser.add_argument("--nr-margin-focus-ratio", type=float, default=1.0)
    parser.add_argument("--wopt-supervision-weight", type=float, default=0.0)
    parser.add_argument("--supervision-weight", type=float, default=None)
    parser.add_argument("--supervision-source", choices=("none", "w_opt", "w_full"), default="w_opt")
    parser.add_argument("--use-canonical-prior", action="store_true")
    parser.add_argument("--canonical-prior-scale", type=float, default=1.0)
    parser.add_argument("--canonical-residual-l2-weight", type=float, default=0.0)
    parser.add_argument("--residual-head-zero-init", action="store_true")
    parser.add_argument("--warmstart-cases", type=int, default=8)
    parser.add_argument("--warmstart-level", type=int, default=3)
    parser.add_argument("--early-window-s", type=float, default=0.25)
    parser.add_argument("--half-target-ratio", type=float, default=0.5)
    parser.add_argument("--min-improvement-db", type=float, default=6.0)
    parser.add_argument("--eval-split", choices=("test", "val", "train", "all"), default="test")
    parser.add_argument("--disable-improvement-gate", action="store_true")

    parser.add_argument("--min-level1-samples", type=int, default=128)
    parser.add_argument("--min-level23-samples", type=int, default=512)
    parser.add_argument("--min-qc-nr-last-p10-db", type=float, default=12.0)
    parser.add_argument("--min-qc-nr-gain-p10-db", type=float, default=12.0)

    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--coarse-seeds", type=str, default="7,42")
    parser.add_argument("--full-seeds", type=str, default="7,42,123,999,20260403")

    parser.add_argument("--round1-epochs-per-level", type=str, default="4,4,6")
    parser.add_argument("--round1-max-train-samples", type=int, default=512)

    parser.add_argument("--round2-epochs-per-level", type=str, default="8,8,12")
    parser.add_argument("--round2-max-train-samples", type=int, default=1024)

    parser.add_argument("--round3-epochs-per-level", type=str, default="20,20,30")

    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_suite(
    *,
    py: str,
    suite_script: Path,
    h5_path: str | None,
    results_root: Path,
    seeds: str,
    device: str,
    batch_size: int,
    curriculum_levels: str,
    epochs_per_level: str,
    embed_dim: int,
    num_heads: int,
    val_frac: float,
    test_frac: float,
    lr: float,
    weight_decay: float,
    nr_margin_weight: float,
    nr_target_ratio: float,
    nr_margin_warmup_epochs: int,
    nr_margin_mode: str,
    nr_margin_focus_ratio: float,
    wopt_supervision_weight: float,
    supervision_weight: float | None,
    supervision_source: str,
    use_canonical_prior: bool,
    canonical_prior_scale: float,
    canonical_residual_l2_weight: float,
    residual_head_zero_init: bool,
    warmstart_cases: int,
    warmstart_level: int,
    early_window_s: float,
    half_target_ratio: float,
    min_improvement_db: float,
    eval_split: str,
    disable_improvement_gate: bool,
    min_level1_samples: int,
    min_level23_samples: int,
    min_qc_nr_last_p10_db: float,
    min_qc_nr_gain_p10_db: float,
    max_train_samples: int,
    config_names: list[str],
    skip_existing: bool,
) -> tuple[int, str]:
    cmd = [
        py,
        str(suite_script),
        "--python-exe",
        py,
        "--results-root",
        str(results_root),
        "--seeds",
        str(seeds),
        "--device",
        str(device),
        "--batch-size",
        str(batch_size),
        "--curriculum-levels",
        str(curriculum_levels),
        "--epochs-per-level",
        str(epochs_per_level),
        "--embed-dim",
        str(embed_dim),
        "--num-heads",
        str(num_heads),
        "--val-frac",
        str(val_frac),
        "--test-frac",
        str(test_frac),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--nr-margin-weight",
        str(nr_margin_weight),
        "--nr-target-ratio",
        str(nr_target_ratio),
        "--nr-margin-warmup-epochs",
        str(nr_margin_warmup_epochs),
        "--nr-margin-mode",
        str(nr_margin_mode),
        "--nr-margin-focus-ratio",
        str(nr_margin_focus_ratio),
        "--wopt-supervision-weight",
        str(wopt_supervision_weight),
        "--supervision-source",
        str(supervision_source),
        "--canonical-prior-scale",
        str(canonical_prior_scale),
        "--canonical-residual-l2-weight",
        str(canonical_residual_l2_weight),
        "--warmstart-cases",
        str(warmstart_cases),
        "--warmstart-level",
        str(warmstart_level),
        "--early-window-s",
        str(early_window_s),
        "--half-target-ratio",
        str(half_target_ratio),
        "--min-improvement-db",
        str(min_improvement_db),
        "--eval-split",
        str(eval_split),
        "--min-level1-samples",
        str(min_level1_samples),
        "--min-level23-samples",
        str(min_level23_samples),
        "--min-qc-nr-last-p10-db",
        str(min_qc_nr_last_p10_db),
        "--min-qc-nr-gain-p10-db",
        str(min_qc_nr_gain_p10_db),
    ]
    if bool(disable_improvement_gate):
        cmd.append("--disable-improvement-gate")
    if bool(use_canonical_prior):
        cmd.append("--use-canonical-prior")
    if bool(residual_head_zero_init):
        cmd.append("--residual-head-zero-init")
    if supervision_weight is not None:
        cmd.extend(["--supervision-weight", str(supervision_weight)])
    if h5_path:
        cmd.extend(["--h5-path", str(h5_path)])
    if max_train_samples > 0:
        cmd.extend(["--max-train-samples", str(max_train_samples)])
    if config_names:
        cmd.extend(["--config-names", ",".join(config_names)])
    if bool(skip_existing):
        cmd.append("--skip-existing")
    return run_cmd(cmd, cwd=ROOT)


def run_aggregate(py: str, agg_script: Path, results_root: Path) -> tuple[int, str]:
    cmd = [py, str(agg_script), "--results-root", str(results_root)]
    return run_cmd(cmd, cwd=ROOT)


def read_top_configs(group_summary_csv: Path, top_k: int) -> list[str]:
    if not group_summary_csv.exists():
        return []

    rows: list[dict[str, str]] = []
    with group_summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def rank_of(r: dict[str, str]) -> int:
        try:
            return int(r.get("rank", "999999"))
        except Exception:
            return 999999

    rows.sort(key=rank_of)
    return [str(r.get("config", "")).strip() for r in rows[: max(1, int(top_k))] if str(r.get("config", "")).strip()]


def main() -> int:
    args = parse_args()

    py = str(Path(args.python_exe))
    suite_script = ROOT / "python_impl" / "python_scripts" / "run_hybrid_ablation_suite.py"
    agg_script = ROOT / "python_impl" / "python_scripts" / "aggregate_hybrid_ablation_results.py"

    root = Path(args.results_root)
    round1_root = root / "round1_coarse"
    round2_root = root / "round2_coarse_focus"
    round3_root = root / "round3_full"

    root.mkdir(parents=True, exist_ok=True)

    logs: dict[str, Any] = {
        "root": str(root),
        "rounds": [],
    }

    print("[Round1] coarse screening on full matrix")
    rc, out = run_suite(
        py=py,
        suite_script=suite_script,
        h5_path=args.h5_path,
        results_root=round1_root,
        seeds=str(args.coarse_seeds),
        device=str(args.device),
        batch_size=int(args.batch_size),
        curriculum_levels=str(args.curriculum_levels),
        epochs_per_level=str(args.round1_epochs_per_level),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        nr_margin_weight=float(args.nr_margin_weight),
        nr_target_ratio=float(args.nr_target_ratio),
        nr_margin_warmup_epochs=int(args.nr_margin_warmup_epochs),
        nr_margin_mode=str(args.nr_margin_mode),
        nr_margin_focus_ratio=float(args.nr_margin_focus_ratio),
        wopt_supervision_weight=float(args.wopt_supervision_weight),
        supervision_weight=None if args.supervision_weight is None else float(args.supervision_weight),
        supervision_source=str(args.supervision_source),
        use_canonical_prior=bool(args.use_canonical_prior),
        canonical_prior_scale=float(args.canonical_prior_scale),
        canonical_residual_l2_weight=float(args.canonical_residual_l2_weight),
        residual_head_zero_init=bool(args.residual_head_zero_init),
        warmstart_cases=int(args.warmstart_cases),
        warmstart_level=int(args.warmstart_level),
        early_window_s=float(args.early_window_s),
        half_target_ratio=float(args.half_target_ratio),
        min_improvement_db=float(args.min_improvement_db),
        eval_split=str(args.eval_split),
        disable_improvement_gate=bool(args.disable_improvement_gate),
        min_level1_samples=int(args.min_level1_samples),
        min_level23_samples=int(args.min_level23_samples),
        min_qc_nr_last_p10_db=float(args.min_qc_nr_last_p10_db),
        min_qc_nr_gain_p10_db=float(args.min_qc_nr_gain_p10_db),
        max_train_samples=int(args.round1_max_train_samples),
        config_names=[],
        skip_existing=bool(args.skip_existing),
    )
    (round1_root / "iterative_round.log.txt").write_text(out, encoding="utf-8")
    logs["rounds"].append({"round": "round1_coarse", "results_root": str(round1_root), "suite_exit_code": rc})
    if rc != 0:
        print("[FAIL] round1 suite failed")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    rc_agg, out_agg = run_aggregate(py=py, agg_script=agg_script, results_root=round1_root)
    (round1_root / "aggregate_round.log.txt").write_text(out_agg, encoding="utf-8")
    logs["rounds"][-1]["aggregate_exit_code"] = rc_agg
    if rc_agg != 0:
        print("[FAIL] round1 aggregate failed")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    round1_top = read_top_configs(round1_root / "_aggregate" / "group_summary_ranked.csv", int(args.top_k))
    if not round1_top:
        print("[FAIL] round1 produced no ranked configs")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    print(f"[Round1] selected top configs: {round1_top}")

    print("[Round2] coarse-focus on round1 top-k")
    rc2, out2 = run_suite(
        py=py,
        suite_script=suite_script,
        h5_path=args.h5_path,
        results_root=round2_root,
        seeds=str(args.coarse_seeds),
        device=str(args.device),
        batch_size=int(args.batch_size),
        curriculum_levels=str(args.curriculum_levels),
        epochs_per_level=str(args.round2_epochs_per_level),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        nr_margin_weight=float(args.nr_margin_weight),
        nr_target_ratio=float(args.nr_target_ratio),
        nr_margin_warmup_epochs=int(args.nr_margin_warmup_epochs),
        nr_margin_mode=str(args.nr_margin_mode),
        nr_margin_focus_ratio=float(args.nr_margin_focus_ratio),
        wopt_supervision_weight=float(args.wopt_supervision_weight),
        supervision_weight=None if args.supervision_weight is None else float(args.supervision_weight),
        supervision_source=str(args.supervision_source),
        use_canonical_prior=bool(args.use_canonical_prior),
        canonical_prior_scale=float(args.canonical_prior_scale),
        canonical_residual_l2_weight=float(args.canonical_residual_l2_weight),
        residual_head_zero_init=bool(args.residual_head_zero_init),
        warmstart_cases=int(args.warmstart_cases),
        warmstart_level=int(args.warmstart_level),
        early_window_s=float(args.early_window_s),
        half_target_ratio=float(args.half_target_ratio),
        min_improvement_db=float(args.min_improvement_db),
        eval_split=str(args.eval_split),
        disable_improvement_gate=bool(args.disable_improvement_gate),
        min_level1_samples=int(args.min_level1_samples),
        min_level23_samples=int(args.min_level23_samples),
        min_qc_nr_last_p10_db=float(args.min_qc_nr_last_p10_db),
        min_qc_nr_gain_p10_db=float(args.min_qc_nr_gain_p10_db),
        max_train_samples=int(args.round2_max_train_samples),
        config_names=round1_top,
        skip_existing=bool(args.skip_existing),
    )
    (round2_root / "iterative_round.log.txt").write_text(out2, encoding="utf-8")
    logs["rounds"].append({"round": "round2_coarse_focus", "results_root": str(round2_root), "suite_exit_code": rc2, "input_configs": round1_top})
    if rc2 != 0:
        print("[FAIL] round2 suite failed")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    rc2_agg, out2_agg = run_aggregate(py=py, agg_script=agg_script, results_root=round2_root)
    (round2_root / "aggregate_round.log.txt").write_text(out2_agg, encoding="utf-8")
    logs["rounds"][-1]["aggregate_exit_code"] = rc2_agg
    if rc2_agg != 0:
        print("[FAIL] round2 aggregate failed")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    round2_top = read_top_configs(round2_root / "_aggregate" / "group_summary_ranked.csv", int(args.top_k))
    if not round2_top:
        print("[FAIL] round2 produced no ranked configs")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    print(f"[Round2] selected top configs for full run: {round2_top}")

    print("[Round3] full evaluation on round2 top-k")
    rc3, out3 = run_suite(
        py=py,
        suite_script=suite_script,
        h5_path=args.h5_path,
        results_root=round3_root,
        seeds=str(args.full_seeds),
        device=str(args.device),
        batch_size=int(args.batch_size),
        curriculum_levels=str(args.curriculum_levels),
        epochs_per_level=str(args.round3_epochs_per_level),
        embed_dim=int(args.embed_dim),
        num_heads=int(args.num_heads),
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        nr_margin_weight=float(args.nr_margin_weight),
        nr_target_ratio=float(args.nr_target_ratio),
        nr_margin_warmup_epochs=int(args.nr_margin_warmup_epochs),
        nr_margin_mode=str(args.nr_margin_mode),
        nr_margin_focus_ratio=float(args.nr_margin_focus_ratio),
        wopt_supervision_weight=float(args.wopt_supervision_weight),
        supervision_weight=None if args.supervision_weight is None else float(args.supervision_weight),
        supervision_source=str(args.supervision_source),
        use_canonical_prior=bool(args.use_canonical_prior),
        canonical_prior_scale=float(args.canonical_prior_scale),
        canonical_residual_l2_weight=float(args.canonical_residual_l2_weight),
        residual_head_zero_init=bool(args.residual_head_zero_init),
        warmstart_cases=int(args.warmstart_cases),
        warmstart_level=int(args.warmstart_level),
        early_window_s=float(args.early_window_s),
        half_target_ratio=float(args.half_target_ratio),
        min_improvement_db=float(args.min_improvement_db),
        eval_split=str(args.eval_split),
        disable_improvement_gate=bool(args.disable_improvement_gate),
        min_level1_samples=int(args.min_level1_samples),
        min_level23_samples=int(args.min_level23_samples),
        min_qc_nr_last_p10_db=float(args.min_qc_nr_last_p10_db),
        min_qc_nr_gain_p10_db=float(args.min_qc_nr_gain_p10_db),
        max_train_samples=0,
        config_names=round2_top,
        skip_existing=bool(args.skip_existing),
    )
    (round3_root / "iterative_round.log.txt").write_text(out3, encoding="utf-8")
    logs["rounds"].append({"round": "round3_full", "results_root": str(round3_root), "suite_exit_code": rc3, "input_configs": round2_top})
    if rc3 != 0:
        print("[FAIL] round3 suite failed")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    rc3_agg, out3_agg = run_aggregate(py=py, agg_script=agg_script, results_root=round3_root)
    (round3_root / "aggregate_round.log.txt").write_text(out3_agg, encoding="utf-8")
    logs["rounds"][-1]["aggregate_exit_code"] = rc3_agg
    if rc3_agg != 0:
        print("[FAIL] round3 aggregate failed")
        (root / "iterative_summary.json").write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    final_top = read_top_configs(round3_root / "_aggregate" / "group_summary_ranked.csv", int(args.top_k))
    logs["final_top_configs"] = final_top
    logs["status"] = "completed"

    out_json = root / "iterative_summary.json"
    out_json.write_text(json.dumps(logs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "completed", "iterative_summary": str(out_json), "final_top_configs": final_top}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
