from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "python_impl") not in sys.path:
    sys.path.insert(0, str(ROOT / "python_impl"))

from python_scripts.train_hybrid_deep_fxlms_single_control import resolve_h5_path


@dataclass
class AblationConfig:
    name: str
    feature_encoding: str
    fusion_mode: str
    disable_feature_b: bool
    loss_domain: str
    lambda_reg: float
    basis_dim: int


def parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


def format_float_compact(v: float) -> str:
    s = f"{v:.0e}" if v < 0.01 else f"{v:g}"
    return s.replace("+", "").replace(".", "p")


def standard_ablation_matrix() -> list[AblationConfig]:
    rows = [
        AblationConfig("b01_ri_cross_freq_lam1e3_b32", "ri", "cross", False, "freq", 1.0e-3, 32),
        AblationConfig("b02_ri_cat_freq_lam1e3_b32", "ri", "cat", False, "freq", 1.0e-3, 32),
        AblationConfig("b03_mp_cross_freq_lam1e3_b32", "mp", "cross", False, "freq", 1.0e-3, 32),
        AblationConfig("b04_mp_cat_freq_lam1e3_b32", "mp", "cat", False, "freq", 1.0e-3, 32),
        AblationConfig("b05_fboff_cat_freq_lam1e3_b32", "ri", "cat", True, "freq", 1.0e-3, 32),
        AblationConfig("b06_ri_cross_time_lam1e3_b32", "ri", "cross", False, "time", 1.0e-3, 32),
        AblationConfig("b07_ri_cross_freq_lam1e4_b32", "ri", "cross", False, "freq", 1.0e-4, 32),
        AblationConfig("b08_ri_cross_freq_lam3e3_b32", "ri", "cross", False, "freq", 3.0e-3, 32),
        AblationConfig("b09_ri_cross_freq_lam1e3_b16", "ri", "cross", False, "freq", 1.0e-3, 16),
        AblationConfig("b10_ri_cross_freq_lam1e3_b64", "ri", "cross", False, "freq", 1.0e-3, 64),
        AblationConfig("b11_mp_cross_time_lam1e3_b32", "mp", "cross", False, "time", 1.0e-3, 32),
        AblationConfig("b12_fboff_cat_time_lam1e3_b32", "ri", "cat", True, "time", 1.0e-3, 32),
    ]
    valid: list[AblationConfig] = []
    for r in rows:
        if r.disable_feature_b and r.fusion_mode == "cross":
            continue
        valid.append(r)
    return valid


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
    return proc.returncode, header + "\n" + proc.stdout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard Hybrid Deep-FxLMS ablation suite with multiseed training/evaluation.")
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--h5-path", type=str, default=None)
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="7,42,123,999,20260403")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--curriculum-levels", type=str, default="1,2,3")
    parser.add_argument("--epochs-per-level", type=str, default="20,20,30")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--nr-margin-weight", type=float, default=0.0)
    parser.add_argument("--nr-target-ratio", type=float, default=0.5)
    parser.add_argument("--nr-margin-warmup-epochs", type=int, default=0)
    parser.add_argument("--nr-margin-mode", choices=("power", "db"), default="power")
    parser.add_argument("--nr-margin-focus-ratio", type=float, default=1.0)
    parser.add_argument("--wopt-supervision-weight", type=float, default=0.0)
    parser.add_argument("--supervision-weight", type=float, default=None)
    parser.add_argument("--supervision-source", choices=("none", "w_opt", "w_full"), default="w_opt")
    parser.add_argument("--acoustic-loss-weight", type=float, default=1.0)
    parser.add_argument("--use-path-features", action="store_true")
    parser.add_argument("--use-index-embedding", action="store_true")
    parser.add_argument("--index-direct-lookup", action="store_true")
    parser.add_argument("--index-direct-init-wopt", action="store_true")
    parser.add_argument("--index-direct-init-source", choices=("none", "w_opt", "w_full"), default="none")
    parser.add_argument("--index-direct-freeze", action="store_true")
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
    parser.add_argument("--disable-half-target-gate", action="store_true")
    parser.add_argument("--eval-only-existing", action="store_true")
    parser.add_argument("--force-rerun-eval", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--config-names", type=str, default="")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--min-level1-samples", type=int, default=128)
    parser.add_argument("--min-level23-samples", type=int, default=512)
    parser.add_argument("--min-qc-nr-last-p10-db", type=float, default=12.0)
    parser.add_argument("--min-qc-nr-gain-p10-db", type=float, default=12.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    py = str(Path(args.python_exe))
    train_script = ROOT / "python_impl" / "python_scripts" / "train_hybrid_deep_fxlms_single_control.py"
    eval_script = ROOT / "python_impl" / "python_scripts" / "evaluate_hybrid_deep_fxlms_single_control.py"
    h5_path = resolve_h5_path(args.h5_path)
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    matrix = standard_ablation_matrix()
    requested_names = [v.strip() for v in str(args.config_names).split(",") if v.strip()]
    if requested_names:
        name_set = set(requested_names)
        matrix = [cfg for cfg in matrix if cfg.name in name_set]
        if not matrix:
            raise ValueError(f"No valid configs matched --config-names={requested_names}")
    if int(args.max_configs) > 0:
        matrix = matrix[: int(args.max_configs)]

    run_manifest: dict[str, Any] = {
        "h5_path": str(h5_path),
        "results_root": str(results_root),
        "seeds": seeds,
        "nr_margin_weight": float(args.nr_margin_weight),
        "nr_target_ratio": float(args.nr_target_ratio),
        "nr_margin_warmup_epochs": int(args.nr_margin_warmup_epochs),
        "nr_margin_mode": str(args.nr_margin_mode),
        "nr_margin_focus_ratio": float(args.nr_margin_focus_ratio),
        "wopt_supervision_weight": float(args.wopt_supervision_weight),
        "supervision_source": str(args.supervision_source),
        "supervision_weight": None if args.supervision_weight is None else float(args.supervision_weight),
        "acoustic_loss_weight": float(args.acoustic_loss_weight),
        "val_frac": float(args.val_frac),
        "test_frac": float(args.test_frac),
        "grad_clip_norm": float(args.grad_clip_norm),
        "use_path_features": bool(args.use_path_features),
        "use_index_embedding": bool(args.use_index_embedding),
        "index_direct_lookup": bool(args.index_direct_lookup),
        "index_direct_init_wopt": bool(args.index_direct_init_wopt),
        "index_direct_init_source": str(args.index_direct_init_source),
        "index_direct_freeze": bool(args.index_direct_freeze),
        "use_canonical_prior": bool(args.use_canonical_prior),
        "canonical_prior_scale": float(args.canonical_prior_scale),
        "canonical_residual_l2_weight": float(args.canonical_residual_l2_weight),
        "residual_head_zero_init": bool(args.residual_head_zero_init),
        "eval_split": str(args.eval_split),
        "half_target_ratio": float(args.half_target_ratio),
        "min_improvement_db": float(args.min_improvement_db),
        "disable_improvement_gate": bool(args.disable_improvement_gate),
        "disable_half_target_gate": bool(args.disable_half_target_gate),
        "num_configs": len(matrix),
        "configs": [asdict(c) for c in matrix],
        "runs": [],
        "invalid_filtered": [],
    }

    for cfg in matrix:
        cfg_dir = results_root / cfg.name
        cfg_dir.mkdir(parents=True, exist_ok=True)

        for seed in seeds:
            seed_dir = cfg_dir / f"seed_{seed}"
            train_dir = seed_dir / "train"
            eval_dir = seed_dir / "eval"
            seed_dir.mkdir(parents=True, exist_ok=True)
            train_summary = train_dir / "summary.json"
            eval_summary = eval_dir / "summary.json"
            run_info: dict[str, Any] = {
                "config": cfg.name,
                "seed": int(seed),
                "train_dir": str(train_dir),
                "eval_dir": str(eval_dir),
                "status": "pending",
            }

            if bool(args.skip_existing) and train_summary.exists() and eval_summary.exists():
                run_info["status"] = "skipped_existing"
                run_manifest["runs"].append(run_info)
                print(f"[SKIP] {cfg.name} seed={seed} (existing)")
                continue

            eval_only_existing = bool(args.eval_only_existing)
            force_rerun_eval = bool(args.force_rerun_eval)

            if eval_only_existing:
                ckpt = train_dir / "final_hybrid_deep_fxlms.pt"
                if not ckpt.exists():
                    run_info["status"] = "train_checkpoint_missing"
                    run_manifest["runs"].append(run_info)
                    print(f"[SKIP] checkpoint missing {cfg.name} seed={seed}")
                    continue

                if bool(args.skip_existing) and eval_summary.exists() and not force_rerun_eval:
                    run_info["status"] = "skipped_existing_eval"
                    run_manifest["runs"].append(run_info)
                    print(f"[SKIP] {cfg.name} seed={seed} (existing eval)")
                    continue

                eval_cmd = [
                    py,
                    str(eval_script),
                    "--checkpoint-path",
                    str(ckpt),
                    "--h5-path",
                    str(h5_path),
                    "--output-dir",
                    str(eval_dir),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    str(args.device),
                    "--warmstart-cases",
                    str(args.warmstart_cases),
                    "--warmstart-level",
                    str(args.warmstart_level),
                    "--early-window-s",
                    str(args.early_window_s),
                    "--half-target-ratio",
                    str(args.half_target_ratio),
                    "--min-improvement-db",
                    str(args.min_improvement_db),
                    "--eval-split",
                    str(args.eval_split),
                ]
                if bool(args.disable_improvement_gate):
                    eval_cmd.append("--disable-improvement-gate")
                if bool(args.disable_half_target_gate):
                    eval_cmd.append("--disable-half-target-gate")

                print(f"[EVAL-ONLY] {cfg.name} seed={seed}")
                rc_eval, eval_log = run_cmd(eval_cmd, cwd=ROOT)
                (seed_dir / "eval.log.txt").write_text(eval_log, encoding="utf-8")
                run_info["train_exit_code"] = None
                run_info["eval_exit_code"] = int(rc_eval)
                run_info["status"] = "done" if rc_eval == 0 else "eval_failed"
                run_manifest["runs"].append(run_info)
                if rc_eval != 0:
                    print(f"[FAIL] eval {cfg.name} seed={seed}")
                continue

            train_cmd = [
                py,
                str(train_script),
                "--h5-path",
                str(h5_path),
                "--output-dir",
                str(train_dir),
                "--curriculum-levels",
                str(args.curriculum_levels),
                "--epochs-per-level",
                str(args.epochs_per_level),
                "--batch-size",
                str(args.batch_size),
                "--val-frac",
                str(args.val_frac),
                "--test-frac",
                str(args.test_frac),
                "--seed",
                str(seed),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--grad-clip-norm",
                str(args.grad_clip_norm),
                "--nr-margin-weight",
                str(args.nr_margin_weight),
                "--nr-target-ratio",
                str(args.nr_target_ratio),
                "--nr-margin-warmup-epochs",
                str(args.nr_margin_warmup_epochs),
                "--nr-margin-mode",
                str(args.nr_margin_mode),
                "--nr-margin-focus-ratio",
                str(args.nr_margin_focus_ratio),
                "--wopt-supervision-weight",
                str(args.wopt_supervision_weight),
                "--supervision-source",
                str(args.supervision_source),
                "--acoustic-loss-weight",
                str(args.acoustic_loss_weight),
                "--lambda-reg",
                str(cfg.lambda_reg),
                "--loss-domain",
                str(cfg.loss_domain),
                "--device",
                str(args.device),
                "--embed-dim",
                str(args.embed_dim),
                "--num-heads",
                str(args.num_heads),
                "--basis-dim",
                str(cfg.basis_dim),
                "--fusion-mode",
                str(cfg.fusion_mode),
                "--feature-encoding",
                str(cfg.feature_encoding),
                "--ablation-tag",
                str(cfg.name),
                "--min-level1-samples",
                str(args.min_level1_samples),
                "--min-level23-samples",
                str(args.min_level23_samples),
                "--min-qc-nr-last-p10-db",
                str(args.min_qc_nr_last_p10_db),
                "--min-qc-nr-gain-p10-db",
                str(args.min_qc_nr_gain_p10_db),
            ]
            if args.supervision_weight is not None:
                train_cmd.extend(["--supervision-weight", str(args.supervision_weight)])
            if cfg.disable_feature_b:
                train_cmd.append("--disable-feature-b")
            if bool(args.use_path_features):
                train_cmd.append("--use-path-features")
            if bool(args.use_index_embedding):
                train_cmd.append("--use-index-embedding")
            if bool(args.index_direct_lookup):
                train_cmd.append("--index-direct-lookup")
            if str(args.index_direct_init_source) != "none":
                train_cmd.extend(["--index-direct-init-source", str(args.index_direct_init_source)])
            if bool(args.index_direct_init_wopt):
                train_cmd.append("--index-direct-init-wopt")
            if bool(args.index_direct_freeze):
                train_cmd.append("--index-direct-freeze")
            if bool(args.use_canonical_prior):
                train_cmd.append("--use-canonical-prior")
            train_cmd.extend(["--canonical-prior-scale", str(args.canonical_prior_scale)])
            train_cmd.extend(["--canonical-residual-l2-weight", str(args.canonical_residual_l2_weight)])
            if bool(args.residual_head_zero_init):
                train_cmd.append("--residual-head-zero-init")
            if int(args.max_train_samples) > 0:
                train_cmd.extend(["--max-train-samples", str(args.max_train_samples)])

            print(f"[TRAIN] {cfg.name} seed={seed}")
            rc_train, train_log = run_cmd(train_cmd, cwd=ROOT)
            (seed_dir / "train.log.txt").write_text(train_log, encoding="utf-8")
            run_info["train_exit_code"] = int(rc_train)
            if rc_train != 0:
                run_info["status"] = "train_failed"
                run_manifest["runs"].append(run_info)
                print(f"[FAIL] train {cfg.name} seed={seed}")
                continue

            ckpt = train_dir / "final_hybrid_deep_fxlms.pt"
            if not ckpt.exists():
                run_info["status"] = "train_checkpoint_missing"
                run_manifest["runs"].append(run_info)
                print(f"[FAIL] checkpoint missing {cfg.name} seed={seed}")
                continue

            eval_cmd = [
                py,
                str(eval_script),
                "--checkpoint-path",
                str(ckpt),
                "--h5-path",
                str(h5_path),
                "--output-dir",
                str(eval_dir),
                "--batch-size",
                str(args.batch_size),
                "--device",
                str(args.device),
                "--warmstart-cases",
                str(args.warmstart_cases),
                "--warmstart-level",
                str(args.warmstart_level),
                "--early-window-s",
                str(args.early_window_s),
                "--half-target-ratio",
                str(args.half_target_ratio),
                "--min-improvement-db",
                str(args.min_improvement_db),
                "--eval-split",
                str(args.eval_split),
            ]
            if bool(args.disable_improvement_gate):
                eval_cmd.append("--disable-improvement-gate")
            if bool(args.disable_half_target_gate):
                eval_cmd.append("--disable-half-target-gate")
            print(f"[EVAL] {cfg.name} seed={seed}")
            rc_eval, eval_log = run_cmd(eval_cmd, cwd=ROOT)
            (seed_dir / "eval.log.txt").write_text(eval_log, encoding="utf-8")
            run_info["eval_exit_code"] = int(rc_eval)
            run_info["status"] = "done" if rc_eval == 0 else "eval_failed"
            run_manifest["runs"].append(run_info)
            if rc_eval != 0:
                print(f"[FAIL] eval {cfg.name} seed={seed}")

    manifest_path = results_root / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(run_manifest["runs"])
    ok = sum(1 for r in run_manifest["runs"] if r.get("status") == "done")
    failed = total - ok
    print(json.dumps({"total_runs": total, "done": ok, "not_done": failed, "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
