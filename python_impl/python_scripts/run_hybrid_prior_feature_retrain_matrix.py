from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = ROOT / "python_impl" / "python_scripts" / "train_hybrid_deep_fxlms_single_control.py"
EVAL_SCRIPT = ROOT / "python_impl" / "python_scripts" / "evaluate_hybrid_deep_fxlms_single_control.py"
DEFAULT_IN_DOMAIN_H5 = ROOT / "python_impl" / "python_scripts" / "cfxlms_qc_dataset_single_control.h5"
DEFAULT_OOD_H5_DIR = ROOT / "python_impl" / "python_scripts"


def parse_csv_list(text: str) -> list[str]:
    return [v.strip() for v in str(text).split(",") if v.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(v.strip()) for v in str(text).split(",") if v.strip()]


def safe_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def resolve_path(path_text: str | None, base: Path = ROOT) -> Path | None:
    if path_text is None:
        return None
    p = Path(path_text)
    return p if p.is_absolute() else (base / p).resolve()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def run_subprocess(cmd: list[str], cwd: Path) -> tuple[int, str, str, float]:
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    elapsed = float(time.time() - t0)
    return int(proc.returncode), str(proc.stdout), str(proc.stderr), elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Small retraining ablation matrix focused on prior dominance and E2R/R2R channel contribution."
        )
    )
    parser.add_argument("--python-exe", type=str, default=sys.executable)
    parser.add_argument("--h5-path", type=str, default=str(DEFAULT_IN_DOMAIN_H5))
    parser.add_argument("--results-root", type=str, required=True)
    parser.add_argument("--seeds", type=str, default="7,42")

    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,no_prior,mask_e2r_r2r,no_prior_mask_e2r_r2r",
        help="Comma-separated subset of baseline,no_prior,mask_e2r_r2r,no_prior_mask_e2r_r2r",
    )
    parser.add_argument("--mask-channel-regex", type=str, default="^(E2R_|R2R_)")

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--curriculum-levels", type=str, default="1,2,3")
    parser.add_argument("--epochs-per-level", type=str, default="8,8,12")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)

    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--basis-dim", type=int, default=32)
    parser.add_argument("--fusion-mode", choices=("cross", "cat"), default="cross")
    parser.add_argument("--feature-encoding", choices=("ri", "mp"), default="ri")
    parser.add_argument("--disable-feature-b", action="store_true")
    parser.add_argument("--use-path-features", action="store_true")

    parser.add_argument("--lambda-reg", type=float, default=1.0e-3)
    parser.add_argument("--loss-domain", choices=("freq", "time"), default="freq")
    parser.add_argument("--acoustic-loss-weight", type=float, default=1.0)
    parser.add_argument("--nr-margin-weight", type=float, default=0.0)
    parser.add_argument("--nr-target-ratio", type=float, default=0.5)
    parser.add_argument("--nr-margin-warmup-epochs", type=int, default=0)
    parser.add_argument("--nr-margin-mode", choices=("power", "db"), default="power")
    parser.add_argument("--nr-margin-focus-ratio", type=float, default=1.0)
    parser.add_argument("--wopt-supervision-weight", type=float, default=0.0)
    parser.add_argument("--supervision-source", choices=("none", "w_opt", "w_full"), default="w_opt")
    parser.add_argument("--canonical-prior-scale", type=float, default=1.0)
    parser.add_argument("--canonical-residual-l2-weight", type=float, default=1.0e-4)
    parser.add_argument("--residual-head-zero-init", action="store_true", default=True)

    parser.add_argument("--max-train-samples", type=int, default=128)
    parser.add_argument("--min-level1-samples", type=int, default=128)
    parser.add_argument("--min-level23-samples", type=int, default=512)
    parser.add_argument("--min-qc-nr-last-p10-db", type=float, default=12.0)
    parser.add_argument("--min-qc-nr-gain-p10-db", type=float, default=12.0)

    parser.add_argument("--eval-domains", type=str, default="in_domain")
    parser.add_argument("--in-domain-h5", type=str, default=str(DEFAULT_IN_DOMAIN_H5))
    parser.add_argument("--ood-mild-h5", type=str, default=None)
    parser.add_argument("--ood-hard-h5", type=str, default=None)
    parser.add_argument("--ood-h5-dir", type=str, default=str(DEFAULT_OOD_H5_DIR))

    parser.add_argument("--warmstart-cases", type=int, default=8)
    parser.add_argument("--warmstart-level", type=int, default=3)
    parser.add_argument("--early-window-s", type=float, default=0.25)
    parser.add_argument("--target-metric", choices=("nr_last_db",), default="nr_last_db")
    parser.add_argument("--half-target-ratio", type=float, default=0.5)
    parser.add_argument("--min-improvement-db", type=float, default=6.0)
    parser.add_argument("--eval-split", choices=("test", "val", "train", "all"), default="test")
    parser.add_argument("--disable-improvement-gate", action="store_true")
    parser.add_argument("--disable-half-target-gate", action="store_true")

    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_domain_h5_paths(args: argparse.Namespace) -> dict[str, Path]:
    domains = parse_csv_list(args.eval_domains)
    allowed = {"in_domain", "ood_mild", "ood_hard"}
    unknown = [d for d in domains if d not in allowed]
    if unknown:
        raise ValueError(f"Unsupported eval-domain(s): {unknown}; allowed={sorted(allowed)}")

    ood_dir = resolve_path(args.ood_h5_dir)
    if ood_dir is None:
        raise ValueError("--ood-h5-dir is invalid")

    mapping = {
        "in_domain": resolve_path(args.in_domain_h5),
        "ood_mild": resolve_path(args.ood_mild_h5)
        if args.ood_mild_h5
        else (ood_dir / "cfxlms_qc_dataset_single_control_ood_mild.h5"),
        "ood_hard": resolve_path(args.ood_hard_h5)
        if args.ood_hard_h5
        else (ood_dir / "cfxlms_qc_dataset_single_control_ood_hard.h5"),
    }

    out: dict[str, Path] = {}
    for d in domains:
        p = mapping[d]
        if p is None:
            raise ValueError(f"Failed to resolve h5 path for domain={d}")
        if not p.exists():
            raise FileNotFoundError(f"Dataset missing for domain={d}: {p}")
        out[d] = p.resolve()
    return out


def variant_flags(variant: str, args: argparse.Namespace) -> dict[str, Any]:
    v = str(variant).strip().lower()
    if v == "baseline":
        return {"use_canonical_prior": True, "acoustic_zero_channel_regex": ""}
    if v == "no_prior":
        return {"use_canonical_prior": False, "acoustic_zero_channel_regex": ""}
    if v == "mask_e2r_r2r":
        return {"use_canonical_prior": True, "acoustic_zero_channel_regex": str(args.mask_channel_regex)}
    if v == "no_prior_mask_e2r_r2r":
        return {"use_canonical_prior": False, "acoustic_zero_channel_regex": str(args.mask_channel_regex)}
    raise ValueError(f"Unsupported variant: {variant}")


def read_eval_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return read_json(path)
    except Exception:
        return {}


def aggregate_rows(rows: list[dict[str, Any]], domains: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        grouped[str(r.get("variant", "unknown"))].append(r)

    out: list[dict[str, Any]] = []
    for variant, items in sorted(grouped.items()):
        n_runs = len(items)
        gate_pass = sum(1 for r in items if str(r.get("gate_status", "")).lower() == "passed")
        nr_vals = [safe_float(r.get("mean_nr_db", float("nan"))) for r in items]
        nr_vals = [v for v in nr_vals if np.isfinite(v)]
        gain_vals = [safe_float(r.get("warmstart_early_gain_db_mean", float("nan"))) for r in items]
        gain_vals = [v for v in gain_vals if np.isfinite(v)]

        domain_presence = {d: 0 for d in domains}
        for r in items:
            d = str(r.get("domain", ""))
            if d in domain_presence:
                domain_presence[d] += 1

        out.append(
            {
                "variant": variant,
                "num_eval_rows": int(n_runs),
                "gate_pass_count": int(gate_pass),
                "gate_pass_rate": float(gate_pass / n_runs) if n_runs > 0 else float("nan"),
                "mean_nr_db": float(np.mean(nr_vals)) if nr_vals else float("nan"),
                "warmstart_early_gain_db_mean": float(np.mean(gain_vals)) if gain_vals else float("nan"),
                "domains_covered": "|".join([f"{k}:{v}" for k, v in domain_presence.items()]),
            }
        )

    out.sort(
        key=lambda r: (
            safe_float(r.get("gate_pass_rate", float("nan")), default=float("-inf")),
            safe_float(r.get("mean_nr_db", float("nan")), default=float("-inf")),
            safe_float(r.get("warmstart_early_gain_db_mean", float("nan")), default=float("-inf")),
        ),
        reverse=True,
    )
    for i, r in enumerate(out, start=1):
        r["rank"] = int(i)
    return out


def main() -> int:
    args = parse_args()

    h5_path = resolve_path(args.h5_path)
    if h5_path is None or not h5_path.exists():
        raise FileNotFoundError(f"Train dataset does not exist: {args.h5_path}")

    results_root = resolve_path(args.results_root)
    if results_root is None:
        raise ValueError("Invalid --results-root")
    results_root.mkdir(parents=True, exist_ok=True)

    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("--seeds resolved to empty list")

    variants = parse_csv_list(args.variants)
    allowed_variants = {"baseline", "no_prior", "mask_e2r_r2r", "no_prior_mask_e2r_r2r"}
    bad_variants = [v for v in variants if v not in allowed_variants]
    if bad_variants:
        raise ValueError(f"Unsupported variants: {bad_variants}; allowed={sorted(allowed_variants)}")

    domain_h5 = resolve_domain_h5_paths(args)
    eval_domains = list(domain_h5.keys())

    manifest = {
        "timestamp": int(time.time()),
        "results_root": str(results_root),
        "h5_path": str(h5_path),
        "seeds": seeds,
        "variants": variants,
        "eval_domains": eval_domains,
        "domain_h5": {k: str(v) for k, v in domain_h5.items()},
        "train_script": str(TRAIN_SCRIPT),
        "eval_script": str(EVAL_SCRIPT),
        "python_exec": str(args.python_exe),
        "args": vars(args),
    }
    write_json(results_root / "run_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for variant in variants:
        vf = variant_flags(variant, args)
        for seed in seeds:
            seed_root = results_root / variant / f"seed_{seed}"
            train_dir = seed_root / "train"
            train_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = train_dir / "final_hybrid_deep_fxlms.pt"

            expected_eval_summaries = [seed_root / f"eval_{d}" / "summary.json" for d in eval_domains]
            if bool(args.skip_existing) and ckpt_path.exists() and all(p.exists() for p in expected_eval_summaries):
                for d in eval_domains:
                    summary = read_eval_summary(seed_root / f"eval_{d}" / "summary.json")
                    rows.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "domain": d,
                            "train_status": "skipped_existing",
                            "train_exit_code": 0,
                            "eval_exit_code": 0,
                            "gate_status": str(summary.get("gate_status", "unknown")),
                            "mean_nr_db": safe_float(summary.get("mean_nr_db", float("nan"))),
                            "warmstart_early_gain_db_mean": safe_float((summary.get("warmstart_metrics") or {}).get("early_gain_db_mean", float("nan"))),
                            "use_canonical_prior": bool(summary.get("use_canonical_prior", False)),
                            "acoustic_zero_channel_regex": str(summary.get("acoustic_zero_channel_regex", "")),
                            "checkpoint": str(ckpt_path),
                            "train_dir": str(train_dir),
                            "eval_dir": str(seed_root / f"eval_{d}"),
                            "train_elapsed_s": 0.0,
                            "eval_elapsed_s": 0.0,
                            "fail_reasons": " | ".join([str(x) for x in (summary.get("fail_reasons") or [])]),
                        }
                    )
                continue

            train_cmd = [
                str(args.python_exe),
                str(TRAIN_SCRIPT),
                "--h5-path",
                str(h5_path),
                "--output-dir",
                str(train_dir),
                "--curriculum-levels",
                str(args.curriculum_levels),
                "--epochs-per-level",
                str(args.epochs_per_level),
                "--batch-size",
                str(int(args.batch_size)),
                "--val-frac",
                str(float(args.val_frac)),
                "--test-frac",
                str(float(args.test_frac)),
                "--seed",
                str(int(seed)),
                "--lr",
                str(float(args.lr)),
                "--weight-decay",
                str(float(args.weight_decay)),
                "--grad-clip-norm",
                str(float(args.grad_clip_norm)),
                "--lambda-reg",
                str(float(args.lambda_reg)),
                "--nr-margin-weight",
                str(float(args.nr_margin_weight)),
                "--nr-target-ratio",
                str(float(args.nr_target_ratio)),
                "--nr-margin-warmup-epochs",
                str(int(args.nr_margin_warmup_epochs)),
                "--nr-margin-mode",
                str(args.nr_margin_mode),
                "--nr-margin-focus-ratio",
                str(float(args.nr_margin_focus_ratio)),
                "--wopt-supervision-weight",
                str(float(args.wopt_supervision_weight)),
                "--supervision-source",
                str(args.supervision_source),
                "--acoustic-loss-weight",
                str(float(args.acoustic_loss_weight)),
                "--loss-domain",
                str(args.loss_domain),
                "--device",
                str(args.device),
                "--embed-dim",
                str(int(args.embed_dim)),
                "--num-heads",
                str(int(args.num_heads)),
                "--basis-dim",
                str(int(args.basis_dim)),
                "--fusion-mode",
                str(args.fusion_mode),
                "--feature-encoding",
                str(args.feature_encoding),
                "--ablation-tag",
                str(variant),
                "--canonical-prior-scale",
                str(float(args.canonical_prior_scale)),
                "--canonical-residual-l2-weight",
                str(float(args.canonical_residual_l2_weight)),
                "--min-level1-samples",
                str(int(args.min_level1_samples)),
                "--min-level23-samples",
                str(int(args.min_level23_samples)),
                "--min-qc-nr-last-p10-db",
                str(float(args.min_qc_nr_last_p10_db)),
                "--min-qc-nr-gain-p10-db",
                str(float(args.min_qc_nr_gain_p10_db)),
            ]
            if bool(args.residual_head_zero_init):
                train_cmd.append("--residual-head-zero-init")
            if bool(args.disable_feature_b):
                train_cmd.append("--disable-feature-b")
            if bool(args.use_path_features):
                train_cmd.append("--use-path-features")
            if bool(vf.get("use_canonical_prior", False)):
                train_cmd.append("--use-canonical-prior")
            regex = str(vf.get("acoustic_zero_channel_regex", "")).strip()
            if regex:
                train_cmd.extend(["--acoustic-zero-channel-regex", regex])
            if int(args.max_train_samples) > 0:
                train_cmd.extend(["--max-train-samples", str(int(args.max_train_samples))])

            if bool(args.dry_run):
                train_code, train_out, train_err, train_elapsed = 0, "", "", 0.0
            else:
                train_code, train_out, train_err, train_elapsed = run_subprocess(train_cmd, cwd=ROOT)
            (seed_root / "train_stdout.log").write_text(train_out, encoding="utf-8", errors="ignore")
            (seed_root / "train_stderr.log").write_text(train_err, encoding="utf-8", errors="ignore")
            (seed_root / "train_command.txt").write_text(" ".join(train_cmd), encoding="utf-8")

            if train_code != 0 or (not ckpt_path.exists() and not bool(args.dry_run)):
                rec = {
                    "variant": variant,
                    "seed": int(seed),
                    "domain": "all",
                    "train_status": "train_failed",
                    "train_exit_code": int(train_code),
                    "eval_exit_code": -1,
                    "gate_status": "missing",
                    "mean_nr_db": float("nan"),
                    "warmstart_early_gain_db_mean": float("nan"),
                    "use_canonical_prior": bool(vf.get("use_canonical_prior", False)),
                    "acoustic_zero_channel_regex": regex,
                    "checkpoint": str(ckpt_path),
                    "train_dir": str(train_dir),
                    "eval_dir": "",
                    "train_elapsed_s": float(train_elapsed),
                    "eval_elapsed_s": 0.0,
                    "fail_reasons": "train_failed",
                }
                rows.append(rec)
                errors.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "phase": "train",
                        "return_code": int(train_code),
                        "stdout_tail": train_out[-1500:],
                        "stderr_tail": train_err[-1500:],
                    }
                )
                if not bool(args.continue_on_error):
                    write_json(results_root / "errors.json", {"errors": errors})
                    raise RuntimeError(f"Training failed: variant={variant} seed={seed}")
                continue

            for d, eval_h5 in domain_h5.items():
                eval_dir = seed_root / f"eval_{d}"
                eval_dir.mkdir(parents=True, exist_ok=True)
                eval_cmd = [
                    str(args.python_exe),
                    str(EVAL_SCRIPT),
                    "--checkpoint-path",
                    str(ckpt_path),
                    "--h5-path",
                    str(eval_h5),
                    "--output-dir",
                    str(eval_dir),
                    "--batch-size",
                    str(int(args.batch_size)),
                    "--device",
                    str(args.device),
                    "--warmstart-cases",
                    str(int(args.warmstart_cases)),
                    "--warmstart-level",
                    str(int(args.warmstart_level)),
                    "--early-window-s",
                    str(float(args.early_window_s)),
                    "--target-metric",
                    str(args.target_metric),
                    "--half-target-ratio",
                    str(float(args.half_target_ratio)),
                    "--min-improvement-db",
                    str(float(args.min_improvement_db)),
                    "--eval-split",
                    str(args.eval_split),
                    "--ablation-tag",
                    str(variant),
                ]
                if bool(args.disable_improvement_gate):
                    eval_cmd.append("--disable-improvement-gate")
                if bool(args.disable_half_target_gate):
                    eval_cmd.append("--disable-half-target-gate")
                if regex:
                    eval_cmd.extend(["--acoustic-zero-channel-regex", regex])

                if bool(args.dry_run):
                    eval_code, eval_out, eval_err, eval_elapsed = 0, "", "", 0.0
                else:
                    eval_code, eval_out, eval_err, eval_elapsed = run_subprocess(eval_cmd, cwd=ROOT)

                (eval_dir / "eval_stdout.log").write_text(eval_out, encoding="utf-8", errors="ignore")
                (eval_dir / "eval_stderr.log").write_text(eval_err, encoding="utf-8", errors="ignore")
                (eval_dir / "eval_command.txt").write_text(" ".join(eval_cmd), encoding="utf-8")

                summary = read_eval_summary(eval_dir / "summary.json")
                rows.append(
                    {
                        "variant": variant,
                        "seed": int(seed),
                        "domain": d,
                        "train_status": "done",
                        "train_exit_code": int(train_code),
                        "eval_exit_code": int(eval_code),
                        "gate_status": str(summary.get("gate_status", "missing")),
                        "mean_nr_db": safe_float(summary.get("mean_nr_db", float("nan"))),
                        "warmstart_early_gain_db_mean": safe_float((summary.get("warmstart_metrics") or {}).get("early_gain_db_mean", float("nan"))),
                        "use_canonical_prior": bool(summary.get("use_canonical_prior", vf.get("use_canonical_prior", False))),
                        "acoustic_zero_channel_regex": str(summary.get("acoustic_zero_channel_regex", regex)),
                        "checkpoint": str(ckpt_path),
                        "train_dir": str(train_dir),
                        "eval_dir": str(eval_dir),
                        "train_elapsed_s": float(train_elapsed),
                        "eval_elapsed_s": float(eval_elapsed),
                        "fail_reasons": " | ".join([str(x) for x in (summary.get("fail_reasons") or [])]),
                    }
                )

                if eval_code != 0:
                    errors.append(
                        {
                            "variant": variant,
                            "seed": int(seed),
                            "domain": d,
                            "phase": "eval",
                            "return_code": int(eval_code),
                            "stdout_tail": eval_out[-1500:],
                            "stderr_tail": eval_err[-1500:],
                        }
                    )
                    if not bool(args.continue_on_error):
                        write_json(results_root / "errors.json", {"errors": errors})
                        raise RuntimeError(f"Eval failed: variant={variant} seed={seed} domain={d}")

    ranked = aggregate_rows(rows, eval_domains)

    run_csv = results_root / "run_rows.csv"
    rank_csv = results_root / "variant_ranked.csv"
    write_csv(run_csv, rows)
    write_csv(rank_csv, ranked)
    write_json(results_root / "errors.json", {"errors": errors})

    summary = {
        "results_root": str(results_root),
        "num_rows": int(len(rows)),
        "num_variants": int(len(variants)),
        "num_seeds": int(len(seeds)),
        "num_eval_domains": int(len(eval_domains)),
        "num_errors": int(len(errors)),
        "files": {
            "manifest": str(results_root / "run_manifest.json"),
            "run_rows_csv": str(run_csv),
            "variant_ranked_csv": str(rank_csv),
            "errors": str(results_root / "errors.json"),
        },
        "top_variant": ranked[0] if ranked else None,
    }
    write_json(results_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
