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
EVAL_SCRIPT = ROOT / "python_impl" / "python_scripts" / "evaluate_hybrid_deep_fxlms_single_control.py"
DEFAULT_IN_DOMAIN_H5 = ROOT / "python_impl" / "python_scripts" / "cfxlms_qc_dataset_single_control.h5"
DEFAULT_OOD_H5_DIR = ROOT / "python_impl" / "python_scripts"


def parse_csv_list(text: str) -> list[str]:
    return [v.strip() for v in str(text).split(",") if v.strip()]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual evaluation for pretrained Hybrid Deep-FxLMS checkpoints: "
            "baseline vs no-prior vs E2R/R2R masking vs sample_idx remap."
        )
    )
    parser.add_argument("--checkpoints-root", type=str, required=True)
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        default=["**/final_hybrid_deep_fxlms.pt"],
        help="Glob relative to --checkpoints-root for checkpoint discovery. Can be repeated.",
    )
    parser.add_argument("--checkpoint-list", type=str, default=None)
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 means all discovered checkpoints.")

    parser.add_argument("--domains", type=str, default="in_domain,ood_mild,ood_hard")
    parser.add_argument("--in-domain-h5", type=str, default=str(DEFAULT_IN_DOMAIN_H5))
    parser.add_argument("--ood-mild-h5", type=str, default=None)
    parser.add_argument("--ood-hard-h5", type=str, default=None)
    parser.add_argument("--ood-h5-dir", type=str, default=str(DEFAULT_OOD_H5_DIR))

    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,no_prior,mask_e2r_r2r,no_prior_mask_e2r_r2r,sample_idx_shuffle",
        help=(
            "Comma-separated subset of baseline,no_prior,mask_e2r_r2r,"
            "no_prior_mask_e2r_r2r,sample_idx_shuffle,sample_idx_reverse,sample_idx_offset"
        ),
    )
    parser.add_argument(
        "--mask-channel-regex",
        type=str,
        default="^(E2R_|R2R_)",
        help="Regex over logical acoustic channels for masking variants.",
    )
    parser.add_argument("--sample-idx-seed", type=int, default=20260405)
    parser.add_argument("--sample-idx-offset", type=int, default=1)

    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--python-exec", type=str, default=sys.executable)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmstart-cases", type=int, default=8)
    parser.add_argument("--warmstart-level", type=int, default=3)
    parser.add_argument("--early-window-s", type=float, default=0.25)
    parser.add_argument("--target-metric", choices=("nr_last_db",), default="nr_last_db")
    parser.add_argument("--half-target-ratio", type=float, default=0.5)
    parser.add_argument("--min-improvement-db", type=float, default=6.0)
    parser.add_argument("--eval-split", choices=("test", "val", "train", "all"), default="test")
    parser.add_argument("--disable-improvement-gate", action="store_true")
    parser.add_argument("--disable-half-target-gate", action="store_true")

    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--retry-lock-max-attempts",
        type=int,
        default=0,
        help="Retry count for HDF5 lock errors (0 disables retry).",
    )
    parser.add_argument(
        "--retry-lock-delay-s",
        type=float,
        default=5.0,
        help="Delay between lock-error retries in seconds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_h5_lock_error(stderr_text: str) -> bool:
    s = str(stderr_text).lower()
    return (
        "unable to lock file" in s
        or "win32 getlasterror() = 33" in s
        or ("h5py.h5f.open" in s and "unable to synchronously open file" in s)
    )


def _load_checkpoint_list_file(path: Path, checkpoints_root: Path) -> list[Path]:
    out: list[Path] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        p = Path(s)
        if not p.is_absolute():
            p = (checkpoints_root / p).resolve()
        out.append(p)
    return out


def discover_checkpoints(args: argparse.Namespace, checkpoints_root: Path) -> list[Path]:
    found: set[Path] = set()
    for pattern in args.checkpoint_glob:
        for p in checkpoints_root.glob(str(pattern)):
            if p.is_file():
                found.add(p.resolve())

    if args.checkpoint_list is not None:
        list_path = resolve_path(args.checkpoint_list)
        if list_path is None or not list_path.exists():
            raise FileNotFoundError(f"checkpoint list not found: {args.checkpoint_list}")
        for p in _load_checkpoint_list_file(list_path, checkpoints_root):
            if p.is_file():
                found.add(p.resolve())

    out = sorted(found, key=lambda p: str(p).lower())
    if int(args.max_checkpoints) > 0:
        out = out[: int(args.max_checkpoints)]
    return out


def resolve_domain_h5_paths(args: argparse.Namespace) -> dict[str, Path]:
    domains = parse_csv_list(args.domains)
    allowed = {"in_domain", "ood_mild", "ood_hard"}
    unknown = [d for d in domains if d not in allowed]
    if unknown:
        raise ValueError(f"Unsupported domain(s): {unknown}; allowed={sorted(allowed)}")

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


def run_subprocess(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    return int(proc.returncode), str(proc.stdout), str(proc.stderr)


def checkpoint_rel(checkpoint_path: Path, checkpoints_root: Path) -> str:
    try:
        return str(checkpoint_path.resolve().relative_to(checkpoints_root.resolve())).replace("\\", "/")
    except Exception:
        return checkpoint_path.name


def run_output_dir(
    output_root: Path,
    checkpoint_path: Path,
    checkpoints_root: Path,
    variant: str,
    domain: str,
) -> Path:
    try:
        rel = checkpoint_path.resolve().relative_to(checkpoints_root.resolve())
        rel_no_suffix = rel.with_suffix("")
        return output_root / "runs" / variant / rel_no_suffix / domain
    except Exception:
        return output_root / "runs" / variant / checkpoint_path.stem / domain


def variant_extra_eval_args(variant: str, args: argparse.Namespace) -> list[str]:
    v = str(variant).strip().lower()
    if v == "baseline":
        return []
    if v == "no_prior":
        return ["--disable-canonical-prior-eval"]
    if v == "mask_e2r_r2r":
        return ["--acoustic-zero-channel-regex", str(args.mask_channel_regex)]
    if v == "no_prior_mask_e2r_r2r":
        return ["--disable-canonical-prior-eval", "--acoustic-zero-channel-regex", str(args.mask_channel_regex)]
    if v == "sample_idx_shuffle":
        return [
            "--sample-idx-remap",
            "shuffle",
            "--sample-idx-seed",
            str(int(args.sample_idx_seed)),
        ]
    if v == "sample_idx_reverse":
        return ["--sample-idx-remap", "reverse"]
    if v == "sample_idx_offset":
        return ["--sample-idx-remap", "offset", "--sample-idx-offset", str(int(args.sample_idx_offset))]
    raise ValueError(f"Unsupported variant: {variant}")


def summary_to_row(
    checkpoint_path: Path,
    checkpoint_rel_path: str,
    variant: str,
    domain: str,
    h5_path: Path,
    out_dir: Path,
    eval_code: int,
    elapsed_s: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_rel": checkpoint_rel_path,
        "variant": str(variant),
        "domain": str(domain),
        "h5_path": str(h5_path),
        "output_dir": str(out_dir),
        "eval_return_code": int(eval_code),
        "elapsed_s": float(elapsed_s),
        "summary_exists": False,
        "gate_status": "missing",
        "mean_nr_db": float("nan"),
        "level_1_nr_db": float("nan"),
        "level_2_nr_db": float("nan"),
        "level_3_nr_db": float("nan"),
        "warmstart_early_gain_db_mean": float("nan"),
        "warmstart_ratio_mean": float("nan"),
        "improvement_pass": False,
        "half_target_pass": False,
        "sample_6db_pass_rate": float("nan"),
        "fail_reasons": "",
        "train_use_canonical_prior": False,
        "eval_use_canonical_prior": False,
    }

    summary_path = out_dir / "summary.json"
    if not summary_path.exists():
        return row

    try:
        summary = read_json(summary_path)
    except Exception:
        return row

    level_results = summary.get("level_results") or {}
    warm = summary.get("warmstart_metrics") or {}
    improvement_gate = summary.get("improvement_gate") or {}
    half_target_gate = summary.get("half_target_gate") or {}

    row.update(
        {
            "summary_exists": True,
            "gate_status": str(summary.get("gate_status", "unknown")),
            "mean_nr_db": safe_float(summary.get("mean_nr_db", float("nan"))),
            "level_1_nr_db": safe_float((level_results.get("level_1") or {}).get("nr_db", float("nan"))),
            "level_2_nr_db": safe_float((level_results.get("level_2") or {}).get("nr_db", float("nan"))),
            "level_3_nr_db": safe_float((level_results.get("level_3") or {}).get("nr_db", float("nan"))),
            "warmstart_early_gain_db_mean": safe_float(warm.get("early_gain_db_mean", float("nan"))),
            "warmstart_ratio_mean": safe_float(warm.get("convergence_step_ratio_mean", float("nan"))),
            "improvement_pass": bool(improvement_gate.get("pass", False)),
            "half_target_pass": bool(half_target_gate.get("pass", False)),
            "sample_6db_pass_rate": safe_float(warm.get("sample_6db_pass_rate", float("nan"))),
            "fail_reasons": " | ".join([str(x) for x in (summary.get("fail_reasons") or [])]),
            "train_use_canonical_prior": bool(summary.get("use_canonical_prior_train", summary.get("use_canonical_prior", False))),
            "eval_use_canonical_prior": bool(summary.get("use_canonical_prior", False)),
        }
    )
    return row


def add_delta_vs_baseline(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_map: dict[tuple[str, str], float] = {}
    for r in rows:
        if str(r.get("variant", "")) == "baseline":
            key = (str(r.get("checkpoint_rel", "")), str(r.get("domain", "")))
            baseline_map[key] = safe_float(r.get("mean_nr_db", float("nan")))

    out: list[dict[str, Any]] = []
    for r in rows:
        rec = dict(r)
        key = (str(r.get("checkpoint_rel", "")), str(r.get("domain", "")))
        base = safe_float(baseline_map.get(key, float("nan")))
        cur = safe_float(r.get("mean_nr_db", float("nan")))
        rec["baseline_mean_nr_db"] = base
        rec["delta_vs_baseline_db"] = cur - base if np.isfinite(cur) and np.isfinite(base) else float("nan")
        out.append(rec)
    return out


def aggregate_checkpoint_variant_rows(rows: list[dict[str, Any]], domains: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    ckpt_abs: dict[str, str] = {}
    for r in rows:
        key = (str(r.get("checkpoint_rel", "")), str(r.get("variant", "")))
        grouped[key].append(r)
        ckpt_abs[str(r.get("checkpoint_rel", ""))] = str(r.get("checkpoint_path", ""))

    out: list[dict[str, Any]] = []
    for (ck_rel, variant), items in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        domain_map = {str(r.get("domain", "")): r for r in items}
        mean_vals: list[float] = []
        delta_vals: list[float] = []
        pass_all = True
        missing_domains: list[str] = []

        rec: dict[str, Any] = {
            "checkpoint_rel": ck_rel,
            "checkpoint_path": ckpt_abs.get(ck_rel, ""),
            "variant": variant,
            "num_domains_expected": int(len(domains)),
            "num_domains_found": int(len(domain_map)),
        }

        for d in domains:
            rr = domain_map.get(d)
            if rr is None:
                rec[f"{d}_gate_status"] = "missing"
                rec[f"{d}_mean_nr_db"] = float("nan")
                rec[f"{d}_delta_vs_baseline_db"] = float("nan")
                pass_all = False
                missing_domains.append(d)
                continue

            gate_status = str(rr.get("gate_status", "missing"))
            mean_nr = safe_float(rr.get("mean_nr_db", float("nan")))
            delta = safe_float(rr.get("delta_vs_baseline_db", float("nan")))
            rec[f"{d}_gate_status"] = gate_status
            rec[f"{d}_mean_nr_db"] = mean_nr
            rec[f"{d}_delta_vs_baseline_db"] = delta

            if gate_status.lower() != "passed":
                pass_all = False
            if np.isfinite(mean_nr):
                mean_vals.append(mean_nr)
            if np.isfinite(delta):
                delta_vals.append(delta)

        rec["mean_nr_db_across_domains"] = float(np.mean(mean_vals)) if mean_vals else float("nan")
        rec["mean_delta_vs_baseline_db"] = float(np.mean(delta_vals)) if delta_vals else float("nan")
        rec["pass_all_domains"] = bool(pass_all and len(domain_map) == len(domains))
        rec["missing_domains"] = "|".join(sorted(set(missing_domains)))
        out.append(rec)

    out.sort(
        key=lambda r: (
            bool(r.get("pass_all_domains", False)),
            safe_float(r.get("mean_delta_vs_baseline_db", float("nan")), default=float("-inf")),
            safe_float(r.get("mean_nr_db_across_domains", float("nan")), default=float("-inf")),
        ),
        reverse=True,
    )
    for i, r in enumerate(out, start=1):
        r["rank"] = int(i)
    return out


def main() -> int:
    args = parse_args()

    checkpoints_root = resolve_path(args.checkpoints_root)
    if checkpoints_root is None or not checkpoints_root.exists():
        raise FileNotFoundError(f"checkpoints root does not exist: {args.checkpoints_root}")

    output_root = resolve_path(args.output_root)
    if output_root is None:
        raise ValueError("Invalid --output-root")
    output_root.mkdir(parents=True, exist_ok=True)

    domain_h5 = resolve_domain_h5_paths(args)
    domains = list(domain_h5.keys())

    variants = parse_csv_list(args.variants)
    allowed_variants = {
        "baseline",
        "no_prior",
        "mask_e2r_r2r",
        "no_prior_mask_e2r_r2r",
        "sample_idx_shuffle",
        "sample_idx_reverse",
        "sample_idx_offset",
    }
    bad_variants = [v for v in variants if v not in allowed_variants]
    if bad_variants:
        raise ValueError(f"Unsupported variants: {bad_variants}; allowed={sorted(allowed_variants)}")
    if "baseline" not in variants:
        variants = ["baseline"] + variants

    checkpoints = discover_checkpoints(args, checkpoints_root)
    if not checkpoints:
        raise RuntimeError("No checkpoints discovered.")

    manifest = {
        "timestamp": int(time.time()),
        "checkpoints_root": str(checkpoints_root),
        "num_checkpoints": int(len(checkpoints)),
        "domains": domains,
        "domain_h5": {k: str(v) for k, v in domain_h5.items()},
        "variants": variants,
        "eval_script": str(EVAL_SCRIPT),
        "python_exec": str(args.python_exec),
        "args": vars(args),
    }
    write_json(output_root / "run_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for cp in checkpoints:
        cp_rel = checkpoint_rel(cp, checkpoints_root)
        for d, h5 in domain_h5.items():
            for variant in variants:
                out_dir = run_output_dir(output_root, cp, checkpoints_root, variant, d)
                out_dir.mkdir(parents=True, exist_ok=True)
                summary_path = out_dir / "summary.json"

                if summary_path.exists() and not bool(args.overwrite_existing):
                    row = summary_to_row(
                        checkpoint_path=cp,
                        checkpoint_rel_path=cp_rel,
                        variant=variant,
                        domain=d,
                        h5_path=h5,
                        out_dir=out_dir,
                        eval_code=0,
                        elapsed_s=0.0,
                    )
                    rows.append(row)
                    continue

                cmd = [
                    str(args.python_exec),
                    str(EVAL_SCRIPT),
                    "--checkpoint-path",
                    str(cp),
                    "--h5-path",
                    str(h5),
                    "--output-dir",
                    str(out_dir),
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
                    cmd.append("--disable-improvement-gate")
                if bool(args.disable_half_target_gate):
                    cmd.append("--disable-half-target-gate")
                cmd.extend(variant_extra_eval_args(variant=variant, args=args))

                if bool(args.dry_run):
                    code, out, err = 0, "", ""
                    elapsed = 0.0
                else:
                    max_retry = max(0, int(args.retry_lock_max_attempts))
                    retry_delay = max(0.0, float(args.retry_lock_delay_s))
                    attempt = 0
                    t0 = time.time()
                    while True:
                        code, out, err = run_subprocess(cmd, cwd=ROOT)
                        if code == 0:
                            break
                        if attempt >= max_retry:
                            break
                        if not is_h5_lock_error(err):
                            break
                        attempt += 1
                        print(
                            f"[retry-lock] checkpoint={cp_rel} domain={d} variant={variant} "
                            f"attempt={attempt}/{max_retry} delay_s={retry_delay:.1f}"
                        )
                        if retry_delay > 0.0:
                            time.sleep(retry_delay)
                    elapsed = float(time.time() - t0)

                (out_dir / "eval_stdout.log").write_text(out, encoding="utf-8", errors="ignore")
                (out_dir / "eval_stderr.log").write_text(err, encoding="utf-8", errors="ignore")
                (out_dir / "eval_command.txt").write_text(" ".join(cmd), encoding="utf-8")

                row = summary_to_row(
                    checkpoint_path=cp,
                    checkpoint_rel_path=cp_rel,
                    variant=variant,
                    domain=d,
                    h5_path=h5,
                    out_dir=out_dir,
                    eval_code=code,
                    elapsed_s=elapsed,
                )
                rows.append(row)

                if code != 0:
                    errors.append(
                        {
                            "checkpoint_rel": cp_rel,
                            "variant": variant,
                            "domain": d,
                            "return_code": int(code),
                            "output_dir": str(out_dir),
                            "stderr_tail": err[-1500:],
                            "stdout_tail": out[-1500:],
                        }
                    )
                    if not bool(args.continue_on_error):
                        write_json(output_root / "errors.json", {"errors": errors})
                        raise RuntimeError(
                            "Evaluation failed and --continue-on-error is false: "
                            f"checkpoint={cp_rel}, variant={variant}, domain={d}"
                        )

    rows_with_delta = add_delta_vs_baseline(rows)
    ranked_rows = aggregate_checkpoint_variant_rows(rows_with_delta, domains)

    domain_csv = output_root / "variant_domain_metrics.csv"
    delta_csv = output_root / "variant_delta_vs_baseline.csv"
    rank_csv = output_root / "variant_checkpoint_ranked.csv"

    write_csv(domain_csv, rows)
    write_csv(delta_csv, rows_with_delta)
    write_csv(rank_csv, ranked_rows)
    write_json(output_root / "errors.json", {"errors": errors})

    summary = {
        "output_root": str(output_root),
        "num_checkpoints": int(len(checkpoints)),
        "num_domains": int(len(domains)),
        "num_variants": int(len(variants)),
        "num_rows": int(len(rows)),
        "num_errors": int(len(errors)),
        "files": {
            "manifest": str(output_root / "run_manifest.json"),
            "domain_csv": str(domain_csv),
            "delta_csv": str(delta_csv),
            "rank_csv": str(rank_csv),
            "errors": str(output_root / "errors.json"),
        },
        "top_rank": ranked_rows[0] if ranked_rows else None,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
