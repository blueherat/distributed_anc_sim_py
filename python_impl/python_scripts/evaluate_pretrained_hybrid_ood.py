from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT = ROOT / "python_impl" / "python_scripts" / "evaluate_hybrid_deep_fxlms_single_control.py"
DATASET_SCRIPT = ROOT / "python_impl" / "python_scripts" / "cfxlms_single_control_dataset_impl.py"
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


@dataclass
class CheckpointMeta:
    requires_sample_alignment: bool
    required_num_samples: int | None
    use_index_direct: bool
    use_index_embedding: bool
    use_canonical_prior: bool
    inspect_error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch OOD evaluation for already-trained Hybrid Deep-FxLMS checkpoints "
            "(no training), with optional OOD dataset auto-build and summary ranking."
        )
    )
    parser.add_argument("--checkpoints-root", type=str, required=True)
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        default=["**/final_hybrid_deep_fxlms.pt"],
        help=(
            "Glob (relative to --checkpoints-root) for checkpoint discovery. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--checkpoint-list",
        type=str,
        default=None,
        help="Optional text file listing checkpoint paths (absolute or relative to --checkpoints-root).",
    )
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 means all discovered checkpoints.")
    parser.add_argument(
        "--domains",
        type=str,
        default="in_domain,ood_mild,ood_hard",
        help="Comma-separated subset of {in_domain,ood_mild,ood_hard}.",
    )

    parser.add_argument("--in-domain-h5", type=str, default=str(DEFAULT_IN_DOMAIN_H5))
    parser.add_argument("--ood-mild-h5", type=str, default=None)
    parser.add_argument("--ood-hard-h5", type=str, default=None)
    parser.add_argument("--ood-h5-dir", type=str, default=str(DEFAULT_OOD_H5_DIR))
    parser.add_argument("--build-missing-ood-h5", action="store_true")
    parser.add_argument("--ood-num-rooms", type=int, default=500)
    parser.add_argument("--ood-max-attempts", type=int, default=30000)
    parser.add_argument("--dataset-seed", type=int, default=20260405)
    parser.add_argument("--dataset-min-nr-last-db", type=float, default=15.0)
    parser.add_argument("--dataset-min-nr-gain-db", type=float, default=15.0)
    parser.add_argument(
        "--dataset-min-both-pass-rate",
        type=float,
        default=0.95,
        help="If QC summary exists, require both_pass_rate >= this threshold.",
    )
    parser.add_argument(
        "--enforce-dataset-qc-gate",
        action="store_true",
        help="Fail when QC summary is missing or below --dataset-min-both-pass-rate.",
    )

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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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
        out[d] = p.resolve()
    return out


def run_subprocess(cmd: list[str], cwd: Path) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    return int(proc.returncode), str(proc.stdout), str(proc.stderr)


def dataset_num_samples(h5_path: Path) -> int:
    with h5py.File(str(h5_path), "r") as h5:
        if "processed/gcc" in h5:
            return int(h5["processed/gcc"].shape[0])
        if "raw/room_params/room_size" in h5:
            return int(h5["raw/room_params/room_size"].shape[0])
    raise KeyError(f"Cannot determine sample count from h5: {h5_path}")


def inspect_checkpoint_meta(checkpoint_path: Path) -> CheckpointMeta:
    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except Exception as exc:
        return CheckpointMeta(
            requires_sample_alignment=False,
            required_num_samples=None,
            use_index_direct=False,
            use_index_embedding=False,
            use_canonical_prior=False,
            inspect_error=f"checkpoint_load_failed:{exc}",
        )

    args = ckpt.get("args") if isinstance(ckpt, dict) else {}
    use_index_direct = bool((args or {}).get("index_direct_lookup", False))
    use_index_embedding = bool((args or {}).get("use_index_embedding", False))
    use_canonical_prior = bool((args or {}).get("use_canonical_prior", False))

    required_candidates: list[int] = []

    split_pack = ckpt.get("split_indices", None) if isinstance(ckpt, dict) else None
    if isinstance(split_pack, dict):
        split_total = 0
        for key in ("train", "val", "test"):
            arr = split_pack.get(key)
            if arr is None:
                continue
            if isinstance(arr, torch.Tensor):
                arr = arr.detach().cpu().numpy()
            split_total += int(np.asarray(arr).reshape(-1).size)
        if split_total > 0:
            required_candidates.append(int(split_total))

    state = ckpt.get("model_state_dict", {}) if isinstance(ckpt, dict) else {}
    if isinstance(state, dict):
        for key, tensor in state.items():
            if not hasattr(tensor, "shape") or len(tensor.shape) < 1:
                continue
            if key.endswith("index_direct_head.weight"):
                required_candidates.append(int(tensor.shape[0]))
            elif key.endswith("index_embedding.weight"):
                required_candidates.append(int(tensor.shape[0]))
            elif key.endswith("canonical_prior_lookup"):
                required_candidates.append(int(tensor.shape[0]))

    requires_alignment = bool(use_index_direct or use_index_embedding or use_canonical_prior)
    required_num_samples: int | None = None
    if required_candidates:
        # Prefer the largest candidate to avoid under-estimation if multiple buffers exist.
        required_num_samples = int(max(required_candidates))

    return CheckpointMeta(
        requires_sample_alignment=requires_alignment,
        required_num_samples=required_num_samples,
        use_index_direct=use_index_direct,
        use_index_embedding=use_index_embedding,
        use_canonical_prior=use_canonical_prior,
        inspect_error=None,
    )


def _domain_seed(base_seed: int, domain: str) -> int:
    if domain == "ood_mild":
        return int(base_seed) + 101
    if domain == "ood_hard":
        return int(base_seed) + 202
    return int(base_seed)


def ensure_dataset_for_domain(
    domain: str,
    h5_path: Path,
    args: argparse.Namespace,
) -> None:
    if h5_path.exists():
        return
    if domain == "in_domain":
        raise FileNotFoundError(f"In-domain dataset does not exist: {h5_path}")
    if not bool(args.build_missing_ood_h5):
        raise FileNotFoundError(
            f"Missing {domain} dataset and auto-build disabled: {h5_path}. "
            "Use --build-missing-ood-h5 to generate it."
        )

    h5_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(args.python_exec),
        str(DATASET_SCRIPT),
        "--domain-profile",
        str(domain),
        "--num-rooms",
        str(int(args.ood_num_rooms)),
        "--max-attempts",
        str(int(args.ood_max_attempts)),
        "--output-h5",
        str(h5_path),
        "--seed",
        str(_domain_seed(int(args.dataset_seed), domain)),
        "--min-nr-last-db",
        str(float(args.dataset_min_nr_last_db)),
        "--min-nr-gain-db",
        str(float(args.dataset_min_nr_gain_db)),
        "--no-preview-layouts",
    ]
    code, out, err = run_subprocess(cmd, cwd=ROOT)
    if code != 0:
        msg = (
            f"Dataset build failed for {domain} (exit={code}).\n"
            f"command={' '.join(cmd)}\n"
            f"stdout_tail={out[-1500:]}\n"
            f"stderr_tail={err[-1500:]}"
        )
        raise RuntimeError(msg)


def load_dataset_qc_summary(h5_path: Path) -> dict[str, Any] | None:
    qc_path = h5_path.with_suffix(".qc_summary.json")
    if not qc_path.exists():
        return None
    try:
        return read_json(qc_path)
    except Exception:
        return None


def validate_dataset_qc(
    domain: str,
    h5_path: Path,
    summary: dict[str, Any] | None,
    min_both_pass_rate: float,
    enforce: bool,
) -> tuple[float, str]:
    if summary is None:
        msg = f"dataset_qc_summary_missing:{h5_path}"
        if enforce:
            raise RuntimeError(msg)
        return float("nan"), msg

    both = safe_float((summary.get("pass_rates") or {}).get("both_pass_rate", float("nan")))
    if not np.isfinite(both):
        msg = f"dataset_qc_both_pass_non_finite:{domain}"
        if enforce:
            raise RuntimeError(msg)
        return float("nan"), msg

    if both < float(min_both_pass_rate):
        msg = (
            f"dataset_qc_both_pass_below_threshold:{domain}:"
            f"{both:.4f}<{float(min_both_pass_rate):.4f}"
        )
        if enforce:
            raise RuntimeError(msg)
        return both, msg

    return both, "ok"


def checkpoint_rel(checkpoint_path: Path, checkpoints_root: Path) -> str:
    try:
        return str(checkpoint_path.resolve().relative_to(checkpoints_root.resolve())).replace("\\", "/")
    except Exception:
        return checkpoint_path.name


def checkpoint_output_dir(
    output_root: Path,
    checkpoint_path: Path,
    checkpoints_root: Path,
    domain: str,
) -> Path:
    try:
        rel = checkpoint_path.resolve().relative_to(checkpoints_root.resolve())
        rel_no_suffix = rel.with_suffix("")
        return output_root / "runs" / rel_no_suffix / domain
    except Exception:
        return output_root / "runs" / checkpoint_path.stem / domain


def _summary_to_row(
    checkpoint_path: Path,
    checkpoint_rel_path: str,
    domain: str,
    h5_path: Path,
    out_dir: Path,
    eval_code: int,
    elapsed_s: float,
    dataset_qc_both_pass_rate: float,
    dataset_qc_status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_rel": checkpoint_rel_path,
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
        "dataset_qc_both_pass_rate": dataset_qc_both_pass_rate,
        "dataset_qc_status": dataset_qc_status,
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
        }
    )
    return row


def _skipped_row(
    checkpoint_path: Path,
    checkpoint_rel_path: str,
    domain: str,
    h5_path: Path,
    out_dir: Path,
    reason: str,
    dataset_qc_both_pass_rate: float,
    dataset_qc_status: str,
) -> dict[str, Any]:
    row = _summary_to_row(
        checkpoint_path=checkpoint_path,
        checkpoint_rel_path=checkpoint_rel_path,
        domain=domain,
        h5_path=h5_path,
        out_dir=out_dir,
        eval_code=-2,
        elapsed_s=0.0,
        dataset_qc_both_pass_rate=dataset_qc_both_pass_rate,
        dataset_qc_status=dataset_qc_status,
    )
    row["gate_status"] = "skipped"
    row["fail_reasons"] = str(reason)
    return row


def aggregate_checkpoint_rows(rows: list[dict[str, Any]], domains: list[str]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    checkpoint_abs: dict[str, str] = {}
    for r in rows:
        ck = str(r.get("checkpoint_rel", ""))
        dm = str(r.get("domain", ""))
        grouped[ck][dm] = r
        checkpoint_abs[ck] = str(r.get("checkpoint_path", ""))

    out: list[dict[str, Any]] = []
    for ck, dmap in grouped.items():
        in_row = dmap.get("in_domain", {})
        in_nr = safe_float(in_row.get("mean_nr_db", float("nan")))

        ood_vals: list[float] = []
        pass_all = True
        fail_domains: list[str] = []

        rec: dict[str, Any] = {
            "checkpoint_rel": ck,
            "checkpoint_path": checkpoint_abs.get(ck, ""),
            "in_domain_mean_nr_db": in_nr,
            "num_domains_expected": int(len(domains)),
            "num_domains_found": int(len(dmap)),
        }

        for d in domains:
            rr = dmap.get(d)
            if rr is None:
                rec[f"{d}_gate_status"] = "missing"
                rec[f"{d}_mean_nr_db"] = float("nan")
                rec[f"{d}_summary_exists"] = False
                pass_all = False
                fail_domains.append(d)
                continue

            d_nr = safe_float(rr.get("mean_nr_db", float("nan")))
            rec[f"{d}_gate_status"] = str(rr.get("gate_status", "missing"))
            rec[f"{d}_mean_nr_db"] = d_nr
            rec[f"{d}_summary_exists"] = bool(rr.get("summary_exists", False))
            rec[f"{d}_sample_6db_pass_rate"] = safe_float(rr.get("sample_6db_pass_rate", float("nan")))

            gate_ok = str(rr.get("gate_status", "")).lower() == "passed"
            if not gate_ok:
                pass_all = False
                fail_domains.append(d)

            if d != "in_domain" and np.isfinite(d_nr):
                ood_vals.append(d_nr)

            if d != "in_domain":
                rec[f"degradation_{d}_db"] = (
                    in_nr - d_nr if np.isfinite(in_nr) and np.isfinite(d_nr) else float("nan")
                )

        rec["ood_mean_nr_db"] = float(np.mean(ood_vals)) if ood_vals else float("nan")
        rec["ood_min_nr_db"] = float(np.min(ood_vals)) if ood_vals else float("nan")
        rec["pass_all_domains"] = bool(pass_all and len(dmap) == len(domains))
        rec["fail_domains"] = "|".join(sorted(set(fail_domains)))
        out.append(rec)

    out.sort(
        key=lambda r: (
            bool(r.get("pass_all_domains", False)),
            safe_float(r.get("ood_mean_nr_db", float("nan")), default=float("-inf")),
            safe_float(r.get("ood_min_nr_db", float("nan")), default=float("-inf")),
            safe_float(r.get("in_domain_mean_nr_db", float("nan")), default=float("-inf")),
        ),
        reverse=True,
    )

    for idx, r in enumerate(out, start=1):
        r["rank"] = int(idx)
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

    for d, h5 in domain_h5.items():
        ensure_dataset_for_domain(d, h5, args)

    dataset_qc: dict[str, dict[str, Any]] = {}
    dataset_sizes: dict[str, int] = {}
    for d, h5 in domain_h5.items():
        qc_summary = load_dataset_qc_summary(h5)
        both_pass, qc_status = validate_dataset_qc(
            domain=d,
            h5_path=h5,
            summary=qc_summary,
            min_both_pass_rate=float(args.dataset_min_both_pass_rate),
            enforce=bool(args.enforce_dataset_qc_gate),
        )
        dataset_qc[d] = {
            "h5_path": str(h5),
            "qc_summary": qc_summary,
            "both_pass_rate": both_pass,
            "qc_status": qc_status,
        }
        dataset_sizes[d] = int(dataset_num_samples(h5))

    checkpoints = discover_checkpoints(args, checkpoints_root)
    if not checkpoints:
        raise RuntimeError(
            "No checkpoints discovered. "
            "Check --checkpoints-root / --checkpoint-glob / --checkpoint-list settings."
        )

    manifest = {
        "timestamp": int(time.time()),
        "checkpoints_root": str(checkpoints_root),
        "num_checkpoints": int(len(checkpoints)),
        "checkpoint_globs": [str(x) for x in args.checkpoint_glob],
        "domains": domains,
        "domain_h5": {k: str(v) for k, v in domain_h5.items()},
        "dataset_qc": dataset_qc,
        "eval_script": str(EVAL_SCRIPT),
        "dataset_script": str(DATASET_SCRIPT),
        "python_exec": str(args.python_exec),
        "args": vars(args),
    }
    write_json(output_root / "run_manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    ckpt_meta_cache: dict[str, CheckpointMeta] = {}

    for cp in checkpoints:
        cp_rel = checkpoint_rel(cp, checkpoints_root)
        meta = ckpt_meta_cache.get(str(cp))
        if meta is None:
            meta = inspect_checkpoint_meta(cp)
            ckpt_meta_cache[str(cp)] = meta

        for d, h5 in domain_h5.items():
            out_dir = checkpoint_output_dir(output_root, cp, checkpoints_root, d)
            out_dir.mkdir(parents=True, exist_ok=True)
            summary_path = out_dir / "summary.json"

            expected_n = meta.required_num_samples
            dataset_n = int(dataset_sizes.get(d, -1))
            if bool(meta.requires_sample_alignment) and expected_n is not None and dataset_n != int(expected_n):
                reason = (
                    f"sample_count_mismatch:checkpoint_requires={int(expected_n)}"
                    f",dataset_has={int(dataset_n)}"
                )
                rows.append(
                    _skipped_row(
                        checkpoint_path=cp,
                        checkpoint_rel_path=cp_rel,
                        domain=d,
                        h5_path=h5,
                        out_dir=out_dir,
                        reason=reason,
                        dataset_qc_both_pass_rate=safe_float(dataset_qc[d].get("both_pass_rate", float("nan"))),
                        dataset_qc_status=str(dataset_qc[d].get("qc_status", "unknown")),
                    )
                )
                errors.append(
                    {
                        "checkpoint_rel": cp_rel,
                        "domain": d,
                        "return_code": -2,
                        "output_dir": str(out_dir),
                        "stderr_tail": "",
                        "stdout_tail": reason,
                    }
                )
                continue

            if summary_path.exists() and not bool(args.overwrite_existing):
                rows.append(
                    _summary_to_row(
                        checkpoint_path=cp,
                        checkpoint_rel_path=cp_rel,
                        domain=d,
                        h5_path=h5,
                        out_dir=out_dir,
                        eval_code=0,
                        elapsed_s=0.0,
                        dataset_qc_both_pass_rate=safe_float(dataset_qc[d].get("both_pass_rate", float("nan"))),
                        dataset_qc_status=str(dataset_qc[d].get("qc_status", "unknown")),
                    )
                )
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
            ]
            if bool(args.disable_improvement_gate):
                cmd.append("--disable-improvement-gate")
            if bool(args.disable_half_target_gate):
                cmd.append("--disable-half-target-gate")

            if bool(args.dry_run):
                code, out, err = 0, "", ""
                elapsed = 0.0
            else:
                t0 = time.time()
                code, out, err = run_subprocess(cmd, cwd=ROOT)
                elapsed = float(time.time() - t0)

            (out_dir / "eval_stdout.log").write_text(out, encoding="utf-8", errors="ignore")
            (out_dir / "eval_stderr.log").write_text(err, encoding="utf-8", errors="ignore")
            (out_dir / "eval_command.txt").write_text(" ".join(cmd), encoding="utf-8")

            row = _summary_to_row(
                checkpoint_path=cp,
                checkpoint_rel_path=cp_rel,
                domain=d,
                h5_path=h5,
                out_dir=out_dir,
                eval_code=code,
                elapsed_s=elapsed,
                dataset_qc_both_pass_rate=safe_float(dataset_qc[d].get("both_pass_rate", float("nan"))),
                dataset_qc_status=str(dataset_qc[d].get("qc_status", "unknown")),
            )
            rows.append(row)

            if code != 0:
                err_rec = {
                    "checkpoint_rel": cp_rel,
                    "domain": d,
                    "return_code": int(code),
                    "output_dir": str(out_dir),
                    "stderr_tail": err[-1500:],
                    "stdout_tail": out[-1500:],
                }
                errors.append(err_rec)
                if not bool(args.continue_on_error):
                    write_json(output_root / "errors.json", {"errors": errors})
                    raise RuntimeError(
                        f"Evaluation failed and --continue-on-error is false: checkpoint={cp_rel}, domain={d}"
                    )

    domain_csv = output_root / "checkpoint_domain_metrics.csv"
    grouped_rows = aggregate_checkpoint_rows(rows, domains)
    grouped_csv = output_root / "checkpoint_robustness_ranked.csv"

    write_csv(domain_csv, rows)
    write_csv(grouped_csv, grouped_rows)
    write_json(output_root / "errors.json", {"errors": errors})

    summary = {
        "output_root": str(output_root),
        "num_checkpoints": int(len(checkpoints)),
        "num_domain_rows": int(len(rows)),
        "num_checkpoint_rows": int(len(grouped_rows)),
        "num_errors": int(len(errors)),
        "files": {
            "manifest": str(output_root / "run_manifest.json"),
            "domain_csv": str(domain_csv),
            "checkpoint_csv": str(grouped_csv),
            "errors": str(output_root / "errors.json"),
        },
        "top_checkpoint": grouped_rows[0] if grouped_rows else None,
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
