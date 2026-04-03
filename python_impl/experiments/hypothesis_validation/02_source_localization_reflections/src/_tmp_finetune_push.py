from pathlib import Path
import json
import sys

repo = Path("z:/anc/distributed_anc_sim_py")
stage02_dir = repo / "python_impl" / "experiments" / "hypothesis_validation" / "02_source_localization_reflections"
src_dir = stage02_dir / "src"
for p in (repo, repo / "python_impl", src_dir):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from gcc_reflection_notebook_utils import build_gcc_reflection_bundle, train_gcc_to_tdoa_model

h5_path = stage02_dir / "data" / "source_localization_single_reflection_l1_stable_v3_w2.h5"
init_ckpt = stage02_dir / "results" / "_target_push_geom_val_1e4" / "long_nocurr_default_120" / "best_model.pt"
out_dir = stage02_dir / "results" / "_target_push_geom_val_1e4" / "finetune_from_long_nocurr_lr2e4"
bundle = build_gcc_reflection_bundle(h5_path)

summary = train_gcc_to_tdoa_model(
    bundle=bundle,
    result_dir=out_dir,
    lr=2.0e-4,
    batch_size=256,
    epochs=140,
    seed=7,
    device="cuda",
    live_plot=False,
    huber_delta_norm=0.05,
    bound_penalty_weight=0.05,
    scheduler_patience=6,
    scheduler_factor=0.5,
    scheduler_min_lr=1.0e-6,
    early_stop_patience=30,
    early_stop_min_delta=0.0,
    curriculum_mode="none",
    init_checkpoint_path=init_ckpt,
)

report = {
    "best_epoch": summary["best_epoch"],
    "epochs_ran": summary["epochs_ran"],
    "iid_val_tdoa_mae_s": summary["iid_val"]["tdoa_mae_s"],
    "geom_val_tdoa_mae_s": summary["geom_val"]["tdoa_mae_s"],
    "iid_test_tdoa_mae_s": summary["iid_test"]["tdoa_mae_s"],
    "geom_test_tdoa_mae_s": summary["geom_test"]["tdoa_mae_s"],
    "pass_geom_val_1e4": bool(float(summary["geom_val"]["tdoa_mae_s"]) < 1.0e-4),
}
print(json.dumps(report, ensure_ascii=False))
