$ErrorActionPreference = 'Stop'

$python = 'D:/Anaconda/Anaconda/envs/myenv/python.exe'
$script = 'z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/src/run_tdoa_multiseed_longpush.py'

$args = @(
  '--h5-path', 'z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/data/source_localization_single_reflection_l1_stable_v3_w2.h5',
  '--init-checkpoint', 'z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/results/_target_push_geom_val_1e4/capacity_w50_pairref_seed7_quick180/seed_7/best_model.pt',
  '--results-root', 'z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/results/_target_push_geom_val_1e4/capacity_w50_pairref_seed7_refine_tight320_rerun',
  '--seeds', '7',
  '--device', 'cuda',
  '--lr', '5e-6',
  '--batch-size', '32',
  '--epochs', '320',
  '--huber-delta-norm', '0.01',
  '--bound-penalty-weight', '0.0',
  '--aux-feature-mode', 'pair_plus_ref_position_norm',
  '--model-width-mult', '5.0',
  '--dropout-p', '0.1',
  '--scheduler-patience', '14',
  '--scheduler-factor', '0.5',
  '--scheduler-min-lr', '1e-10',
  '--early-stop-patience', '0',
  '--target-geom-val-mae-s', '1e-4'
)

& $python $script @args
