$ErrorActionPreference = 'Stop'

$python = 'D:/Anaconda/Anaconda/envs/myenv/python.exe'
$script = 'z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/src/run_tdoa_multiseed_longpush.py'

$args = @(
  '--h5-path', 'z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/data/source_localization_single_reflection_l1_stable_v3_w2.h5',
  '--no-init-checkpoint',
  '--results-root', 'z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/results/_target_push_geom_val_1e4/capacity_w50_pairref_multiseed4_quick180',
  '--seeds', '7,42,123,999',
  '--device', 'cuda',
  '--lr', '7e-4',
  '--batch-size', '32',
  '--epochs', '180',
  '--huber-delta-norm', '0.02',
  '--bound-penalty-weight', '0.01',
  '--aux-feature-mode', 'pair_plus_ref_position_norm',
  '--model-width-mult', '5.0',
  '--dropout-p', '0.1',
  '--scheduler-patience', '8',
  '--scheduler-factor', '0.5',
  '--scheduler-min-lr', '1e-7',
  '--early-stop-patience', '0',
  '--target-geom-val-mae-s', '1e-4'
)

& $python $script @args
