$ErrorActionPreference = 'Stop'

$python = 'D:/Anaconda/Anaconda/envs/myenv/python.exe'
$script = 'z:/anc/distributed_anc_sim_py/python_impl/python_scripts/run_hybrid_ablation_suite.py'
$resultsRoot = 'z:/anc/distributed_anc_sim_py/python_impl/experiments/anc_single_control/hybrid_ablation_standard_20260403'

& $python $script `
  '--python-exe' $python `
  '--h5-path' 'z:/anc/distributed_anc_sim_py/python_impl/python_scripts/cfxlms_qc_dataset_single_control.h5' `
  '--results-root' $resultsRoot `
  '--seeds' '7,42,123,999,20260403' `
  '--device' 'cpu' `
  '--batch-size' '64' `
  '--curriculum-levels' '1,2,3' `
  '--epochs-per-level' '20,20,30' `
  '--embed-dim' '128' `
  '--num-heads' '4' `
  '--val-frac' '0.2' `
  '--lr' '1e-3' `
  '--weight-decay' '1e-4' `
  '--warmstart-cases' '8' `
  '--warmstart-level' '3' `
  '--early-window-s' '0.25' `
  '--half-target-ratio' '0.5' `
  '--min-level1-samples' '128' `
  '--min-level23-samples' '512' `
  '--min-qc-nr-last-p10-db' '12' `
  '--min-qc-nr-gain-p10-db' '12' `
  '--skip-existing'

$aggScript = 'z:/anc/distributed_anc_sim_py/python_impl/python_scripts/aggregate_hybrid_ablation_results.py'
& $python $aggScript '--results-root' $resultsRoot
