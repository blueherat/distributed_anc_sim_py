$ErrorActionPreference = 'Stop'

param(
    [string]$CheckpointsRoot = 'python_impl/experiments/anc_single_control/hybrid_ablation_standard_20260403',
    [string]$OutputRoot = 'python_impl/experiments/anc_single_control/pretrained_ood_eval_20260405',
    [string]$InDomainH5 = 'python_impl/python_scripts/cfxlms_qc_dataset_single_control.h5',
    [string]$OodMildH5 = 'python_impl/python_scripts/cfxlms_qc_dataset_single_control_ood_mild.h5',
    [string]$OodHardH5 = 'python_impl/python_scripts/cfxlms_qc_dataset_single_control_ood_hard.h5',
    [string]$Device = 'auto',
    [int]$MaxCheckpoints = 0,
    [switch]$BuildMissingOod,
    [switch]$DryRun
)

$py = 'D:/Anaconda/Anaconda/envs/myenv/python.exe'
$script = 'python_impl/python_scripts/evaluate_pretrained_hybrid_ood.py'

$args = @(
    $script,
    '--checkpoints-root', $CheckpointsRoot,
    '--output-root', $OutputRoot,
    '--in-domain-h5', $InDomainH5,
    '--ood-mild-h5', $OodMildH5,
    '--ood-hard-h5', $OodHardH5,
    '--device', $Device,
    '--max-checkpoints', "$MaxCheckpoints",
    '--continue-on-error'
)

if ($BuildMissingOod) {
    $args += '--build-missing-ood-h5'
}
if ($DryRun) {
    $args += '--dry-run'
}

& $py @args
