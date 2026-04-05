$ErrorActionPreference = 'Stop'

param(
    [string]$CheckpointsRoot = 'python_impl/python_scripts/_tmp_smoke_canon_l123_s3',
    [string]$OutputRoot = 'python_impl/experiments/anc_single_control/pretrained_counterfactual_eval_20260405',
    [string]$Domains = 'in_domain,ood_mild,ood_hard',
    [string]$Variants = 'baseline,no_prior,mask_e2r_r2r,no_prior_mask_e2r_r2r,sample_idx_shuffle',
    [string]$InDomainH5 = 'python_impl/python_scripts/cfxlms_qc_dataset_single_control.h5',
    [string]$OodMildH5 = 'python_impl/python_scripts/cfxlms_qc_dataset_single_control_ood_mild.h5',
    [string]$OodHardH5 = 'python_impl/python_scripts/cfxlms_qc_dataset_single_control_ood_hard.h5',
    [string]$Device = 'auto',
    [int]$MaxCheckpoints = 0,
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$py = 'D:/Anaconda/Anaconda/envs/myenv/python.exe'
$script = 'python_impl/python_scripts/evaluate_pretrained_hybrid_counterfactual.py'

$args = @(
    $script,
    '--checkpoints-root', $CheckpointsRoot,
    '--output-root', $OutputRoot,
    '--domains', $Domains,
    '--variants', $Variants,
    '--in-domain-h5', $InDomainH5,
    '--ood-mild-h5', $OodMildH5,
    '--ood-hard-h5', $OodHardH5,
    '--device', $Device,
    '--max-checkpoints', "$MaxCheckpoints"
)

if ($ContinueOnError) {
    $args += '--continue-on-error'
}
if ($DryRun) {
    $args += '--dry-run'
}

& $py @args
