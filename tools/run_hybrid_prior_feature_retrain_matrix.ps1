$ErrorActionPreference = 'Stop'

param(
    [string]$ResultsRoot = 'python_impl/experiments/anc_single_control/hybrid_prior_feature_retrain_matrix_20260405',
    [string]$Seeds = '7,42',
    [string]$EvalDomains = 'in_domain',
    [string]$Variants = 'baseline,no_prior,mask_e2r_r2r,no_prior_mask_e2r_r2r',
    [string]$Device = 'cpu',
    [switch]$ContinueOnError,
    [switch]$DryRun
)

$py = 'D:/Anaconda/Anaconda/envs/myenv/python.exe'
$script = 'python_impl/python_scripts/run_hybrid_prior_feature_retrain_matrix.py'

$args = @(
    $script,
    '--results-root', $ResultsRoot,
    '--seeds', $Seeds,
    '--eval-domains', $EvalDomains,
    '--variants', $Variants,
    '--device', $Device,
    '--skip-existing'
)

if ($ContinueOnError) {
    $args += '--continue-on-error'
}
if ($DryRun) {
    $args += '--dry-run'
}

& $py @args
