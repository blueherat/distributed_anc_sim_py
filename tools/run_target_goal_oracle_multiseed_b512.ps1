$ErrorActionPreference = 'Stop'

$py = 'D:/Anaconda/Anaconda/envs/myenv/python.exe'
$train = 'python_impl/python_scripts/train_hybrid_deep_fxlms_single_control.py'
$eval = 'python_impl/python_scripts/evaluate_hybrid_deep_fxlms_single_control.py'
$agg = 'python_impl/python_scripts/aggregate_hybrid_ablation_results.py'
$h5 = 'python_impl/python_scripts/cfxlms_qc_dataset_single_control.h5'

$resultsRoot = 'python_impl/experiments/anc_single_control/target_goal_oracle_multiseed_b512'
$configName = 'cfg_oracle_idxdirect_b512'
$seeds = @(7, 42, 123, 999, 20260403)

foreach ($seed in $seeds) {
    $seedDir = Join-Path $resultsRoot "$configName/seed_$seed"
    $trainDir = Join-Path $seedDir 'train'
    $evalDir = Join-Path $seedDir 'eval'

    & $py $train `
        --h5-path $h5 `
        --output-dir $trainDir `
        --seed $seed `
        --ablation-tag $configName `
        --curriculum-levels '1,2,3' `
        --epochs-per-level '1,1,1' `
        --fusion-mode cross `
        --feature-encoding ri `
        --basis-dim 512 `
        --lr 0 `
        --weight-decay 0 `
        --grad-clip-norm 0 `
        --lambda-reg 0 `
        --loss-domain freq `
        --wopt-supervision-weight 0 `
        --acoustic-loss-weight 1 `
        --index-direct-lookup `
        --index-direct-init-wopt `
        --val-frac 0.001 `
        --nr-margin-weight 0 `
        --device cpu

    & $py $eval `
        --checkpoint-path (Join-Path $trainDir 'final_hybrid_deep_fxlms.pt') `
        --h5-path $h5 `
        --output-dir $evalDir `
        --device cpu
}

& $py $agg --results-root $resultsRoot
