$ErrorActionPreference = 'Stop'

$py = "D:/Anaconda/Anaconda/envs/myenv/python.exe"
$script = "z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/src/run_tdoa_multiseed_longpush.py"
$h5 = "z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/data/source_localization_single_reflection_l1_stable_v3_w2.h5"
$init = "z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/results/_target_push_geom_val_1e4/capacity_w20_seed7_refine_tight360/seed_7/best_model.pt"
$root = "z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/results/_target_push_geom_val_1e4"

$configs = @(
    @{ Name = 'capacity_w20_grid_a_lr3e5_h002_b001_bs128_e220'; Lr = '3e-5'; Huber = '0.02'; Bound = '0.01'; Batch = '128'; Epochs = '220'; Dropout = '0.1'; Patience = '10'; MinLr = '1e-8' },
    @{ Name = 'capacity_w20_grid_b_lr4e5_h003_b000_bs128_e220'; Lr = '4e-5'; Huber = '0.03'; Bound = '0.0'; Batch = '128'; Epochs = '220'; Dropout = '0.1'; Patience = '10'; MinLr = '1e-8' },
    @{ Name = 'capacity_w20_grid_c_lr2e5_h0015_b0005_bs96_e220'; Lr = '2e-5'; Huber = '0.015'; Bound = '0.005'; Batch = '96'; Epochs = '220'; Dropout = '0.1'; Patience = '12'; MinLr = '1e-9' },
    @{ Name = 'capacity_w20_grid_d_lr1e5_h001_b000_bs96_e240'; Lr = '1e-5'; Huber = '0.01'; Bound = '0.0'; Batch = '96'; Epochs = '240'; Dropout = '0.1'; Patience = '12'; MinLr = '1e-9' }
)

$results = @()
foreach ($cfg in $configs) {
    $resultRoot = Join-Path $root $cfg.Name
    Write-Host "[run] $($cfg.Name)"
    & $py $script `
        --h5-path $h5 `
        --init-checkpoint $init `
        --results-root $resultRoot `
        --seeds 7 `
        --device cuda `
        --lr $cfg.Lr `
        --batch-size $cfg.Batch `
        --epochs $cfg.Epochs `
        --huber-delta-norm $cfg.Huber `
        --bound-penalty-weight $cfg.Bound `
        --model-width-mult 2.0 `
        --dropout-p $cfg.Dropout `
        --scheduler-patience $cfg.Patience `
        --scheduler-factor 0.5 `
        --scheduler-min-lr $cfg.MinLr `
        --early-stop-patience 0 `
        --target-geom-val-mae-s 1e-4

    $reportPath = Join-Path $resultRoot 'multiseed_longpush_report.json'
    if (-not (Test-Path $reportPath)) {
        throw "Missing report file: $reportPath"
    }
    $j = Get-Content -Raw -Path $reportPath | ConvertFrom-Json
    $results += [PSCustomObject]@{
        run = $cfg.Name
        geom_val_min = [double]$j.aggregate.geom_val_min
        geom_test_at_best = [double]$j.aggregate.geom_test_at_best_seed
        pass_rate = [double]$j.aggregate.pass_rate
        best_result_dir = [string]$j.aggregate.best_result_dir
    }
}

$sorted = $results | Sort-Object geom_val_min
$summaryPath = Join-Path $root 'capacity_w20_refine_grid_report.json'
$payload = [PSCustomObject]@{
    init_checkpoint = $init
    runs = $results
    best = $sorted[0]
}
$payload | ConvertTo-Json -Depth 6 | Out-File -FilePath $summaryPath -Encoding utf8

Write-Host "[grid-done] summary=$summaryPath"
$sorted | Format-Table -AutoSize
