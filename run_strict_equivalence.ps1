param(
    [string]$PythonCommand = "python",
    [string]$MatlabCommand = "matlab"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

Write-Host "[1/4] Generating strict shared dataset..."
& $PythonCommand "python_impl/python_scripts/generate_strict_dataset.py"

Write-Host "[2/4] Running strict MATLAB equivalence..."
if (-not (Get-Command $MatlabCommand -ErrorAction SilentlyContinue)) {
    throw "MATLAB command '$MatlabCommand' not found in PATH."
}
& $MatlabCommand -batch "try, run('matlab_impl/run_equivalence_matlab_strict.m'); catch ME, disp(getReport(ME,'extended','hyperlinks','off')); exit(1); end"

Write-Host "[3/4] Running strict Python equivalence..."
& $PythonCommand "python_impl/python_scripts/run_strict_equivalence.py"

Write-Host "[4/4] Building convergence difference report and figures..."
& $PythonCommand "python_impl/python_scripts/compare_strict_convergence.py"

Write-Host "Done. Outputs are in python_impl/python_scripts/."
