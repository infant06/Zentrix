<#
.SYNOPSIS
    Build ZenLLM (the `zenllm` CLI/server and/or the `zentrix` Python bridge).

.DESCRIPTION
    Pins the CUDA compute capability so kernels compile for your GPU, then builds
    the requested targets. Run from a "x64 Native Tools Command Prompt for VS"
    (or a shell where MSVC `cl.exe` is on PATH) on Windows.

.PARAMETER ComputeCap
    CUDA compute capability without the dot (e.g. 75 for Turing, 86 for Ampere,
    89 for Ada). Defaults to $env:CUDA_COMPUTE_CAP or auto-detection by the build.

.PARAMETER Target
    What to build: 'cli' (zenllm), 'python' (zentrix), or 'all'. Default: 'cli'.

.PARAMETER Cpu
    Build without CUDA (CPU-only).

.EXAMPLE
    ./scripts/build.ps1 -ComputeCap 75 -Target all
#>
param(
    [string]$ComputeCap = $env:CUDA_COMPUTE_CAP,
    [ValidateSet('cli', 'python', 'all')][string]$Target = 'cli',
    [switch]$Cpu
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Cpu -and $ComputeCap) {
    $env:CUDA_COMPUTE_CAP = $ComputeCap
    Write-Host "[build] CUDA_COMPUTE_CAP=$($env:CUDA_COMPUTE_CAP)"
}

$features = if ($Cpu) { 'code-execution' } else { 'cuda,code-execution' }

if ($Target -in @('cli', 'all')) {
    Write-Host "[build] cargo build --release -p zenllm --features $features"
    cargo build --release -p zenllm --features $features
}

if ($Target -in @('python', 'all')) {
    Write-Host "[build] maturin develop --release --features `"$features`" (run inside your venv)"
    Push-Location (Join-Path $root 'crates/zen-python')
    maturin develop --release --features $features
    Pop-Location
}

Write-Host "[build] done."
