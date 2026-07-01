<#
.SYNOPSIS
Orchestrates the ZenLLM benchmarking suite.

.DESCRIPTION
This script automates the compilation and execution of the zen-vector micro-benchmarks
and the zen-bench-runtime tests for both Normal and FlexLoad execution paths.

It is designed to run only when explicitly executed by the user.
#>

param (
    [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",
    [int]$MaxTokens = 16,
    [int]$Runs = 1
)

Write-Host ""
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host " WARNING: This benchmark may be heavy. Run only when plugged in. " -ForegroundColor Yellow
Write-Host "===================================================================" -ForegroundColor Cyan
Write-Host ""

# Ensure benchmarks directory exists
if (-not (Test-Path "benchmarks")) {
    New-Item -ItemType Directory -Path "benchmarks" | Out-Null
}

Write-Host "[1/3] Compiling and running zen-vector micro-benchmarks..." -ForegroundColor Green
# Running only the small dimension micro-benchmark to avoid massive CPU load
cargo bench -p zen-vector --bench search_bench

Write-Host "`n[2/3] Running zen-bench-runtime in Normal Mode..." -ForegroundColor Green
cargo run -p zen-bench-runtime --release -- `
    --model-id $Model `
    --max-tokens $MaxTokens `
    --runs $Runs `
    --skip-flexload `
    --output-json "benchmarks/normal.json"

Write-Host "`n[3/3] Running zen-bench-runtime in FlexLoad Mode..." -ForegroundColor Green
cargo run -p zen-bench-runtime --release -- `
    --model-id $Model `
    --max-tokens $MaxTokens `
    --runs $Runs `
    --use-flexload `
    --output-json "benchmarks/flexload.json"

Write-Host "`nDone! Raw results dumped to benchmarks/*.json" -ForegroundColor Green
