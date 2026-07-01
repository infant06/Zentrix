<#
  ZenLLM end-to-end feature test (Windows / AMD CPU + NVIDIA CUDA).

  Tests each CLI feature one-by-one, verifies HuggingFace downloads/calls,
  and the OpenAI API + web UI server. Force-stops all zenllm processes at the end.

  Usage:
    powershell -ExecutionPolicy Bypass -File test_features.ps1 [-Port 11500]
#>
param(
    [int]$Port = 11500,
    [string]$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
)

$ErrorActionPreference = 'Continue'
$root     = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$zenllm   = Join-Path $root 'target\release\zenllm.exe'
$modelDir = Join-Path $root 'models'
$gguf     = 'qwen2.5-0.5b-instruct-q4_0.gguf'
$tokId    = 'Qwen/Qwen2.5-0.5B-Instruct'

# Native exe needs the CUDA runtime DLLs on PATH.
if (Test-Path $CudaBin) { $env:PATH = "$CudaBin;$env:PATH" }

$script:pass = 0
$script:fail = 0
$script:skip = 0

function Section($t) {
    Write-Host "`n============================================================" -ForegroundColor Cyan
    Write-Host " $t" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
}

function Test-Step {
    param([string]$Name, [scriptblock]$Body, [switch]$Optional)
    Write-Host "`n[TEST] $Name" -ForegroundColor Yellow
    try {
        $ok = & $Body
        if ($ok) { Write-Host "[PASS] $Name" -ForegroundColor Green; $script:pass++ }
        else {
            if ($Optional) { Write-Host "[SKIP] $Name" -ForegroundColor DarkYellow; $script:skip++ }
            else { Write-Host "[FAIL] $Name" -ForegroundColor Red; $script:fail++ }
        }
    } catch {
        if ($Optional) { Write-Host "[SKIP] $Name : $($_.Exception.Message)" -ForegroundColor DarkYellow; $script:skip++ }
        else { Write-Host "[FAIL] $Name : $($_.Exception.Message)" -ForegroundColor Red; $script:fail++ }
    }
}

function Quote-Args {
    param([string[]]$A)
    # PowerShell 5.1 Start-Process does not quote array elements containing spaces.
    ($A | ForEach-Object { if ($_ -match '[\s]') { '"' + $_ + '"' } else { $_ } }) -join ' '
}

function Invoke-Zen {
    param([string[]]$CliArgs, [int]$TimeoutSec = 240)
    $out = New-TemporaryFile
    $err = New-TemporaryFile
    $line = Quote-Args $CliArgs
    $p = Start-Process -FilePath $zenllm -ArgumentList $line -NoNewWindow -PassThru `
            -RedirectStandardOutput $out.FullName -RedirectStandardError $err.FullName
    $null = $p.Handle  # cache handle so ExitCode populates reliably (PS 5.1 quirk)
    if (-not $p.WaitForExit($TimeoutSec * 1000)) { try { $p.Kill() } catch {}; throw "timed out after ${TimeoutSec}s" }
    $stdout = Get-Content $out.FullName -Raw -ErrorAction SilentlyContinue
    $stderr = Get-Content $err.FullName -Raw -ErrorAction SilentlyContinue
    Remove-Item $out.FullName, $err.FullName -Force -ErrorAction SilentlyContinue
    return [pscustomobject]@{ Code = $p.ExitCode; Out = "$stdout`n$stderr" }
}

function Stop-AllZen {
    Get-Process -Name zenllm -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 500
}

# ---------------------------------------------------------------------------
Section "Pre-flight"
Test-Step "zenllm.exe exists and prints version" {
    if (-not (Test-Path $zenllm)) { throw "not built: $zenllm (run scripts/build.ps1 -Target cli)" }
    $r = Invoke-Zen -CliArgs @('--version') -TimeoutSec 30
    Write-Host $r.Out.Trim()
    $r.Code -eq 0
}
Test-Step "target model present" { Test-Path (Join-Path $modelDir $gguf) }

# ---------------------------------------------------------------------------
Section "Diagnostics (hardware detection: AMD CPU + CUDA)"
Test-Step "doctor detects CPU + CUDA and reports healthy" {
    $r = Invoke-Zen -CliArgs @('doctor') -TimeoutSec 120
    Write-Host $r.Out
    ($r.Out -match 'CUDA runtime available') -and ($r.Out -match 'CPU:')
}
Test-Step "runtime planner reports execution modes" {
    $r = Invoke-Zen -CliArgs @('runtime') -TimeoutSec 120
    Write-Host $r.Out
    $r.Out -match '(?i)mode'
}
Test-Step "models list" {
    $r = Invoke-Zen -CliArgs @('models','list') -TimeoutSec 60
    Write-Host $r.Out
    $r.Code -eq 0
} -Optional
Test-Step "cache list" {
    $r = Invoke-Zen -CliArgs @('cache','list') -TimeoutSec 60
    Write-Host $r.Out
    $r.Code -eq 0
} -Optional

# ---------------------------------------------------------------------------
Section "HuggingFace download + call (tokenizer fetch during inference)"
Test-Step "CPU inference on AMD - fetches tokenizer from HF and generates" {
    $r = Invoke-Zen -CliArgs @('run','--mode','cpu-only','-i','Reply with one short sentence about Rust.',
                       'auto','-m',$modelDir,'--format','gguf','-f',$gguf,'--tok-model-id',$tokId) -TimeoutSec 300
    Write-Host $r.Out
    $r.Out -match 'T/s'
}
Test-Step "GPU inference on CUDA - auto mode with fallback" {
    $r = Invoke-Zen -CliArgs @('run','--mode','auto','--allow-fallback','-i','Say hello in one sentence.',
                       'auto','-m',$modelDir,'--format','gguf','-f',$gguf,'--tok-model-id',$tokId) -TimeoutSec 300
    Write-Host $r.Out
    $r.Out -match 'T/s'
}

# ---------------------------------------------------------------------------
Section "Server: OpenAI API + Web UI"
Test-Step "serve starts, /v1/models + /v1/chat/completions + /ui respond" {
    Stop-AllZen
    $log = Join-Path $root 'serve_test.log'
    $srvArgs = @('serve','--mode','auto','--allow-fallback','--host','127.0.0.1','--port',"$Port",
                 'auto','-m',$modelDir,'--format','gguf','-f',$gguf,'--tok-model-id',$tokId)
    $srv = Start-Process -FilePath $zenllm -ArgumentList (Quote-Args $srvArgs) -NoNewWindow -PassThru `
             -RedirectStandardOutput $log -RedirectStandardError "$log.err"
    try {
        $base = "http://127.0.0.1:$Port"
        $ready = $false
        for ($i = 0; $i -lt 90; $i++) {
            if ($srv.HasExited) {
                $tail = (Get-Content $log,"$log.err" -Tail 15 -ErrorAction SilentlyContinue) -join "`n"
                throw "server exited early (code $($srv.ExitCode)).`n$tail"
            }
            try {
                $m = Invoke-WebRequest "$base/v1/models" -TimeoutSec 3 -UseBasicParsing
                if ($m.StatusCode -eq 200) { $ready = $true; break }
            } catch {}
            Start-Sleep -Seconds 2
        }
        if (-not $ready) { throw "server did not become ready on $base" }
        Write-Host "  /v1/models -> 200 OK"

        $body = @{ model='default'; messages=@(@{ role='user'; content='Hi' }); max_tokens=16 } | ConvertTo-Json -Depth 5
        $chat = Invoke-WebRequest "$base/v1/chat/completions" -Method POST -Body $body -ContentType 'application/json' -TimeoutSec 120 -UseBasicParsing
        if ($chat.StatusCode -ne 200) { throw "chat completion HTTP $($chat.StatusCode)" }
        Write-Host "  /v1/chat/completions -> 200 OK"

        $ui = Invoke-WebRequest "$base/ui" -TimeoutSec 10 -UseBasicParsing
        if ($ui.StatusCode -ne 200) { throw "web UI HTTP $($ui.StatusCode)" }
        Write-Host "  /ui -> 200 OK ($($ui.Content.Length) bytes)"
        $true
    } finally {
        if (-not $srv.HasExited) { try { $srv.Kill() } catch {} }
    }
}

# ---------------------------------------------------------------------------
Section "Force-stopping all zenllm processes"
Stop-AllZen
Remove-Item (Join-Path $root 'serve_test.log'), (Join-Path $root 'serve_test.log.err') -Force -ErrorAction SilentlyContinue
Write-Host "All zenllm processes stopped." -ForegroundColor Green

# ---------------------------------------------------------------------------
Section "SUMMARY"
Write-Host ("PASS: {0}  FAIL: {1}  SKIP: {2}" -f $script:pass, $script:fail, $script:skip) -ForegroundColor White
if ($script:fail -eq 0) { Write-Host "ALL REQUIRED TESTS PASSED" -ForegroundColor Green; exit 0 }
else { Write-Host "SOME TESTS FAILED" -ForegroundColor Red; exit 1 }
