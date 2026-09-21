# Creates the Windows-native Isaac Lab environment used by `hls train|play|export`.
# Isaac Sim 5.x is officially supported on Windows x86_64 (not WSL2), Python 3.11.
# Usage (PowerShell):  .\scripts\setup-isaaclab-windows.ps1 [-VenvDir C:\path]
param(
    [string]$VenvDir = "$env:USERPROFILE\dev\venvs\isaaclab",
    [string]$IsaacLabVersion = "2.3.2.post1"
)
$repo = Split-Path -Parent $PSScriptRoot
$py = "$VenvDir\Scripts\python.exe"
$env:OMNI_KIT_ACCEPT_EULA = "YES"   # NVIDIA Omniverse EULA; required for any non-interactive Kit launch

function Step([string[]]$cmd) {
    Write-Host ">> $($cmd -join ' ')"
    & $cmd[0] $cmd[1..($cmd.Length - 1)] 2>&1 | ForEach-Object { "$_" }
    if ($LASTEXITCODE -ne 0) { throw "step failed ($LASTEXITCODE): $($cmd -join ' ')" }
}

if (-not (Test-Path $py)) { Step @("uv", "venv", $VenvDir, "--python", "3.11") }
Step @("uv", "pip", "install", "-p", $py, "torch==2.7.0", "torchvision==0.22.0",
       "--index-url", "https://download.pytorch.org/whl/cu128")
Step @("uv", "pip", "install", "-p", $py, "isaaclab[isaacsim,all]==$IsaacLabVersion",
       "--extra-index-url", "https://pypi.nvidia.com", "--index-strategy", "unsafe-best-match")
Step @("uv", "pip", "install", "-p", $py, "-e", "$repo[isaac]")
Step @($py, "-c", "import torch, isaaclab; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('isaaclab', isaaclab.__version__)")
