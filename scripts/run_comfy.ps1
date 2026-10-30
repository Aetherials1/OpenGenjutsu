# Run ComfyUI on Windows for local development.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_comfy.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_comfy.ps1 -ComfyDir "D:\AI\ComfyUI"
#
# Then start OpenGenjutsu with ComfyUI's input dir exposed:
#   $env:OG_COMFY_INPUT_DIR = "<ComfyDir>\input"
#   python run.py

param(
    [string]$ComfyDir = "..\ComfyUI",
    [int]$Port = 8188,
    [switch]$LowVram,
    [switch]$Expose
)

$ErrorActionPreference = "Stop"
$comfy = Resolve-Path -LiteralPath $ComfyDir -ErrorAction SilentlyContinue

if (-not $comfy -or -not (Test-Path (Join-Path $comfy "main.py"))) {
    Write-Host "ComfyUI not found at '$ComfyDir'." -ForegroundColor Red
    Write-Host "Clone it first:  git clone https://github.com/comfyanonymous/ComfyUI.git"
    exit 1
}

$listen = if ($Expose) { "0.0.0.0" } else { "127.0.0.1" }
$args = @("main.py", "--listen", $listen, "--port", "$Port", "--preview-method", "none")
if ($LowVram) { $args += @("--lowvram", "--use-split-cross-attention") }

Write-Host "Starting ComfyUI from $comfy on http://${listen}:${Port}" -ForegroundColor Cyan
Write-Host "Point the backend at it with:" -ForegroundColor DarkGray
Write-Host "  `$env:OG_COMFY_INPUT_DIR = `"$(Join-Path $comfy 'input')`"" -ForegroundColor DarkGray
Write-Host ""

Push-Location $comfy
try {
    & python @args
}
finally {
    Pop-Location
}
