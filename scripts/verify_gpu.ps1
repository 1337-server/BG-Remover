param(
    [string]$Python = $null
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not $Python) {
    if (Test-Path .\.venv\Scripts\python.exe) {
        $Python = (Resolve-Path .\.venv\Scripts\python.exe).Path
    } else {
        $Python = "python"
    }
}

$env:PYTHONPATH = (Resolve-Path .).Path

& $Python - <<'PY'
import json
import platform

try:
    import onnxruntime as ort
except ModuleNotFoundError:
    ort = None

try:
    import torch
except ModuleNotFoundError:
    torch = None

providers = [] if ort is None else list(ort.get_available_providers())
summary = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "onnxruntime_providers": providers,
    "torch_cuda_available": bool(torch and torch.cuda.is_available()),
    "torch_device": torch.cuda.get_device_name(0) if torch and torch.cuda.is_available() else None,
}
print(json.dumps(summary, indent=2))
PY
