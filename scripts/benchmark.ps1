param(
    [int]$Count = 5,
    [int]$Warmup = 2,
    [string]$Model = "u2net",
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

# Use a *single-quoted* here-string so PowerShell doesn't expand { } inside f-strings
$pythonCode = @'
import statistics
import sys
from time import perf_counter
from PIL import Image
from app import create_app
from app.services import model_registry
from app.services import bg_remove

app = create_app({"BG_ACCELERATOR": "auto"})
model_registry.preload_models(config=app.config, model_names=[sys.argv[1]])
session = bg_remove.ensure_global_session(model_name=sys.argv[1], config=app.config)
model_name = bg_remove._session_model_name(session)

sample = Image.new("RGB", (512, 512), color=(128, 128, 128))

for _ in range(int(sys.argv[2])):
    bg_remove._run_inference(sample, session, model_name, log_timing=False)

timings = []
for _ in range(int(sys.argv[3])):
    start = perf_counter()
    mask = bg_remove._run_inference(sample, session, model_name, log_timing=False)
    mask.load()
    timings.append((perf_counter() - start) * 1000)

print(f"Model: {model_name}")
print(f"Runs: {len(timings)} (warm-up {sys.argv[2]})")
print(f"Mean: {statistics.mean(timings):.2f} ms")
print(f"Median: {statistics.median(timings):.2f} ms")
print(f"Min: {min(timings):.2f} ms, Max: {max(timings):.2f} ms")
'@

# Run the Python code with arguments
& $Python -c $pythonCode $Model $Warmup $Count
