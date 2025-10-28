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
import io
import statistics
import sys
from time import perf_counter
from PIL import Image
from app import create_app
import bg_removal as bg_remove

app = create_app(run_startup_tasks=False)
session = bg_remove.ensure_global_session(model_name=sys.argv[1])
context = bg_remove.get_session_context()
model_name = context.model_name if context else sys.argv[1]

sample = Image.new("RGB", (512, 512), color=(128, 128, 128))
buffer = io.BytesIO()
sample.save(buffer, "PNG")
payload = buffer.getvalue()

for _ in range(int(sys.argv[2])):
    result = bg_remove.remove_background_bytes(payload, model_name=model_name)
    if result.image is not None:
        result.image.close()

timings = []
for _ in range(int(sys.argv[3])):
    start = perf_counter()
    result = bg_remove.remove_background_bytes(payload, model_name=model_name)
    if result.image is not None:
        result.image.close()
    timings.append((perf_counter() - start) * 1000)

print(f"Model: {model_name}")
print(f"Runs: {len(timings)} (warm-up {sys.argv[2]})")
print(f"Mean: {statistics.mean(timings):.2f} ms")
print(f"Median: {statistics.median(timings):.2f} ms")
print(f"Min: {min(timings):.2f} ms, Max: {max(timings):.2f} ms")
'@

# Run the Python code with arguments
& $Python -c $pythonCode $Model $Warmup $Count
