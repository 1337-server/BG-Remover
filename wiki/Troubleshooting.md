# Troubleshooting

> 🆘 **Scope:** Diagnose common runtime issues across CLI, GUI, and Flask apps.

## Missing model warnings

- **Symptom:** A toast or dialog reports that a model could not be loaded.
- **Cause:** The ONNX weights are missing, corrupted, or require authentication.
- **Fix:**
  1. Confirm that `MODEL_DIR` points to a writable directory.
  2. Delete the corrupted file in the model cache; BG-Remover will download it
     again.
  3. Set `HUGGINGFACEHUB_API_TOKEN` or `HF_API_TOKEN` when accessing private
     repositories.

The GUI highlights the model row in red and surfaces a tooltip with the HTTP
status code when downloads fail.

## GPU provider failures

- **Symptom:** Logs mention `CUDAExecutionProvider` or `DmlExecutionProvider`
  initialisation errors.
- **Behaviour:** The CLI, GUI, and Flask runtimes automatically retry with the
  CPU provider. You can continue processing, albeit more slowly.
- **Fix:** Update GPU drivers, ensure the correct ONNX Runtime package is
  installed, and verify that your hardware meets the provider requirements.

## Output or config saved to home directory

All runtimes now write to the project root (`input/`, `output/`,
`config/config.json`). If files appear in your home directory, confirm that:

- `BGR_CONFIG_PATH` is unset (or points to the project’s `config/config.json`).
- You are launching binaries from the repository root when using relative paths.

## Permission errors

- Verify write access to `output/` and the configured `model_dir`.
- On Windows, avoid storing the project inside protected folders like `Program
  Files` without elevated permissions.
- Inside Docker containers, mount host directories with appropriate ownership.

## Flask UI issues

- **Missing previews:** Ensure that static assets are served correctly; rebuilding
  the frontend bundle (if customised) restores hashed filenames.
- **Dark mode not applying:** Clear browser storage (localStorage) to reset the
  persisted theme toggle.

## GUI quirks

- **Advanced panels not collapsing:** Delete `config/config.json` to reset the
  stored layout state.
- **Tooltip text missing:** Confirm that you are running the latest Python
  version (3.10+). Older versions may not support the themed widget toolkit.

## CLI exit codes

Refer to [CLI Usage](CLI-Usage.md#exit-codes) when scripting. Non-zero exit codes
indicate that at least one image failed or the input path was invalid.
