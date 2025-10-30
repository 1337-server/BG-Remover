# Configuration Reference

> ⚙️ **Scope:** Settings shared by the CLI, GUI, and Flask runtimes.

BG-Remover persists configuration to `config/config.json`. This table lists the
available keys, defaults, and acceptable values. All numeric ranges match the
validation performed in `bgremover_core.processing.pipeline.ProcessingOptions`.

## Core settings

| Key | Default | Description |
| --- | --- | --- |
| `model_dir` | `~/.cache/bg-remover/models` (overridden by GUI/CLI when saving) | Directory that caches ONNX weights. Use an absolute path for portability. |
| `default_model` | `isnet-general-use` | Model used when the runtime is launched without an explicit selection. |
| `provider_hints` | `[]` | Ordered list of ONNX Runtime providers to prioritise (e.g. `CUDAExecutionProvider`, `DmlExecutionProvider`, `CPUExecutionProvider`). |
| `log_level` | `INFO` | Logging verbosity recognised by Python’s logging module. |

These settings are merged with environment variables (`MODEL_DIR`,
`BGR_DEFAULT_MODEL`, `BGR_PROVIDER_HINTS`, `BGR_LOGLEVEL`).

## Advanced processing options

Runtime UIs expose additional knobs that are passed to the pipeline. Not every
model supports every option; incompatible selections are ignored.

| Option | Default | Valid range / values | Notes |
| --- | --- | --- | --- |
| `feather_radius` | `3` | `0` – `50` | Softens the matte edges after inference. |
| `resize_mode` | `stretch` | `stretch`, `longer-side`, `pad` | Controls preprocessing resize strategy. |
| `alpha_matting` | `false` | Boolean | Enables matting refinement for portraits. |
| `foreground_threshold` | `240` | `0` – `255` | Only used when `alpha_matting` is true. |
| `background_threshold` | `10` | `0` – `255` | Only used when `alpha_matting` is true. |
| `erode_size` | `10` | `0` – `30` | Kernel size for matting pre-processing. |
| `smoothing` | `0.0` | `0.0` – `1.0` | Legacy smoothing ratio (used when `mask_blur` is zero). |
| `mask_blur` | `0.0` | `0.0` – `25.0` | Gaussian blur radius applied to the alpha mask. |
| `edge_refinement` | `false` | Boolean | Enables contour-aware refinement. |
| `post_process_mask` | `true` | Boolean | Toggle for post-processing stage. |
| `only_mask` | `false` | Boolean | Output the alpha mask instead of a composited image. |
| `mask_threshold` | `0.0` | `0.0` – `1.0` | Threshold used when `only_mask` is enabled. |
| `cut_out_mode` | `object` | `object`, `mask`, `bbox` | Determines which region is returned. |
| `background_mode` | `clear` | `clear`, `fill`, `none` | `fill` activates `background_color`. |
| `background_color` | `null` | RGB tuple | Provide `[r, g, b]` when filling the background. |
| `output_format` | `PNG` | `PNG`, `JPEG`, `WEBP` | Limits depend on PIL support. |
| `preserve_names` | `false` | Boolean | Keeps original filenames when saving batches. |
| `max_workers` | `1` | `1` – `16` | Number of worker threads in batch mode. |

The GUI writes these options into the config file when “Save Settings” is used.
The Flask app stores its UI state client-side but synchronises persistent
settings through the same configuration object.

## Configuration tips

- Keep `config/config.json` under version control if you want reproducible
  deployments.
- When distributing prebuilt binaries, include `config/config.json` and the
  `input/`/`output/` folders so paths resolve without manual setup.
- Use absolute paths for `model_dir` when running inside Docker or on shared
  storage to avoid permission issues.
