# Using the GUI

> 🖥️ **Scope:** Desktop runtime powered by `ttkbootstrap` with persistent
> configuration and advanced controls.

## Launching the GUI

```bash
python -m runtimes.gui.bg_remover_gui
```

The GUI reads from `config/config.json` and writes updated preferences back to
that file so the CLI and Flask runtimes stay in sync.

## Theme and layout

- **Persistent dark mode:** Toggle the 🌙 / ☀️ button in the title bar. Your
  choice is stored in the config file and restored on restart.
- **Collapsible advanced settings:** All advanced panels (models, matting,
  output options, background fill, etc.) return to the exact expanded/collapsed
  state from the previous session. They start collapsed by default to highlight
  the primary controls.
- **Tooltips everywhere:** Hover over any advanced control to view the valid
  range, default value, and a short usage hint. Ranges match the validation
  performed by `ProcessingOptions` in the core pipeline.

## Selecting models and storage

- **Model picker:** Choose any registered model from the dropdown. The selection
  persists between runs when you click “Save Settings”.
- **Model storage location:** Click “Browse” next to the model directory field to
  store weights on a fast disk or shared location. The GUI validates write
  access and updates `config/config.json` so other runtimes reuse the same cache.

## Processing images

1. **Single image tab** — Drag & drop or browse for an image. Click “Remove
   Background” to generate a preview and save the output. Outputs default to the
   `output/` directory when no destination is specified.
2. **Batch tab** — Select a folder. Results are written to `output/` unless you
   override the destination.
3. **Live status log** — Progress, warnings, and errors render inline. Failed
   items display in red and surface tooltips containing the error message.

## Advanced settings overview

| Section | Highlights |
| --- | --- |
| Matting | Alpha matting toggle, erode size (0–30), background/foreground thresholds (0–255). |
| Mask tweaks | Gaussian blur radius (0–25), smoothing ratio (0–1.0), edge refinement toggle. |
| Output | Preserve filenames, choose PNG/JPEG output format, select background fill colour or transparency. |
| Performance | Control max worker threads (1–16) and feather radius (0–50 px). |

Tooltips list min/max constraints and explain when a setting is ignored. The GUI
only enables incompatible options when the selected model supports them.

## Saving preferences

Click “Save Settings” to persist the current selections. The file
`config/config.json` is updated in-place. If the write fails, the GUI surfaces an
inline error and logs the exception for later inspection.
