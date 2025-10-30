# CLI Usage

> 🛠️ **Scope:** Command-line utilities exposed via `runtimes.cli.bgr_cli`.

## Basic command

```bash
python -m runtimes.cli.bgr_cli remove --input ./input/example.png --output ./output/example_no_bg.png
```

The CLI defaults to `input/` and `output/` inside the project root, mirroring the
other runtimes. Configuration is read from and persisted to `config/config.json`.

## Providers and CPU fallback

Use `--provider` to hint preferred execution providers:

```bash
python -m runtimes.cli.bgr_cli remove --input ./input/photo.jpg --provider cuda --provider cpu
```

BG-Remover automatically falls back to the CPU provider when a GPU initialisation
fails, so workflows remain reliable on mixed hardware.

## Single vs batch processing

### Single file

```bash
python -m runtimes.cli.bgr_cli remove --input ./input/headshot.png --feather-radius 8
```

Outputs default to `output/<name>_no_bg.png` unless `--output` points to a file
or directory.

### Batch folder

```bash
python -m runtimes.cli.bgr_cli remove --input ./input/portraits --batch --model isnet-general-use
```

- Accepts directories containing PNG, JPG, BMP, and TIFF files.
- Generates a summary table with ✓/✗ status icons, elapsed time, and errors.
- Writes to `output/` when `--output` is omitted.

### Batch folder with explicit destination

```bash
python -m runtimes.cli.bgr_cli remove \
  --input ./input/catalogue \
  --output ./output/catalogue_run \
  --batch \
  --provider directml
```

## Persisting configuration

Add `--persist-config` to save the effective configuration back to
`config/config.json`. This preserves the model directory, provider hints, and
model selection for the GUI and Flask runtimes.

```bash
python -m runtimes.cli.bgr_cli remove --model-dir /mnt/models --persist-config
```

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | All requested images processed successfully. |
| `1` | At least one image failed or raised an exception. |
| `2` | Input path errors (missing files/directories). |

Remember to activate your Python environment (or install the package globally)
before running the CLI.
