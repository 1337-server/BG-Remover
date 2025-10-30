# Using the Flask Web App

> 🌐 **Scope:** Browser-based runtime with single-image and batch pages backed by
> the shared `bgremover_core` pipeline.

## Starting the server

```bash
export FLASK_APP=runtimes.flask.app
python -m flask run --debug
```

For production deployments use Gunicorn or another WSGI server:

```bash
gunicorn "runtimes.flask.app:create_app()" --bind 0.0.0.0:8080 --workers 4
```

The Flask runtime reads configuration from `config/config.json` and stores
outputs in `output/`.

## Navigation structure

- **Single** — Default landing page for ad-hoc uploads. Drag & drop or use the
  upload button to queue images. A preview pane is embedded in the upload grid so
  results render alongside the controls.
- **Batch** — Dedicated page for folder or archive processing. Upload a ZIP
  containing images to process multiple files at once. Summary details (success
  count, elapsed time) and download links appear below the upload grid.
- The navigation bar at the top keeps both pages one click away and displays the
  current dark/light mode toggle.

## Advanced settings

Advanced options expand underneath the preview area and stretch across the full
width for clarity. Tooltips describe min/max values and match the validations in
`ProcessingOptions`. Settings include model selection, feather radius, matting,
mask blurs, background fill, and output format.

Preferences persist via the same config file used by the CLI and GUI. Browser
localStorage stores transient UI state (last active tab, theme, etc.).

## Theme and accessibility

- **Dark mode everywhere:** Toggling the theme updates the `<body>`, root
  container, cards, modals, and dropdowns to maintain contrast.
- **Keyboard shortcuts:** `⌘/Ctrl + Enter` starts processing when focus is inside
  the upload form.
- **Status feedback:** Toasts summarise background activity while a history drawer
  lists recent jobs.

## Batch downloads

Completed batch jobs surface a consolidated ZIP download alongside per-image
links. These assets live in the `output/` directory and follow the same naming
conventions as the CLI.

## API endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Render the single-image UI. |
| `GET` | `/batch` | Render the batch/folder UI. |
| `POST` | `/process` | Process one or more uploaded images. |
| `POST` | `/batch/process` | Process ZIP/folder uploads asynchronously. |
| `GET` | `/result/<id>` | Stream processed images for previews. |
| `GET` | `/download/<id>` | Provide download links for single results and ZIP archives. |
| `GET` | `/history` | Return metadata for previous jobs in the current session. |

Use these routes to integrate with automation pipelines or custom front-ends.
