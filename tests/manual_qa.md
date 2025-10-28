# Manual QA - Background Removal Alerts

## Date
- 2025-10-28

## Environment
- Browser: Firefox 121 (desktop)
- Backend: Local development server via `flask run`

## Scenarios
- **Single image form without file**
  - Navigated to `/image/remove-bg`.
  - Clicked "Remove background" with no file selected.
  - Verified inline error alert appears above the form with the message "Please choose an image to upload." and the form keeps focus on the alert.
- **Single image form socket unavailable**
  - Disconnected the websocket server (simulated by blocking requests).
  - Submitted the form and observed inline alert stating that the live preview channel is still connecting.
- **Folder form missing path**
  - Submitted the folder form without specifying a folder path.
  - Confirmed inline alert requests fixes before submission.
- **Folder form server error**
  - Forced an API error by pointing to a non-existent directory.
  - Observed inline alert summarizing the server error while keeping previous results visible.

## Notes
- Alerts remain visible until the underlying issue is resolved, ensuring important errors are not missed during subsequent submissions.
