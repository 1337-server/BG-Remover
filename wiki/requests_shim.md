# requests_shim.py — Minimal Requests-Compatible Downloader

## Overview
`requests_shim.py` provides a small compatibility layer that emulates enough of the `requests` API for streaming downloads. It is used when the real `requests` dependency is unavailable, such as in restricted Python environments.

## Role in the Project
- Allows `bg_removal.py` to download model weights without depending on `requests` at runtime.
- Keeps deployment footprints small, especially for bundled desktop apps where optional dependencies are trimmed.

## Key Components
- **Exceptions**: `RequestException` and `HTTPError` mirror the common exception hierarchy from `requests` so callers can handle errors uniformly.
- **`Response` class**: Wraps `urllib` responses and exposes `.status_code`, `.headers`, `.iter_content()`, and `.raise_for_status()` methods.
- **`get` function**: Accepts the same keyword arguments as `requests.get` (headers, streaming, timeout) and returns a shim `Response`.

## Implementation Notes
- Uses `urllib.request.urlopen` under the hood; this means proxy configuration follows standard library behaviour.
- `iter_content` yields chunks of bytes suitable for incremental hashing or file writes.
- Timeouts passed as tuples are treated like `requests`' connect/read timeouts (only the first value is used).
- Always call `response.close()` (or use a context manager) to release the underlying network resources.

## Invocation Example
```python
from requests_shim import get

with get("https://example.com/model.onnx", stream=True) as response:
    response.raise_for_status()
    for chunk in response.iter_content(1024 * 1024):
        ...  # write chunk to disk
```

> **Dependency Helper:** Transparent fallback for network downloads.
