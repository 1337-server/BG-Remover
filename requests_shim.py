"""Minimal compatibility layer emulating :mod:`requests` for streaming downloads."""

from __future__ import annotations

import contextlib
import urllib.error
import urllib.request
from collections.abc import Iterator


class RequestException(Exception):
    """Base exception raised for network-related issues in the shim."""


class HTTPError(RequestException):
    """Raised when the remote server reports a non-success HTTP status code."""

    def __init__(self, message: str, *, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class Response:
    """Wrap an :class:`urllib.response.addinfourl` stream with ``requests``-like helpers."""

    def __init__(self, raw: urllib.response.addinfourl):
        self._raw = raw
        self.status_code = getattr(raw, "status", None) or raw.getcode()
        self.headers = dict(raw.headers.items()) if getattr(raw, "headers", None) else {}

    def __enter__(self) -> Response:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying stream, suppressing errors."""

        with contextlib.suppress(Exception):
            self._raw.close()

    def raise_for_status(self) -> None:
        """Raise :class:`HTTPError` when the response indicates a failure."""

        if self.status_code is None:
            return
        if 400 <= int(self.status_code):
            raise HTTPError(f"HTTP Error {self.status_code}", status_code=int(self.status_code))

    def iter_content(self, chunk_size: int = 8192) -> Iterator[bytes]:
        """Yield bytes from the response body in ``chunk_size`` increments."""

        while True:
            chunk = self._raw.read(chunk_size)
            if not chunk:
                break
            yield chunk


def get(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    stream: bool | None = None,
    timeout: float | tuple[float, float] | None = None,
) -> Response:
    """Return a :class:`Response` for ``url`` similar to :func:`requests.get`."""

    request = urllib.request.Request(url, headers=headers or {})
    try:
        timeout_value = timeout[0] if isinstance(timeout, tuple) else timeout
        raw = urllib.request.urlopen(request, timeout=timeout_value)
    except urllib.error.HTTPError as error:  # pragma: no cover - best-effort compatibility
        raise HTTPError(str(error), status_code=error.code) from error
    except urllib.error.URLError as error:
        raise RequestException(str(error)) from error
    except TimeoutError as error:
        raise RequestException(str(error)) from error
    return Response(raw)


__all__ = ["get", "HTTPError", "RequestException", "Response"]
