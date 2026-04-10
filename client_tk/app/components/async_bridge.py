"""async_bridge.py — non-blocking API calls for Tkinter.

Usage
-----
    from client_tk.app.components.async_bridge import run_async

    def on_done(result, error):
        if error:
            show_error(str(error))
        else:
            populate_table(result)

    run_async(root_widget, api_client.list_templates, callback=on_done)

``run_async`` submits *fn* to a background daemon thread. When it finishes,
the result (or exception) is delivered to *callback* via ``widget.after(0,
...)`` so it runs safely on the Tk main thread.  The Tkinter widget is only
used for scheduling and is never accessed from the worker thread.
"""
from __future__ import annotations

import threading
from typing import Any, Callable


def run_async(
    widget,
    fn: Callable[[], Any],
    *,
    callback: Callable[[Any, Exception | None], None] | None = None,
    args: tuple = (),
    kwargs: dict | None = None,
) -> threading.Thread:
    """Run *fn(*args, **kwargs)* in a background thread.

    Parameters
    ----------
    widget:
        Any live Tkinter widget used to schedule the callback via
        ``widget.after(0, ...)``.
    fn:
        The callable to execute off the main thread (e.g. an API call).
    callback:
        ``callback(result, error)`` — called on the Tk main thread once *fn*
        completes.  *error* is ``None`` on success, an ``Exception`` on
        failure.  *result* is ``None`` on failure.
    args / kwargs:
        Forwarded to *fn*.
    """
    _kwargs = kwargs or {}

    def _worker():
        try:
            result = fn(*args, **_kwargs)
            if callback is not None:
                try:
                    widget.after(0, lambda: callback(result, None))
                except Exception:
                    return
        except Exception as exc:  # noqa: BLE001
            if callback is not None:
                try:
                    widget.after(0, lambda e=exc: callback(None, e))
                except Exception:
                    return

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
