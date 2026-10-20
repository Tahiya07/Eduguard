from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Iterator

_events: dict[str, threading.Event] = {}
_events_lock = threading.Lock()
_current_event: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar(
    "eduguard_operation_event", default=None
)


class OperationCancelled(RuntimeError):
    """Raised when a user explicitly cancels a running operation."""


@contextmanager
def operation_context(operation_id: str) -> Iterator[threading.Event]:
    with _events_lock:
        # Reuse a cancellation signal if the user closed the modal before the
        # request reached its worker thread.
        event = _events.get(operation_id)
        if event is None:
            event = threading.Event()
            _events[operation_id] = event
    token = _current_event.set(event)
    try:
        yield event
    finally:
        _current_event.reset(token)
        with _events_lock:
            if _events.get(operation_id) is event:
                _events.pop(operation_id, None)


def cancel_operation(operation_id: str) -> bool:
    if not operation_id or len(operation_id) > 128:
        return False
    with _events_lock:
        event = _events.get(operation_id)
        if event is None:
            # Remember early cancellation until the matching request registers.
            event = threading.Event()
            _events[operation_id] = event
        event.set()
        # Bound entries if a client submits cancellation IDs for requests that
        # never arrive. Active operations retain their own Event references.
        while len(_events) > 512:
            oldest_id = next(iter(_events))
            if oldest_id == operation_id and len(_events) > 1:
                oldest_id = next(key for key in _events if key != operation_id)
            _events.pop(oldest_id, None)
        return True


def current_event() -> threading.Event | None:
    return _current_event.get()


def check_cancelled() -> None:
    event = current_event()
    if event is not None and event.is_set():
        raise OperationCancelled("Operation cancelled by the user.")


def llama_stopping_criteria():
    """Return llama-cpp stopping criteria that observes the active operation."""
    event = current_event()
    if event is None:
        return None
    try:
        from llama_cpp import StoppingCriteriaList
        return StoppingCriteriaList([lambda _input_ids, _logits: event.is_set()])
    except Exception:
        # Non-llama callers and incompatible optional builds remain supported.
        return None
