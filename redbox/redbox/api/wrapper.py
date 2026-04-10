import asyncio
from typing import Any, Callable


class SensitiveValue:
    """Wrap any metadata value you never want serialized."""

    def __init__(self, value: Any | Callable[[], Any]):
        self._value = value

    def get(self) -> Any:
        if callable(self._value):
            result = self._value()
            if asyncio.iscoroutine(result):
                # Run the coroutine synchronously
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(result)
            return result
        return self._value

    def __repr__(self):
        return "SensitiveValue(**redacted**)"

    def __deepcopy__(self, memo):
        # Don't deepcopy the inner value — just return a new SensitiveValue
        # referencing the same callable/value. This prevents deepcopy from
        # walking into bound methods and their unpicklable __self__.
        new = SensitiveValue.__new__(SensitiveValue)
        new._value = self._value  # shallow reference, not deepcopy
        memo[id(self)] = new
        return new


class NonPicklableCallable:
    """Wraps a callable to prevent deepcopy/pickle from walking into it."""

    def __init__(self, fn: Callable):
        self._fn = fn

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)

    def __deepcopy__(self, memo):
        new = NonPicklableCallable.__new__(NonPicklableCallable)
        new._fn = self._fn  # shallow reference only
        memo[id(self)] = new
        return new

    def __copy__(self, memo=None):
        return NonPicklableCallable(self._fn)

    def __repr__(self):
        return "NonPicklableCallable(**redacted**)"
