import pytest
from redbox.api.wrapper import SensitiveValue, NonPicklableCallable


class TestSensitiveValue:
    @pytest.mark.parametrize("value", ["abc123", {"token": "x"}, None, 42])
    def test_stores_and_retrieves(self, value):
        assert SensitiveValue(value).get() == value

    @pytest.mark.parametrize("value", ["super_secret", "abc123"])
    def test_redacts_in_repr_and_str(self, value):
        sv = SensitiveValue(value)
        assert value not in repr(sv)
        assert value not in str(sv)


class TestNonPicklableCallable:
    @pytest.mark.parametrize(
        "fn, args, kwargs, expected",
        [
            (lambda: 42, [], {}, 42),
            (lambda x: x * 2, [5], {}, 10),
            (lambda x, y: x + y, [3, 4], {}, 7),
            (lambda x, y=10: x + y, [1], {"y": 20}, 21),
        ],
    )
    def test_calls_wrapped_fn(self, fn, args, kwargs, expected):
        assert NonPicklableCallable(fn)(*args, **kwargs) == expected

    @pytest.mark.parametrize("fn", [lambda: "secret", lambda x: x])
    def test_redacts_in_repr_and_str(self, fn):
        npc = NonPicklableCallable(fn)
        assert "secret" not in repr(npc)
        assert "secret" not in str(npc)
        assert repr(npc) == "NonPicklableCallable(**redacted**)"
