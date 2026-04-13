import pytest
from redbox.api.wrapper import SensitiveValue


class TestSensitiveValue:
    @pytest.mark.parametrize("value", ["abc123", {"token": "x"}, None, 42])
    def test_stores_and_retrieves(self, value):
        assert SensitiveValue(value).get() == value

    @pytest.mark.parametrize("value", ["super_secret", "abc123"])
    def test_redacts_in_repr_and_str(self, value):
        sv = SensitiveValue(value)
        assert value not in repr(sv)
        assert value not in str(sv)
