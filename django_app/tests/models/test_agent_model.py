from redbox_app.redbox_core.models import Agent


def test_agents_list_not_empty():
    field = Agent._meta.get_field("name")
    choices = field.choices

    assert choices is not None, "List of agent names should not be None"

    assert len(choices) > 0, "List of agent names should not be empty"


def test_agents_list_contains_only_strings():
    field = Agent._meta.get_field("name")
    choices = field.choices

    for value, _label in choices:
        assert isinstance(value, str), "List of agent names must be strings"
        assert value.strip() != "", "List of agent names cannot be empty strings"
