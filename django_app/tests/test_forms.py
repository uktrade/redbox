import logging

import pytest
from django.contrib.auth import get_user_model
from django.http import QueryDict

from redbox_app.redbox_core.forms import (
    ToolAccessRuleForm,
    ToolSettingsForm,
    UserToolBulkAddForm,
)
from redbox_app.redbox_core.models import Tool, ToolAccessRule, UserTool

User = get_user_model()
logger = logging.getLogger(__name__)


def test_govuk_form_adds_input_classes():
    form = ToolSettingsForm()

    assert "govuk-input" in form.fields["name"].widget.attrs["class"]


def test_govuk_form_adds_textarea_classes():
    form = ToolSettingsForm()

    assert "govuk-textarea" in form.fields["description"].widget.attrs["class"]


@pytest.mark.django_db
def test_tool_access_rule_form_adds_htmx_attrs():
    form = ToolAccessRuleForm()

    attrs = form.fields["rule_type"].widget.attrs

    assert attrs["hx-get"]
    assert attrs["hx-target"] == "#id_value"
    assert attrs["hx-trigger"] == "change"


@pytest.mark.django_db
def test_tool_access_rule_form_sets_placeholder():
    form = ToolAccessRuleForm(
        initial={
            "rule_type": ToolAccessRule.RuleType.DOMAIN,
        }
    )

    placeholder = form.fields["value"].widget.attrs["placeholder"]

    assert "example.com" in placeholder


@pytest.mark.django_db
def test_tool_access_rule_form_normalizes_value():
    form = ToolAccessRuleForm(
        data={
            "rule_type": ToolAccessRule.RuleType.DOMAIN,
            "value": "  EXAMPLE.COM  ",
            "access_type": ToolAccessRule.AccessType.ALLOW,
        }
    )

    assert form.is_valid()

    assert form.cleaned_data["value"] == "example.com"


@pytest.mark.django_db
def test_tool_access_rule_form_rejects_email_address():
    form = ToolAccessRuleForm(
        data={
            "rule_type": ToolAccessRule.RuleType.DOMAIN,
            "value": "alice@example.com",
            "access_type": ToolAccessRule.AccessType.ALLOW,
        }
    )

    assert not form.is_valid()

    assert "Enter a domain like example.com" in form.errors["__all__"][0]


@pytest.mark.django_db
def test_bulk_add_form_accepts_valid_users(alice: User, default_tool: Tool):
    data = QueryDict(mutable=True)

    data.setlist("user_ids", [str(alice.pk)])

    data.update(
        {
            "role": UserTool.RoleType.USER,
            "access_type": UserTool.AccessType.ALLOW,
        }
    )

    form = UserToolBulkAddForm(tool=default_tool, data=data)

    assert form.is_valid()
