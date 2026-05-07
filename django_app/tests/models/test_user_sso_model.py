import pytest
from django.contrib.auth import get_user_model
from django.db import IntegrityError

from redbox_app.redbox_core.models import (
    UserSSOAttribute,
)

User = get_user_model()


@pytest.mark.django_db
def test_user_sso_str(alice, sso_factory):
    sso = sso_factory(alice)

    assert str(sso) == f"{alice} SSO"


@pytest.mark.django_db
def test_related_emails(alice, sso_factory):
    sso = sso_factory(
        alice,
        related_emails=[
            "a@test.com",
            "b@test.com",
        ],
    )

    assert set(sso.related_emails) == {
        "a@test.com",
        "b@test.com",
    }


@pytest.mark.django_db
def test_related_emails_display(alice, sso_factory):
    sso = sso_factory(
        alice,
        related_emails=[
            "a@test.com",
            "b@test.com",
        ],
    )

    display = sso.related_emails_display

    assert "a@test.com" in display
    assert "b@test.com" in display


@pytest.mark.django_db
def test_sso_all_emails(alice, sso_factory):
    sso = sso_factory(
        alice,
        related_emails=["related@test.com"],
        contact_email="contact@test.com",
    )

    emails = sso.all_emails

    assert sso.email in emails
    assert "related@test.com" in emails
    assert "contact@test.com" in emails


@pytest.mark.django_db
def test_all_emails_display(alice, sso_factory):
    sso = sso_factory(
        alice,
        related_emails=["related@test.com"],
    )

    display = sso.all_emails_display

    assert "related@test.com" in display


@pytest.mark.django_db
def test_sso_name(alice, sso_factory):
    sso = sso_factory(
        alice,
        first_name="Alice",
        last_name="Smith",
    )

    assert sso.name == "Alice Smith"


@pytest.mark.django_db
def test_email_domains(alice, sso_factory):
    sso = sso_factory(
        alice,
        related_emails=[
            "a@example.com",
            "b@test.com",
        ],
    )

    domains = sso.email_domains

    assert "example.com" in domains
    assert "test.com" in domains


@pytest.mark.django_db
def test_user_sso_attribute_str(alice, sso_factory):
    sso = sso_factory(alice)

    attr = UserSSOAttribute.objects.create(
        sso=sso,
        attribute_type=UserSSOAttribute.AttributeType.RELATED_EMAILS,
        value="test@example.com",
    )

    assert str(attr) == f"{alice} (related_emails) -> test@example.com"


@pytest.mark.django_db
def test_user_sso_attribute_unique_constraint(alice, sso_factory):
    sso = sso_factory(alice)

    UserSSOAttribute.objects.create(
        sso=sso,
        attribute_type=UserSSOAttribute.AttributeType.RELATED_EMAILS,
        value="test@example.com",
    )

    with pytest.raises(IntegrityError):
        UserSSOAttribute.objects.create(
            sso=sso,
            attribute_type=UserSSOAttribute.AttributeType.RELATED_EMAILS,
            value="test@example.com",
        )
