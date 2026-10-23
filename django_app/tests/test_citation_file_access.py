import logging
from unittest.mock import AsyncMock

import pytest
from channels.db import database_sync_to_async
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile

from redbox.models.chain import Citation as AICitation
from redbox.models.chain import Source
from redbox_app.redbox_core.consumers import ChatConsumer, resolve_citable_file
from redbox_app.redbox_core.models import ChatMessage, Citation, File

User = get_user_model()


def make_file(user: User, name: str) -> File:
    return File.objects.create(
        user=user,
        original_file=SimpleUploadedFile(name, b"Lorem Ipsum."),
        status=File.Status.complete,
    )


def make_consumer(chat_message: ChatMessage, citable_files: list[File]) -> ChatConsumer:
    consumer = ChatConsumer()
    consumer._file_cache = {}  # noqa: SLF001
    consumer._citable_files = citable_files  # noqa: SLF001
    consumer.chat_message = chat_message
    consumer.send_to_client = AsyncMock()
    return consumer


@pytest.fixture
def alice_file(alice: User, s3_client) -> File:  # noqa: ARG001
    return make_file(alice, "alice_report.txt")


@pytest.fixture
def bob_file(bob: User, s3_client) -> File:  # noqa: ARG001
    return make_file(bob, "bob_secret.txt")


# === resolving citation references ===


def test_resolve_citable_file_only_matches_permitted_files():
    report = File(original_file="alice@example.com/report_20250101.pdf", original_file_name="report.pdf")
    notes = File(original_file="alice@example.com/notes_20250101.pdf", original_file_name="notes.pdf")
    files = [report, notes]

    assert resolve_citable_file("alice@example.com/report_20250101.pdf", files) is report
    assert resolve_citable_file("report_20250101.pdf", files) is report
    assert resolve_citable_file("report.pdf", files) is report

    # previously each of these matched the most recent file of any user via `original_file__endswith`
    assert resolve_citable_file("", files) is None
    assert resolve_citable_file("   ", files) is None
    assert resolve_citable_file(None, files) is None
    assert resolve_citable_file("pdf", files) is None
    assert resolve_citable_file("bob@example.com/secret_20250101.pdf", files) is None


def test_resolve_citable_file_rejects_ambiguous_names():
    first = File(original_file="alice@example.com/report_1.pdf", original_file_name="report.pdf")
    second = File(original_file="alice@example.com/report_2.pdf", original_file_name="report.pdf")

    assert resolve_citable_file("report.pdf", [first, second]) is None


# === handling citations from the AI ===


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_handle_citations_never_links_another_users_file(
    chat_message: ChatMessage, alice_file: File, bob_file: File, caplog
):
    consumer = make_consumer(chat_message, [alice_file])

    refs = ["", "txt", bob_file.original_file.name, alice_file.original_file.name]
    sources = [
        Source(source=ref, source_type=Citation.Origin.USER_UPLOADED_DOCUMENT, ref_id=f"ref_{i}")
        for i, ref in enumerate(refs, start=1)
    ]

    with caplog.at_level(logging.WARNING, logger="redbox_app.redbox_core.consumers"):
        await consumer.handle_citations([AICitation(sources=sources)])

    saved = await database_sync_to_async(
        lambda: list(Citation.objects.filter(chat_message=chat_message).values_list("citation_name", "file_id"))
    )()
    # unmatched document citations are dropped entirely, not saved without a file
    assert saved == [("ref_4", alice_file.id)]

    for ref_id in ("ref_1", "ref_2", "ref_3"):
        assert f"dropping citation {ref_id}: source does not match a permitted file" in caplog.text
    assert "dropping citation ref_4" not in caplog.text


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_handle_citations_keeps_external_citations(chat_message: ChatMessage, alice_file: File):
    consumer = make_consumer(chat_message, [alice_file])

    source = Source(source="https://www.gov.uk/guidance", source_type=Citation.Origin.GOV_UK, ref_id="ref_1")

    await consumer.handle_citations([AICitation(sources=[source])])

    saved = await database_sync_to_async(
        lambda: list(Citation.objects.filter(chat_message=chat_message).values_list("file_id", "url", "source"))
    )()
    assert saved == [(None, "https://www.gov.uk/guidance", Citation.Origin.GOV_UK)]


@pytest.mark.django_db(transaction=True)
@pytest.mark.asyncio
async def test_handle_citations_links_tool_knowledge_file(chat_message: ChatMessage, bob_file: File):
    # knowledge-base files belong to the tool admin, so they are citable via `knowledge_files`, not ownership
    consumer = make_consumer(chat_message, [bob_file])

    source = Source(
        source=bob_file.original_file.name, source_type=Citation.Origin.USER_UPLOADED_DOCUMENT, ref_id="ref_1"
    )

    await consumer.handle_citations([AICitation(sources=[source])])

    cited_file_ids = await database_sync_to_async(
        lambda: list(Citation.objects.filter(chat_message=chat_message).values_list("file_id", flat=True))
    )()
    assert cited_file_ids == [bob_file.id]
