import logging
import os

import pytest
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from pages import LandingPage
from playwright.sync_api import Page

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@pytest.fixture(scope="class")
def live_server_url():
    """Provide live server URL to test class."""
    server = StaticLiveServerTestCase
    server.setUpClass()
    yield server.live_server_url
    server.tearDownClass()


@pytest.mark.django_db(transaction=True)
def test_user_journey(page: Page, live_server_url):
    """End to end user journey test.

    Simulates a single user journey through the application, running against the full suite of microservices.

    Uses the Page Object Model - see https://pinboard.in/u:brunns/t:page-object for some resources explaining this.
    Please add to the page objects in `pages.py` where necessary - don't put page specific logic at this level.

    We should not be asserting anything about AI generated content in this test, aside from asserting that there
    is some."""

    # create_user(email_address)

    # Landing page
    landing_page = LandingPage(page, live_server_url)

    # Sign in
    chats_page = landing_page.sign_in()

    # Settings - My details page
    my_details_page = chats_page.navigate_my_details()
    my_details_page.name = "Roland Hamilton-Jones"
    my_details_page.update()
    assert my_details_page.name == "Roland Hamilton-Jones"

    # Documents page
    my_details_page.navigate_to_documents()

    # # Upload files
    # # document_upload_page = documents_page.navigate_to_upload()
    # upload_files: Sequence[Path] = [f for f in TEST_ROOT.parent.glob("*.md") if f.stat().st_size < 10000]
    # print("upload files: ", upload_files)
    # documents_page = documents_page.upload_documents(upload_files)

    # document_rows = documents_page.all_documents
    # print("document_rows: ", document_rows)
    # assert {r.filename for r in document_rows} == {f.name for f in upload_files}
    # assert documents_page.document_count() == original_doc_count + len(upload_files)
    # documents_page.wait_for_documents_to_complete()

    # # Chats page
    # chats_page = documents_page.navigate_to_chats()
    # chats_page.write_message = "What architecture is in use?"
    # chats_page = chats_page.send()
    # logger.debug("page: %s", chats_page)
    # latest_chat_response = chats_page.wait_for_latest_message()
    # assert latest_chat_response.text
    # # Commented out until we make this visible
    # assert chats_page.selected_llm == "gpt-4o (default)"

    # # Give user feedback
    # chats_page.feedback_stars = 2
    # chats_page.improve()
    # chats_page.feedback_chips = ["Inaccurate"]
    # chats_page.feedback_text = "Could be better."
    # chats_page.submit_feedback()

    # # Select files
    # chats_page = chats_page.start_new_chat()
    # files_to_select = {f.name for f in upload_files if "README" in f.name}
    # chats_page.selected_file_names = files_to_select
    # chats_page.write_message = "What licence is in use?"
    # chats_page = chats_page.send()

    # assert chats_page.selected_file_names == files_to_select
    # latest_chat_response = chats_page.wait_for_latest_message()
    # assert latest_chat_response.text

    # # Use specific routes
    # for keyword, route, select_file, should_have_citation in [
    #     ("search", "search", False, True),
    #     ("search", "search", True, True),
    # ]:
    #     question = f"@{keyword} What do I need to install?"
    #     logger.info("Asking %r", question)
    #     chats_page.write_message = question
    #     if select_file:
    #         current_files = files_to_select.copy()
    #         chats_page.selected_file_names = current_files
    #         logger.info("selected %s", current_files)
    #     else:
    #         chats_page.selected_file_names = []
    #     chats_page = chats_page.send()
    #     latest_chat_response = chats_page.wait_for_latest_message()
    #     assert latest_chat_response.text
    #     assert latest_chat_response.route.startswith(route)
    #     if should_have_citation:
    #         citations_page = latest_chat_response.navigate_to_citations()
    #         chats_page = citations_page.back_to_chat()
    #         assert any(file in latest_chat_response.sources for file in files_to_select)
    #     else:
    #         assert len(latest_chat_response.sources) == 0

    # # Delete a file
    # documents_page = chats_page.navigate_to_documents()
    # pre_delete_doc_count = documents_page.document_count()
    # document_delete_page = documents_page.delete_latest_document()
    # documents_page = document_delete_page.confirm_deletion()
    # assert documents_page.document_count() == pre_delete_doc_count - 1

    # # Delete a chat
    # chats_page = documents_page.navigate_to_chats()
    # pre_chats_count = chats_page.count_chats()
    # chats_page.delete_first_chat()
    # assert chats_page.count_chats() == pre_chats_count - 1


def test_support_pages(page: Page, live_server_url):
    # Landing page
    landing_page = LandingPage(page, live_server_url)

    # Privacy page
    landing_page.navigate_to_privacy_page()

    # Accessibility page
    landing_page.navigate_to_accessibility_page()

    # Support page
    landing_page.navigate_to_support_page()
