import logging
import os

from pages import LandingPage
from playwright.sync_api import Page

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


BASE_URL = "https://dev.assist.uktrade.digital"


def test_support_pages(page: Page):
    # Landing page
    landing_page = LandingPage(page, BASE_URL)

    # Privacy page
    landing_page.navigate_to_privacy_page()

    # Accessibility page
    landing_page.navigate_to_accessibility_page()

    # Support page
    landing_page.navigate_to_support_page()
