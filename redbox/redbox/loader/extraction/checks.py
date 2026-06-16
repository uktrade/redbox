import logging
from io import BytesIO
from typing import Tuple

import fitz
import math

logger = logging.getLogger(__name__)


def is_large_pdf(file_name: str, filebytes: BytesIO, page_threshold: int = 150) -> Tuple[bool, int]:
    if not file_name.lower().endswith(".pdf"):
        return False, 0
    try:
        doc = fitz.open(stream=filebytes.getvalue(), filetype="pdf")
        return len(doc) > page_threshold, len(doc)
    except Exception as e:
        logger.warning("error opening PDF - %s", e)
        # assume its not large if you can't open it
        return False, 0


def _pdf_is_image_heavy(file_bytes: BytesIO, sample_pages: int = 5, image_threshold: int = 1) -> bool:
    try:
        doc = fitz.open(stream=file_bytes.getvalue(), filetype="pdf")
        pages_to_check = min(len(doc), sample_pages)
        images_found = 0
        for i in range(pages_to_check):
            page = doc[i]
            images = page.get_images(full=True)
            if images:
                images_found += 1
        # if more than half of sampled pages have images, file == image-heavy
        return images_found >= math.ceil(pages_to_check / 2)
    except Exception as e:
        logger.debug("can't work out quantity of images for the file - %s", e)
        return False
