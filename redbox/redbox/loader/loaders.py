import logging
import math
import re
from io import BytesIO
from typing import List, Tuple

import environ
import fitz
import pandas as pd
from pydantic import ValidationError
from redbox_app.setting_enums import Environment

from redbox.models.file import TabularSchema
from redbox.transform import bedrock_tokeniser

env = environ.Env()
ENVIRONMENT = Environment[env.str("ENVIRONMENT").upper()]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tokeniser = bedrock_tokeniser


def _to_safe_sql_identifier(value: str, fallback_prefix: str, index: int | None = None) -> str:
    """convert random names provided by users into safe SQL identifiers"""
    candidate = re.sub(r"\W+", "_", str(value).strip())
    if not candidate:
        suffix = f"_{index}" if index is not None else ""
        candidate = f"{fallback_prefix}{suffix}"
    if candidate[0].isdigit():
        candidate = f"_{candidate}"
    return candidate


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


def split_pdf(filebytes: BytesIO, pages_per_chunk: int = 75) -> List[BytesIO]:
    doc = fitz.open(stream=filebytes.getvalue(), filetype="pdf")
    chunks: List[BytesIO] = []
    total_pages = len(doc)
    if total_pages == 0:
        return chunks

    for start in range(0, total_pages, pages_per_chunk):
        sub_doc = fitz.open()
        end = min(start + pages_per_chunk, total_pages)
        sub_doc.insert_pdf(doc, from_page=start, to_page=end)
        if len(sub_doc) == 0:
            continue  # Skip empty chunks
        chunk_bytes = BytesIO(sub_doc.tobytes())
        chunks.append(chunk_bytes)
    return chunks


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


def infer_sqlite_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    return "TEXT"


def read_csv_text(file_bytes: BytesIO) -> list[dict[str, str | dict]]:
    """Reads in a csv file, validates it using pandas and then returns the csv as string with a null metadata dictionary"""
    try:
        file_bytes.seek(0)
        # Read bytes into pandas df. This acts as a pre-check that the csv is well formed
        df = pd.read_csv(file_bytes)
        if df.empty:
            logger.error("Empty File Uploaded")
            raise ValidationError("Empty File Uploaded")

        csv_text, sheet_schema = parse_tabular_schema(table_name="csv", df=df)
        return [
            {
                "text": csv_text,
                "metadata": {
                    "document_schema": sheet_schema,
                },
            }
        ]
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        logger.error(f"Error while trying to upload csv file {e}")

        try:
            file_bytes.seek(0)
            raw_text = file_bytes.read().decode("utf-8", errors="replace")
            return [
                {
                    "text": raw_text,
                    "metadata": {},
                }
            ]
        except Exception as fallback_e:
            logger.error("Fallback raw CSV parsing also failed: %s", fallback_e)
            return []


def detect_header_row(df: pd.DataFrame) -> int:
    """
    Find the first row that looks like a real table header.

    Rules:
    - Skip completely empty rows.
    - Ignore rows with fewer than 2 populated cells (likely titles).
    - Prefer rows whose populated cells are mostly strings.
    """

    first_non_empty = 0

    for idx, row in df.iterrows():
        values = row.dropna()

        if values.empty:
            continue

        first_non_empty = idx

        # "Sales Report" -> not a header
        if len(values) < 2:
            continue

        string_ratio = sum(isinstance(v, str) for v in values) / len(values)

        if string_ratio >= 0.75:
            return idx

    return first_non_empty


def normalize_excel_sheet(raw_df: pd.DataFrame) -> pd.DataFrame:
    # Remove completely empty rows/columns
    raw_df = raw_df.dropna(axis=0, how="all")
    raw_df = raw_df.dropna(axis=1, how="all")

    if raw_df.empty:
        return raw_df

    header_row = detect_header_row(raw_df)

    headers = raw_df.iloc[header_row].fillna("").astype(str).str.strip()

    headers = [h if h else f"column_{i}" for i, h in enumerate(headers)]

    df = raw_df.iloc[header_row + 1 :].reset_index(drop=True)
    df.columns = headers

    return df.convert_dtypes()


def read_excel_file(file_bytes: BytesIO) -> list[dict[str, str | dict]]:
    """Reads in an excel file, validates each sheet using pandas and then returns a list of each valid sheet as string with a null metadata dictionary"""
    try:
        sheets = pd.read_excel(file_bytes, sheet_name=None, header=None)
        elements = []

        for name, raw_df in sheets.items():
            try:
                df = normalize_excel_sheet(raw_df)

                if df.empty:
                    logger.info(f"Skipping Sheet {name}")
                    continue

                csv_text, sheet_schema = parse_tabular_schema(
                    table_name=name.lower().replace(" ", "_"),
                    df=df,
                )

                elements.append(
                    {
                        "text": csv_text,
                        "metadata": {"document_schema": sheet_schema},
                    }
                )
            except Exception as e:
                logger.info(f"Skipping Sheet {name} due to error: {e}")
                continue
        return elements if len(elements) else None
    except Exception as e:
        logger.error(f"Excel Read Error: {e}")
        try:
            file_bytes.seek(0)
            raw_text = file_bytes.read().decode("utf-8", errors="replace")
            return [{"text": raw_text, "metadata": {}}]
        except Exception as fallback_e:
            logger.error("Fallback raw Excel parsing also failed: %s", fallback_e)
            return None


def load_tabular_file(file_name: str, file_bytes: BytesIO) -> list[dict[str, str]]:
    """Selects the right read method for each file type. Returns an empty list if n"""
    if file_name.lower().endswith(".tsv"):
        file_bytes.seek(0)
        df = pd.read_csv(file_bytes, sep="\t")
        csv_text, sheet_schema = parse_tabular_schema(table_name="tsv", df=df)
        return [{"text": csv_text, "metadata": {"document_schema": sheet_schema}}]
    elif file_name.endswith(".csv"):
        elements = read_csv_text(file_bytes=file_bytes)
    else:
        elements = read_excel_file(file_bytes=file_bytes) or []

    return elements if elements else []


def parse_tabular_schema(table_name: str, df: pd.DataFrame) -> tuple[str, dict] | None:
    """Reconstruct document_schema from legacy document text at runtime."""
    # Parse CSV to get column dtypes
    try:
        safe_table_name = _to_safe_sql_identifier(table_name, "table")
        safe_columns = []
        seen: dict[str, int] = {}

        for idx, col in enumerate(df.columns):
            base = _to_safe_sql_identifier(col, "column", idx)
            count = seen.get(base, 0)
            seen[base] = count + 1
            safe_col = f"{base}_{count}" if count else base
            safe_columns.append(safe_col)

        safe_df = df.copy()
        safe_df.columns = safe_columns

        csv_text = f"<table_name>{safe_table_name}</table_name>" + safe_df.to_csv(index=False)
        sheet_schema = TabularSchema(
            name=safe_table_name,
            columns={col: infer_sqlite_type(safe_df[col].dtype) for col in safe_df.columns},
        )
        return csv_text, sheet_schema.model_dump()
    except Exception as e:
        logger.warning(f"Failed to compute schema from legacy document: {e}")
        return None
