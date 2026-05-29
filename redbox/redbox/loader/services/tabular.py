import logging
import pandas as pd
from io import BytesIO
from redbox.models.file import TabularSchema

logger = logging.getLogger(__name__)


def infer_sqlite_type(dtype) -> str:
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    return "TEXT"


def parse_tabular_schema(table_name: str, df: pd.DataFrame) -> tuple[str, dict]:
    csv_text = f"<table_name>{table_name}</table_name>" + df.to_csv(index=False)
    sheet_schema = TabularSchema(name=table_name, columns={col: infer_sqlite_type(df[col].dtype) for col in df.columns})
    return csv_text, sheet_schema.model_dump()


def read_csv_text(file_bytes: BytesIO) -> list[dict]:
    file_bytes.seek(0)
    df = pd.read_csv(file_bytes)
    if df.empty:
        raise ValueError("Empty CSV")
    csv_text, schema = parse_tabular_schema("csv", df)
    return [{"text": csv_text, "metadata": {"document_schema": schema}}]


def read_excel_file(file_bytes: BytesIO) -> list[dict]:
    sheets = pd.read_excel(file_bytes, sheet_name=None)
    elements = []
    for name, df in sheets.items():
        if df.empty:
            continue
        table_name = name.lower().replace(" ", "_")
        csv_text, schema = parse_tabular_schema(table_name, df)
        elements.append({"text": csv_text, "metadata": {"document_schema": schema}})
    return elements


def load_tabular_file(file_name: str, file_bytes: BytesIO) -> list[dict]:
    try:
        if file_name.lower().endswith(".tsv"):
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes, sep="\t")
            csv_text, schema = parse_tabular_schema("tsv", df)
            return [{"text": csv_text, "metadata": {"document_schema": schema}}]
        elif file_name.lower().endswith(".csv"):
            return read_csv_text(file_bytes)
        else:  # excel
            return read_excel_file(file_bytes) or []
    except Exception as e:
        logger.error("Tabular parsing failed for %s: %s", file_name, e)
        # fallback to raw text
        file_bytes.seek(0)
        raw = file_bytes.read().decode("utf-8", errors="replace")
        return [{"text": raw, "metadata": {}}]
