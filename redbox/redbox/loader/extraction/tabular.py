import logging
from io import BytesIO
from redbox.models.file import TabularSchema
import pandas as pd
from pydantic import ValidationError


logger = logging.getLogger(__name__)


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


def read_excel_file(file_bytes: BytesIO) -> list[dict[str, str | dict]]:
    """Reads in an excel file, validates each sheet using pandas and then returns a list of each valid sheet as string with a null metadata dictionary"""
    try:
        sheets = pd.read_excel(file_bytes, sheet_name=None)
        elements = []

        for name, df in sheets.items():
            try:
                if df.empty:
                    logger.info(f"Skipping Sheet {name}")
                    continue

                # Include the table name in the text that is stored. This will be extracted by the retriever
                table_name = name.lower().replace(" ", "_")
                csv_text, sheet_schema = parse_tabular_schema(table_name=table_name, df=df)

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
        csv_text = f"<table_name>{table_name}</table_name>" + df.to_csv(index=False)
        sheet_schema = TabularSchema(
            name=table_name, columns={col: infer_sqlite_type(df[col].dtype) for col in df.columns}
        )
        return csv_text, sheet_schema.model_dump()
    except Exception as e:
        logger.warning(f"Failed to compute schema from legacy document: {e}")
        return None
