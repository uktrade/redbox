import asyncio
import logging
import os
import tempfile
from io import BytesIO
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from fastapi import Depends, FastAPI, File, UploadFile
from minio import Minio
from minio.error import S3Error

app = FastAPI()
logger = logging.getLogger(__name__)

USE_MINIO = os.getenv("USE_MINIO", "False").lower() == "true"
BUCKET_NAME = os.getenv("BUCKET_NAME", "redbox-storage-dev")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")
MINIO_HOST = os.getenv("MINIO_HOST", "minio")
MINIO_PORT = os.getenv("MINIO_PORT", "9000")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL", f"http://{MINIO_HOST}:{MINIO_PORT}")

if USE_MINIO:
    storage_client = Minio(
        endpoint=f"{MINIO_HOST}:{MINIO_PORT}",
        access_key=AWS_ACCESS_KEY or "minioadmin",
        secret_key=AWS_SECRET_KEY or "minioadmin",
        secure=False,
    )
else:
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    storage_client = session.client("s3", endpoint_url=AWS_S3_ENDPOINT_URL if USE_MINIO else None)

APPROVED_FILE_EXTENSIONS = [
    ".eml",
    ".html",
    ".json",
    ".md",
    ".msg",
    ".rst",
    ".rtf",
    ".txt",
    ".xml",
    ".csv",
    ".doc",
    ".docx",
    ".epub",
    ".odt",
    ".pdf",
    ".ppt",
    ".pptx",
    ".tsv",
    ".xlsx",
    ".htm",
]
MAX_FILE_SIZE = 209715200  # 200 MB or 200 * 1024 * 1024


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/process-files/")
async def process_files(files: list[UploadFile] = Depends(File(...))) -> dict:  # noqa: B008
    errors: list[str] = []
    processed_files: list[dict] = []

    for uploaded_file in files:
        file_errors, processed_file = await process_single_file(uploaded_file)
        errors.extend(file_errors)
        if processed_file:
            processed_files.append(processed_file)

    return {"errors": errors, "processed_files": processed_files}


async def process_single_file(uploaded_file: UploadFile) -> tuple[list[str], dict]:
    errors = validate_uploaded_file(uploaded_file)
    if errors:
        return errors, None

    # Handle .doc to .docx conversion
    if is_doc_file(uploaded_file):
        uploaded_file = await convert_doc_to_docx(uploaded_file)

    # Handle UTF-8 compatibility
    if not await is_utf8_compatible(uploaded_file):
        uploaded_file = await convert_to_utf8(uploaded_file)

    # Store in storage (MinIO or S3)
    minio_path = f"{uploaded_file.filename}"
    try:
        content = await uploaded_file.read()
        if USE_MINIO:
            storage_client.put_object(
                BUCKET_NAME,
                minio_path,
                BytesIO(content),
                len(content),
                content_type=uploaded_file.content_type or "application/octet-stream",
            )
        else:
            storage_client.upload_fileobj(
                Fileobj=BytesIO(content),
                Bucket=BUCKET_NAME,
                Key=minio_path,
                ExtraArgs={"ContentType": uploaded_file.content_type or "application/octet-stream"},
            )
    except (S3Error, ClientError):
        logger.exception("Error uploading to storage for file %s", uploaded_file.filename)
        errors.append("Error storing %s in storage", uploaded_file.filename)
        return errors, None

    return errors, {
        "filename": uploaded_file.filename,
        "minio_path": minio_path,
        "content_type": uploaded_file.content_type,
        "size": uploaded_file.size,
    }


def validate_uploaded_file(uploaded_file: UploadFile) -> list[str]:
    errors: list[str] = []
    if not uploaded_file.filename:
        errors.append("File has no name")
    else:
        file_extension = Path(uploaded_file.filename).suffix.lower()
        if file_extension not in APPROVED_FILE_EXTENSIONS:
            errors.append(f"Error with {uploaded_file.filename}: File type {file_extension} not supported")
    if not uploaded_file.content_type:
        errors.append(f"Error with {uploaded_file.filename}: File has no content-type")
    if uploaded_file.size > MAX_FILE_SIZE:
        errors.append(f"Error with {uploaded_file.filename}: File is larger than 200MB")
    return errors


async def is_utf8_compatible(uploaded_file: UploadFile) -> bool:
    if not Path(uploaded_file.filename).suffix.lower().endswith((".doc", ".txt")):
        logger.info("File does not require UTF-8 compatibility check")
        return True
    try:
        content = await uploaded_file.read()
        content.decode("utf-8")
        await uploaded_file.seek(0)
    except UnicodeDecodeError:
        logger.info("File is incompatible with UTF-8. Converting...")
        return False
    return True


async def convert_to_utf8(uploaded_file: UploadFile) -> UploadFile:
    try:
        content = await uploaded_file.read()
        decoded_content = content.decode("ISO-8859-1")
        new_bytes = decoded_content.encode("utf-8")
        new_file = UploadFile(
            filename=uploaded_file.filename,
            file=BytesIO(new_bytes),
            content_type="application/octet-stream",
            size=len(new_bytes),
        )
        logger.info("Conversion to UTF-8 successful")
        return new_file  # noqa: TRY300
    except Exception:
        logger.exception("Error converting file %s to UTF-8", uploaded_file.filename)
        return uploaded_file


def is_doc_file(uploaded_file: UploadFile) -> bool:
    return Path(uploaded_file.filename).suffix.lower() == ".doc"


async def convert_doc_to_docx(uploaded_file: UploadFile) -> UploadFile:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp_input:
        tmp_input.write(await uploaded_file.read())
        tmp_input.flush()
        input_path = Path(tmp_input.name)
        output_dir = input_path.parent
        temp_output_path = input_path.with_suffix(".docx")

        try:
            result = asyncio.create_subprocess_exec(
                "/usr/bin/libreoffice",
                "--headless",
                "--convert-to",
                "docx",
                str(input_path),
                "--outdir",
                str(output_dir),
            )
            logger.info("LibreOffice output: %s", result.stdout.decode("utf-8"))
            if not temp_output_path.exists():
                logger.exception("Output file not found: %s", temp_output_path)
                return uploaded_file

            with temp_output_path.open("rb") as f:
                converted_content = f.read()
                new_file = UploadFile(
                    filename=Path(uploaded_file.filename).with_suffix(".docx").name,
                    file=BytesIO(converted_content),
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    size=len(converted_content),
                )
                logger.info("Doc file conversion to docx successful for %s", uploaded_file.filename)
                return new_file
        except Exception:
            logger.exception("Error converting doc file %s to docx", uploaded_file.filename)
            return uploaded_file
        finally:
            try:
                input_path.unlink()
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning("Error cleaning up temporary files: %s", cleanup_error)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104
