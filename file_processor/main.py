import logging
import os
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from fastapi import FastAPI, File, UploadFile
from minio import Minio
from minio.error import S3Error

app = FastAPI()
logger = logging.getLogger(__name__)

minio_client = Minio(
    endpoint="minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

USES_MINIO = os.getenv("ENVIRONMENT", "LOCAL") == "LOCAL"
BUCKET_NAME = os.getenv("BUCKET_NAME", "redbox-storage-dev")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-2")
AWS_S3_ENDPOINT_URL = os.getenv("AWS_S3_ENDPOINT_URL")

if not USES_MINIO:
    session = boto3.Session(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    storage_client = session.client("s3", endpoint_url=AWS_S3_ENDPOINT_URL if AWS_S3_ENDPOINT_URL else None)
else:
    storage_client = minio_client

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
async def process_files(files: list[UploadFile] = File(...)) -> dict:
    errors: list[str] = []
    processed_files: list[dict] = []

    if USES_MINIO:
        try:
            buckets = storage_client.list_buckets()
        except S3Error as e:
            logger.error(f"Error connecting to MinIO: {e}")
            errors.append("Cannot connect to MinIO server")
            return {"errors": errors, "processed_files": []}

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
        logger.debug(f"Uploading file {uploaded_file.filename} to bucket {BUCKET_NAME}")
        if USES_MINIO:
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
        logger.info(f"Successfully uploaded {uploaded_file.filename} to {BUCKET_NAME}/{minio_path}")
    except (S3Error, ClientError) as e:
        logger.error(f"Error uploading to storage: {e}", exc_info=True)
        errors.append(f"Error storing {uploaded_file.filename}: {e!s}")
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
        return new_file
    except Exception as e:
        logger.exception(f"Error converting file {uploaded_file.filename} to UTF-8: {e}")
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
            result = subprocess.run(
                [
                    "/usr/bin/libreoffice",
                    "--headless",
                    "--convert-to",
                    "docx",
                    str(input_path),
                    "--outdir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                cwd=output_dir,
            )
            logger.info(f"LibreOffice output: {result.stdout.decode()}")
            if not temp_output_path.exists():
                logger.error(f"Output file not found: {temp_output_path}")
                return uploaded_file

            with temp_output_path.open("rb") as f:
                converted_content = f.read()
                new_file = UploadFile(
                    filename=Path(uploaded_file.filename).with_suffix(".docx").name,
                    file=BytesIO(converted_content),
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    size=len(converted_content),
                )
                logger.info(f"Doc file conversion to docx successful for {uploaded_file.filename}")
                return new_file
        except Exception as e:
            logger.exception(f"Error converting doc file {uploaded_file.filename} to docx: {e}")
            return uploaded_file
        finally:
            try:
                input_path.unlink()
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception as cleanup_error:
                logger.warning(f"Error cleaning up temporary files: {cleanup_error}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
