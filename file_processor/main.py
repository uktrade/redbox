import asyncio

import django

django.setup()
import logging
import tempfile
from collections.abc import MutableSequence, Sequence
from io import BytesIO
from pathlib import Path

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import InMemoryUploadedFile
from django_q.tasks import async_task
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from starlette.responses import JSONResponse

app = FastAPI()
logger = logging.getLogger(__name__)
User = get_user_model()

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
MAX_FILE_SIZE = 209715200  # 200 MB

from asgiref.sync import sync_to_async

from redbox_app.redbox_core.models import File
from redbox_app.worker import ingest


@sync_to_async
def get_user(x_user_id: str = Header(...)):
    """Dependency to fetch user from X-User-ID header."""
    try:
        user = User.objects.get(id=f"{x_user_id}")
        return user
    except User.DoesNotExist:
        raise HTTPException(status_code=401, detail="Invalid user ID")
    except Exception:
        logger.exception("Error fetching user with ID %s", x_user_id)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/process-files/")
async def upload_files(files: list[UploadFile] = File(...), user=Depends(get_user)):
    errors: MutableSequence[str] = []
    processed_files: MutableSequence[dict] = []

    if not files:
        errors.append("No document selected")

    uploaded_files: MutableSequence[UploadFile] = []
    for uploaded_file in files:
        file_errors = validate_uploaded_file(uploaded_file)
        errors.extend(file_errors)
        if file_errors:
            continue

        if is_doc_file(uploaded_file):
            uploaded_file = await convert_doc_to_docx(uploaded_file)

        if not await is_utf8_compatible(uploaded_file):
            uploaded_file = await convert_to_utf8(uploaded_file)

        uploaded_files.append(uploaded_file)

    if not errors:
        ingest_errors: MutableSequence[str] = []
        for uploaded_file in uploaded_files:
            file_obj, store_errors = await store_and_create_file(uploaded_file, user)
            if store_errors:
                errors.extend(store_errors)
                continue

            processed_files.append(
                {
                    "filename": uploaded_file.filename,
                    "minio_path": file_obj.minio_path,
                }
            )

            async_task(ingest, file_obj.id, task_name=file_obj.unique_name, group="ingest")
            ingest_errors.extend(getattr(file_obj, "ingest_error", []) or [])

        if ingest_errors:
            errors.extend(ingest_errors)

    return JSONResponse(
        status_code=400 if errors else 200,
        content={
            "errors": errors,
            "processed_files": processed_files,
            "uploaded": not errors,
        },
    )


async def store_and_create_file(uploaded_file: UploadFile, user):
    errors: MutableSequence[str] = []
    minio_path = str(uploaded_file.filename)

    try:
        content = await uploaded_file.read()
        logger.debug("Processing file %s", uploaded_file.filename)

        django_file = InMemoryUploadedFile(
            file=BytesIO(content),
            field_name=uploaded_file.filename,
            name=uploaded_file.filename,
            content_type=uploaded_file.content_type or "application/octet-stream",
            size=len(content),
            charset="utf-8",
        )

        file_obj = await sync_to_async(File.objects.create)(
            status=File.Status.processing.value,
            user=user,
            original_file=django_file,
            minio_path=minio_path,
        )
        logger.info("Successfully created File object for %s with minio_path %s", uploaded_file.filename, minio_path)

    except Exception as e:
        logger.exception("Unexpected error processing %s: %s", uploaded_file.filename, e)
        errors.append(f"Error processing {uploaded_file.filename}: {e!s}")
        return None, errors

    return file_obj, errors


def validate_uploaded_file(uploaded_file: UploadFile) -> Sequence[str]:
    errors: MutableSequence[str] = []
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
        await uploaded_file.seek(0)  # Reset file pointer
        return True
    except UnicodeDecodeError:
        logger.info("File is incompatible with UTF-8. Converting...")
        return False
    except Exception as e:
        logger.exception("Error checking UTF-8 compatibility for %s: %s", uploaded_file.filename, e)
        return False


async def convert_to_utf8(uploaded_file: UploadFile) -> UploadFile:
    try:
        content = await uploaded_file.read()
        decoded_content = content.decode("ISO-8859-1")
        new_bytes = decoded_content.encode("utf-8")
        return UploadFile(
            filename=uploaded_file.filename,
            file=BytesIO(new_bytes),
            content_type="application/octet-stream",
        )
    except Exception as e:
        logger.exception("Error converting file %s to UTF-8: %s", uploaded_file.filename, e)
        return uploaded_file


def is_doc_file(uploaded_file: UploadFile) -> bool:
    return Path(uploaded_file.filename).suffix.lower() == ".doc"


async def convert_doc_to_docx(uploaded_file: UploadFile) -> UploadFile:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp_input:
        content = await uploaded_file.read()
        tmp_input.write(content)
        tmp_input.flush()
        input_path = Path(tmp_input.name)
        output_dir = input_path.parent
        temp_output_path = input_path.with_suffix(".docx")

        try:
            process = await asyncio.create_subprocess_exec(
                "/usr/bin/libreoffice",
                "--headless",
                "--convert-to",
                "docx",
                str(input_path),
                "--outdir",
                str(output_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(output_dir),
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                logger.error("LibreOffice conversion failed: %s", stderr.decode())
                return uploaded_file

            logger.info("LibreOffice output: %s", stdout.decode())

            if not temp_output_path.exists():
                logger.error("Output file not found: %s", temp_output_path)
                return uploaded_file

            with temp_output_path.open("rb") as f:
                converted_content = f.read()
                if len(converted_content) == 0:
                    logger.error("Converted file is empty")
                    return uploaded_file

                return UploadFile(
                    filename=Path(uploaded_file.filename).with_suffix(".docx").name,
                    file=BytesIO(converted_content),
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
        except Exception as e:
            logger.exception("Error converting doc file %s to docx: %s", uploaded_file.filename, e)
            return uploaded_file
        finally:
            try:
                input_path.unlink(missing_ok=True)
                if temp_output_path.exists():
                    temp_output_path.unlink(missing_ok=True)
            except Exception as e:
                logger.warning("Error cleaning up temporary files: %s", e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)  # noqa: S104
