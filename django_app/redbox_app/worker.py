import logging
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path
from uuid import UUID

from django.core.files.uploadedfile import InMemoryUploadedFile, UploadedFile

from redbox.loader.ingester import ingest_file
from redbox.models.settings import get_settings

env = get_settings()

logger = logging.getLogger(__name__)


def get_file_name(uploaded_file) -> str:
    """Return the correct name field for both model and file objects."""
    return getattr(uploaded_file, "unique_name", getattr(uploaded_file, "name", ""))


def is_utf8_compatible(uploaded_file) -> bool:
    if not Path(get_file_name(uploaded_file)).suffix.lower().endswith((".doc", ".txt")):
        logger.info("File does not require utf8 compatibility check")
        return True

    # Determine the file-like object to read from
    file_obj = uploaded_file.original_file if hasattr(uploaded_file, "unique_name") else uploaded_file

    try:
        content = file_obj.read()
        content.decode("utf-8")
        file_obj.seek(0)
    except UnicodeDecodeError:
        logger.info("File is incompatible with utf-8. Converting...")
        file_obj.seek(0)
        return False
    else:
        logger.info("File is compatible with utf-8 - ready for processing")
        return True


def convert_to_utf8(uploaded_file: UploadedFile) -> UploadedFile:
    # Determine the file-like object to read from
    file_obj = uploaded_file.original_file if hasattr(uploaded_file, "unique_name") else uploaded_file

    try:
        content = file_obj.read().decode("ISO-8859-1")
        new_bytes = content.encode("utf-8")
        # Creating a new InMemoryUploadedFile object with the converted content
        new_uploaded_file = InMemoryUploadedFile(
            file=BytesIO(new_bytes),
            field_name=get_file_name(uploaded_file),
            name=get_file_name(uploaded_file),
            content_type="application/octet-stream",
            size=len(new_bytes),
            charset="utf-8",
        )
    except Exception as e:
        logger.exception("Error converting file %s to UTF-8.", uploaded_file, exc_info=e)
        file_obj.seek(0)
        return uploaded_file
    else:
        logger.info("Conversion to UTF-8 successful")
        return new_uploaded_file


def is_doc_file(uploaded_file: UploadedFile) -> bool:
    return Path(get_file_name(uploaded_file)).suffix.lower() == ".doc"


def convert_doc_to_docx(uploaded_file: UploadedFile) -> UploadedFile:
    # Determine the file-like object to read from
    file_obj = uploaded_file.original_file if hasattr(uploaded_file, "unique_name") else uploaded_file

    content = file_obj.read()
    file_obj.seek(0)

    new_file = uploaded_file  # Default to original

    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp_input:
        tmp_input.write(content)
        tmp_input.flush()
        input_path = Path(tmp_input.name)
        output_dir = input_path.parent

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        temp_output_path = input_path.with_suffix(".docx")

        try:
            result = subprocess.run(  # noqa: S603
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
            logger.info("LibreOffice output: %s", result.stdout.decode())
            logger.info("LibreOffice errors: %s", result.stderr.decode())

            if not temp_output_path.exists():
                logger.error("Output file not found: %s", temp_output_path)
                return uploaded_file

            logger.info("Output path: %s", temp_output_path)

            time.sleep(1)
            with temp_output_path.open("rb") as f:
                converted_content = f.read()
                logger.info("Converted file size: %d bytes", len(converted_content))
                if len(converted_content) == 0:
                    logger.error("Converted file is empty - this won't get converted")
                    return uploaded_file

                output_filename = Path(get_file_name(uploaded_file)).with_suffix(".docx").name
                bytes_io = BytesIO(converted_content)
                bytes_io.seek(0)
                new_file = InMemoryUploadedFile(
                    file=bytes_io,
                    field_name=get_file_name(uploaded_file),
                    name=output_filename,
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    size=len(converted_content),
                    charset="utf-8",
                )

                logger.info("doc file conversion to docx successful for %s", get_file_name(uploaded_file))
        except Exception as e:
            logger.exception("Error converting doc file %s to docx", get_file_name(uploaded_file), exc_info=e)
        finally:
            try:
                input_path.unlink()
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception as cleanup_error:  # noqa: BLE001
                logger.warning("Error cleaning up temporary files: %s", cleanup_error)

    return new_file


def ingest(file_id: UUID, es_index: str | None = None) -> None:
    # These models need to be loaded at runtime otherwise they can be loaded before they exist
    from redbox_app.redbox_core.models import File  # noqa: PLC0415

    if not es_index:
        es_index = env.elastic_chunk_alias

    file = File.objects.get(id=file_id)

    # handling doc -> docx conversion
    if is_doc_file(file):
        file.original_file = convert_doc_to_docx(file)
        file.original_file.file.seek(0)
        file.save()
    # handling utf8 compatibility
    if not is_utf8_compatible(file):
        file.original_file = convert_to_utf8(file)
        file.original_file.file.seek(0)
        file.save()

    logger.info("Ingesting file: %s", file)

    if error := ingest_file(file.unique_name, es_index):
        file.status = File.Status.errored
        file.ingest_error = error
    else:
        file.status = File.Status.complete

    file.save()
