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


def is_utf8_compatible(uploaded_file: UploadedFile) -> bool:
    if not Path(uploaded_file.unique_name).suffix.lower().endswith((".doc", ".txt")):
        logging.info("File does not require utf8 compatibility check")
        return True
    try:
        uploaded_file.open()
        uploaded_file.read().decode("utf-8")
        uploaded_file.seek(0)
    except UnicodeDecodeError:
        logging.info("File is incompatible with utf-8. Converting...")
        return False
    else:
        logging.info("File is compatible with utf-8 - ready for processing")
        return True


def convert_to_utf8(uploaded_file: UploadedFile) -> UploadedFile:
    try:
        uploaded_file.open()
        content = uploaded_file.read().decode("ISO-8859-1")

        # Detect and replace non-UTF-8 characters
        new_bytes = content.encode("utf-8")

        # Creating a new InMemoryUploadedFile object with the converted content
        new_uploaded_file = InMemoryUploadedFile(
            file=BytesIO(new_bytes),
            field_name=uploaded_file.unique_name,
            name=uploaded_file.unique_name,
            content_type="application/octet-stream",
            size=len(new_bytes),
            charset="utf-8",
        )
    except Exception as e:
        logging.exception("Error converting file %s to UTF-8.", uploaded_file, exc_info=e)
        return uploaded_file
    else:
        logging.info("Conversion to UTF-8 successful")
        return new_uploaded_file


def is_doc_file(uploaded_file: UploadedFile) -> bool:
    return Path(uploaded_file.unique_name).suffix.lower() == ".doc"


def convert_doc_to_docx(uploaded_file: UploadedFile) -> UploadedFile:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as tmp_input:
        tmp_input.write(uploaded_file.read())
        tmp_input.flush()
        input_path = Path(tmp_input.unique_name)
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
            logging.info("LibreOffice output: %s", result.stdout.decode())
            logging.info("LibreOffice errors: %s", result.stderr.decode())

            if not temp_output_path.exists():
                logging.error("Output file not found: %s", temp_output_path)
                return uploaded_file

            logging.info("Output path: %s", temp_output_path)

            time.sleep(1)
            with temp_output_path.open("rb") as f:
                converted_content = f.read()
                logging.info("Converted file size: %d bytes", len(converted_content))
                if len(converted_content) == 0:
                    logging.error("Converted file is empty - this won't get converted")

                output_filename = Path(uploaded_file.unique_name).with_suffix(".docx").name
                new_file = InMemoryUploadedFile(
                    file=BytesIO(converted_content),
                    field_name=uploaded_file.unique_name,
                    name=output_filename,
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    size=len(converted_content),
                    charset="utf-8",
                )
                logging.info("doc file conversion to docx successful for %s", uploaded_file.unique_name)
        except Exception as e:
            logging.exception("Error converting doc file %s to docx", uploaded_file.unique_name, exc_info=e)
            new_file = uploaded_file
        finally:
            try:
                input_path.unlink()
                if temp_output_path.exists():
                    temp_output_path.unlink()
            except Exception as cleanup_error:  # noqa: BLE001
                logging.warning("Error cleaning up temporary files: %s", cleanup_error)

        return new_file


def ingest(file_id: UUID, es_index: str | None = None) -> None:
    # These models need to be loaded at runtime otherwise they can be loaded before they exist
    from redbox_app.redbox_core.models import File

    if not es_index:
        es_index = env.elastic_chunk_alias

    file = File.objects.get(id=file_id)

    # handling doc -> docx conversion
    if is_doc_file(file):
        file = convert_doc_to_docx(file)
    # handling utf8 compatibility
    if not is_utf8_compatible(file):
        file = convert_to_utf8(file)

    file.save()

    logging.info("Ingesting file: %s", file)

    if error := ingest_file(file.unique_name, es_index):
        file.status = File.Status.errored
        file.ingest_error = error
    else:
        file.status = File.Status.complete

    file.save()
