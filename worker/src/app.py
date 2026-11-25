from logging import getLogger

from django_q.tasks import async_task

logger = getLogger(__name__)


def load_message(file_name: str):
    print("loading message: ", file_name)
    logger.info("loading message")


def create_message():
    logger.info("creating message")
    async_task(
        "src.worker.ingest",
        "",
        task_name="test_file.pdf",
        group="re-ingest",
    )


def main():
    logger.info("Starting worker")


if __name__ == "__main__":
    main()
