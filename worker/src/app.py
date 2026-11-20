from logging import getLogger

logger = getLogger(__name__)


def load_message(file_name: str):
    print("loading message: ", file_name)
    logger.info("loading message")


def main():
    logger.info("Starting worker")


if __name__ == "__main__":
    main()
