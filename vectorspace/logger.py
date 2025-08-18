import logging
import os


def setup_logger(log_path: str | None = None):
    if log_path is None:
        log_path = os.getenv("HOME") + "/.cache/vectorspace"
    os.makedirs(log_path, exist_ok=True)

    logging.basicConfig(
        filename=log_path + "/vectorspace.log",
        filemode="a",
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
    )
