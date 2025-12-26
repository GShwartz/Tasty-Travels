import logging
import logging.handlers
import queue
import os


def init_logger(log_path, name):
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        log_queue = queue.Queue()
        queue_handler = logging.handlers.QueueHandler(log_queue)

        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            if not os.path.isfile(log_path):
                open(log_path, 'w').close()  # Create an empty log file if it doesn't exist

            info = logging.FileHandler(log_path)
            info.setLevel(logging.DEBUG)
            info.setFormatter(formatter)
            logger.addHandler(info)

        return logger
    else:
        raise ValueError("Invalid log path provided.")
