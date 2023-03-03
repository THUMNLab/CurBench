import os
import logging


def create_log_dir(log_name):
    log_root = 'runs'
    log_dir = os.path.join(log_root, log_name)
    if not os.path.exists(log_root): os.mkdir(log_root)
    if not os.path.exists(log_dir): os.mkdir(log_dir)
    return log_dir


def get_logger(log_file, log_name=None):
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        fmt='%(asctime)s\t%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    fh = logging.FileHandler(log_file)
    fh.setFormatter(log_format)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)

    return logger