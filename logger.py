
from datetime import datetime
import logging
from os import path

def filename_date(name):
    date = datetime.now().strftime(format='%Y-%m-%d')
    return path.join("log",f"{date}_{name}.log")

def get_logger(name,
               file_dst = None,
               stdout = True):

    logger = logging.getLogger(name=name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if file_dst :
            file_handler = logging.FileHandler(filename=file_dst)
            formatter2 = logging.Formatter(fmt='%(asctime)s | %(funcName)s | %(message)s')
            file_handler.setFormatter(formatter2)
            logger.addHandler(file_handler)

        if stdout:
            formatter3 = logging.Formatter(fmt='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter3)
            logger.addHandler(stream_handler)

    return logger
