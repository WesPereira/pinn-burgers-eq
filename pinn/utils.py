import sys
import logging


def get_log(log_name: str = 'root') -> logging.Logger:
    """
    Sets up logging configs
    """
    new_log = logging.getLogger(log_name)
    log_format = '%(asctime)-24s %(process)-2.2s %(threadName)-8.30s '\
                 '%(levelname)-8s %(name)10s:%(lineno)-4s- %(funcName)s: '\
                 '%(message)s'

    logging.basicConfig(
        stream=sys.stdout,
        format=log_format,
        level=logging.INFO
    )

    return new_log

log = get_log()
