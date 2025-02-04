""" CEMA: Causal Explanations for Decision Making in Multi-Agent Systems. """
import os
import logging
import datetime

from cema import xavi, oxavi, llm


def setup_cema_logging(log_dir: str = None, log_name: str = None):
    """ Setup the logging configuration for the CEMA application

    Args:
        log_dir: The directory to save the log files in.
        log_name: The name of the log file.
    """
    # Add %(asctime)s  for time
    log_formatter = logging.Formatter(
        "[%(threadName)-10.10s:%(name)-20.20s] [%(levelname)-6.6s]  %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    logging.getLogger("igp2.core.velocitysmoother").setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.INFO)
    if log_dir and log_name:
        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        file_handler = logging.FileHandler(f"{log_dir}/{log_name}_{date_time}.log")
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)