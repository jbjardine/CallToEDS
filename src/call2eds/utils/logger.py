import logging
from call2eds.config.settings import settings


def setup_logger():
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    return logging.getLogger("call2eds")


logger = setup_logger()
