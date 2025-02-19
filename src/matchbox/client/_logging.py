import logging
import sys

client_logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(levelname)s: %(message)s")
handler.setFormatter(formatter)

client_logger.addHandler(handler)
