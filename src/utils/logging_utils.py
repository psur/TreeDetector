import logging
def configure_logging(verbose=False):logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,format="%(levelname)s | %(message)s")
