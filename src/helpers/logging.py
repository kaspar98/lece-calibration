import logging
import logging.config
import traceback
from pathlib import Path
from src.CONFIG import LOGS_DIR
from src.helpers.load_save_data import get_path_from_project_root


class IndentedAdapter(logging.LoggerAdapter):
    @staticmethod
    def indent():
        calls_from_src_module = [item for item in traceback.extract_stack() if "src" in item.filename]
        n_calls_from_src = len(calls_from_src_module)
        return n_calls_from_src - 3  # remove logging infrastructure calls and first call

    def process(self, msg, kwargs):
        return '\t' * self.indent() + str(msg), kwargs


def get_logger(name):
    return IndentedAdapter(logging.getLogger(name), {})


def setup_logging(name):
    logging.basicConfig(format="[%(asctime)s] - %(message)s",
                        level=logging.INFO,
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(Path(get_path_from_project_root(), LOGS_DIR, name + ".log"))])
