"""Rank-aware logging helper (DDP)."""

import inspect
import logging
import os


def log_for_0(msg, *args, level=logging.INFO):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    caller = inspect.currentframe().f_back.f_globals.get("__name__", __name__)
    logging.getLogger(caller).log(level, msg, *args)
