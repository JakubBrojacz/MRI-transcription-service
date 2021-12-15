import functools
import time
import logging

import config


time_logger = logging.getLogger("Time_Logger")


def timeit(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = f(*args, **kwargs)
        t2 = time.time()
        time_logger.info(f'function:{f.__name__}, args:[{args}, {kwargs}] took: {t2-t1} sec')
        return result
    return wrapper
