#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 12:08 PM. Copyright (c) The Contributors.

import unittest
from functools import wraps

from xga import SAS_AVAIL, ESASS_AVAIL, CIAO_AVAIL


def require_sas(test_func):
    """
    Decorator to skip tests if SAS is not available.
    """
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        if not SAS_AVAIL:
            raise unittest.SkipTest("SAS is not available.")
        return test_func(*args, **kwargs)
    return wrapper


def require_esass(test_func):
    """
    Decorator to skip tests if eSASS is not available.
    """
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        if not ESASS_AVAIL:
            raise unittest.SkipTest("eSASS is not available.")
        return test_func(*args, **kwargs)
    return wrapper


def require_ciao(test_func):
    """
    Decorator to skip tests if CIAO is not available.
    """
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        if not CIAO_AVAIL:
            raise unittest.SkipTest("CIAO is not available.")
        return test_func(*args, **kwargs)
    return wrapper
