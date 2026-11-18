#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 12:08 PM. Copyright (c) The Contributors.
"""Utility decorators for skipping unit tests when external mission software backends are unavailable."""

import unittest
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from xga import CIAO_AVAIL, ESASS_AVAIL, SAS_AVAIL

# TypeVar acts as a generic placeholder for the return type of whatever function is being decorated; binding _R in
#  Callable[..., _R] and wrapper(...) -> _R preserves the original function's return type through the decorator
#  without resorting to typing.Any (which triggers Ruff ANN401 and discards return-type information for mypy).
#  Similarly, *args: object and **kwargs: object indicate that the wrapper accepts and forwards arbitrary Python
#  objects without disabling static type checking inside the wrapper body.
_R = TypeVar("_R")


def require_sas(test_func: Callable[..., _R]) -> Callable[..., _R]:
    """Skip the decorated test if XMM-SAS is not available on the host system."""

    @wraps(test_func)
    def wrapper(*args: object, **kwargs: object) -> _R:
        if not SAS_AVAIL:
            raise unittest.SkipTest("SAS is not available.")
        return test_func(*args, **kwargs)

    return wrapper


def require_esass(test_func: Callable[..., _R]) -> Callable[..., _R]:
    """Skip the decorated test if eSASS is not available on the host system."""

    @wraps(test_func)
    def wrapper(*args: object, **kwargs: object) -> _R:
        if not ESASS_AVAIL:
            raise unittest.SkipTest("eSASS is not available.")
        return test_func(*args, **kwargs)

    return wrapper


def require_ciao(test_func: Callable[..., _R]) -> Callable[..., _R]:
    """Skip the decorated test if CIAO is not available on the host system."""

    @wraps(test_func)
    def wrapper(*args: object, **kwargs: object) -> _R:
        if not CIAO_AVAIL:
            raise unittest.SkipTest("CIAO is not available.")
        return test_func(*args, **kwargs)

    return wrapper
