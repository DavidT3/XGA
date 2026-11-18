#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/11/26, 1:15 PM. Copyright (c) The Contributors.
"""
The xga.generate.sas.run module contains the availability check and execution decorator used by all XGA wrapper
functions that call XMM-Newton's SAS software.
"""

from collections.abc import Callable
from typing import Any

from xga import SAS_AVAIL, SAS_VERSION
from xga.exceptions import SASNotFoundError
from xga.generate.common import mission_software_call


def sas_avail_check() -> None:
    """Check function for the SAS mission software."""
    if not SAS_AVAIL and SAS_VERSION is None:
        raise SASNotFoundError("No SAS installation has been found on this machine")
    elif not SAS_AVAIL:
        raise SASNotFoundError(
            f"A SAS installation (v{SAS_VERSION}) has been found, but the SAS_CCFPATH environment variable is not set."
        )


def sas_call(sas_func: Callable[..., Any]) -> Callable[..., Any]:
    """
    This is used as a decorator for functions that produce SAS command strings.
    Depending on the system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the SAS command will change. This supports simple multi-threading.

    :param Callable sas_func: The SAS wrapper function being decorated.
    :return: The wrapped SAS function.
    :rtype: Callable
    """
    return mission_software_call("xmm", sas_avail_check)(sas_func)
