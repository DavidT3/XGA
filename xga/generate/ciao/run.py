#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/11/26, 1:16 PM. Copyright (c) The Contributors.

from xga import CIAO_AVAIL, CALDB_AVAIL
from xga.exceptions import CIAONotFoundError, CALDBNotFoundError
from xga.generate.common import mission_software_call


def ciao_avail_check():
    """
    Check function for the CIAO mission software.
    """
    if not CIAO_AVAIL:
        raise CIAONotFoundError("No CIAO installation has been found on this machine.")

    if not CALDB_AVAIL:
        raise CALDBNotFoundError("No CALDB installation has been found on this machine.")


def ciao_call(ciao_func):
    """
    This is used as a decorator for functions that produce CIAO command strings. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the CIAO command will change.
    :return:
    """
    return mission_software_call('chandra', ciao_avail_check)(ciao_func)