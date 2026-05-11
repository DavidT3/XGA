#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/11/26, 1:16 PM. Copyright (c) The Contributors.

from xga import ESASS_AVAIL
from xga.exceptions import eSASSNotFoundError
from xga.generate.common import mission_software_call


def esass_avail_check():
    """
    Check function for the eSASS mission software.
    """
    if not ESASS_AVAIL:
        raise eSASSNotFoundError("No eSASS installation has been found on this machine.")


def esass_call(esass_func):
    """
    This is used as a decorator for functions that produce eSASS command strings. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the eSASS command will change. This supports both simple multi-threading.
    :return:
    """
    return mission_software_call('erosita', esass_avail_check)(esass_func)