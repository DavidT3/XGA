#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/11/26, 1:15 PM. Copyright (c) The Contributors.

from xga import SAS_AVAIL, SAS_VERSION
from xga.exceptions import SASNotFoundError
from xga.generate.common import mission_software_call


def sas_avail_check():
    """
    Check function for the SAS mission software.
    """
    if not SAS_AVAIL and SAS_VERSION is None:
        raise SASNotFoundError("No SAS installation has been found on this machine")
    elif not SAS_AVAIL:
        raise SASNotFoundError(
            "A SAS installation (v{}) has been found, but the SAS_CCFPATH environment variable is"
            " not set.".format(SAS_VERSION))


def sas_call(sas_func):
    """
    This is used as a decorator for functions that produce SAS command strings. Depending on the
    system that XGA is running on (and whether the user requests parallel execution), the method of
    executing the SAS command will change. This supports simple multi-threading.
    :return:
    """
    return mission_software_call('xmm', sas_avail_check)(sas_func)