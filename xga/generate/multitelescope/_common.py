#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 07/07/2025, 13:03. Copyright (c) The Contributors

from typing import Union, List

import numpy as np

from xga.exceptions import TelescopeNotAssociatedError
from xga.samples import BaseSample
from xga.sources import BaseSource
from xga.sourcetools._common import _get_all_telescopes


def check_tel_associated(sources: Union[BaseSource, BaseSample], telescope: Union[str, List[str], None]) -> List[str]:
    """
    Simple check meant to be used on user-specified telescope names, to ensure they are associated with at
    least one source. If None is passed for telescope then all telescopes names associated with a source will
    be returned

    :param BaseSource/BaseSample sources: XGA source/sample to validate telescope names against, or to retrieve
        all associated telescopes for.
    :param str/List[str]/None telescope: User-specified telescope name, or list of telescope names, to check and
        see if they are associated with the source/sample.
    :return: A list of validated telescope names for use in the calling function.
    :rtype: List[str]
    """
    # Fetches out the telescope names that are associated with at least one source (this function can handle
    #  sources being either a sample or a single source).
    all_tels = _get_all_telescopes(sources)

    # Making sure that if the user has passed specific telescope(s) then the telescope argument is a list,
    #  even if there is only one name
    if isinstance(telescope, str):
        telescope = [telescope]

    # Now we perform a check on whether the input telescopes are associated with any sources
    if telescope is not None:
        good_tel = np.array([t in all_tels for t in telescope])
        if not all(good_tel):
            which_bad = ", ".join(np.array(all_tels)[~good_tel])
            raise TelescopeNotAssociatedError("Telescopes {t} are not associated with any "
                                              "source".format(t=which_bad))

    # If we get here then there are no problems with whatever the user passed in, so we choose whether we're looking
    #  at all the associated telescopes or a subset defined by the user
    rel_tels = all_tels if telescope is None else telescope

    # And return that list
    return rel_tels