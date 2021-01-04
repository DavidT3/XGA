#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 04/01/2021, 20:04. Copyright (c) David J Turner

import numpy as np
from astropy.units.quantity import Quantity
from pandas import DataFrame

from .. import CENSUS, BLACKLIST
from ..exceptions import NoMatchFoundError


def simple_xmm_match(src_ra: float, src_dec: float, distance: Quantity = Quantity(30.0, 'arcmin')) -> DataFrame:
    """
    Returns ObsIDs within a given distance from the input ra and dec values.

    :param float src_ra: RA coordinate of the source, in degrees.
    :param float src_dec: DEC coordinate of the source, in degrees.
    :param Quantity distance: The distance to search for XMM observations within, default should be
        able to match a source on the edge of an observation to the centre of the observation.
    :return: The ObsID, RA_PNT, and DEC_PNT of matching XMM observations.
    :rtype: DataFrame
    """
    rad = distance.to('deg').value
    local_census = CENSUS.copy()
    local_census["dist"] = np.sqrt((local_census["RA_PNT"] - src_ra)**2
                                   + (local_census["DEC_PNT"] - src_dec)**2)
    matches = local_census[local_census["dist"] <= rad]
    matches = matches[~matches["ObsID"].isin(BLACKLIST["ObsID"])]
    if len(matches) == 0:
        raise NoMatchFoundError("No XMM observation found within {a} of ra={r} "
                                "dec={d}".format(r=round(src_ra, 4), d=round(src_dec, 4), a=distance))
    return matches
