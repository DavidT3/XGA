#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 21/08/2020, 13:15. Copyright (c) David J Turner

from astropy.units.quantity import Quantity
from pandas import DataFrame

from xga import CENSUS
from xga.exceptions import NoMatchFoundError


def simple_xmm_match(src_ra: float, src_dec: float, half_width: Quantity = Quantity(20.0, 'arcmin')) -> DataFrame:
    """
    Returns ObsIDs within a square of +-half width from the input ra and dec. The default half_width is
    15 arcminutes, which approximately corresponds to the size of the XMM FOV.
    :param float src_ra: RA coordinate of the source, in degrees.
    :param float src_dec: DEC coordinate of the source, in degrees.
    :param Quantity half_width: Half width of square to search in..
    :return: The ObsID, RA_PNT, and DEC_PNT of matching XMM observations.
    :rtype: DataFrame
    """
    hw = half_width.to('deg').value
    matches = CENSUS[(CENSUS["RA_PNT"] <= src_ra+hw) & (CENSUS["RA_PNT"] >= src_ra-hw) &
                     (CENSUS["DEC_PNT"] <= src_dec+hw) & (CENSUS["DEC_PNT"] >= src_dec-hw)]
    if len(matches) == 0:
        raise NoMatchFoundError("No XMM observation found within {a} of ra={r} "
                                "dec={d}".format(r=round(src_ra, 4), d=round(src_dec, 4), a=half_width))
    return matches


# TODO Want this to do a check as to where it falls wrt chip gaps ideally
def full_xmm_match():
    raise NotImplemented("More complex XMM matching is not implemented yet.")