#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 04/05/2020, 12:18. Copyright (c) David J Turner

from subprocess import Popen, PIPE
from numpy import array, ndarray

from xga.exceptions import HeasoftError
from xga import CENSUS
from pandas import DataFrame


def nhlookup(ra: float, dec: float) -> ndarray:
    """
    Uses HEASOFT to lookup hydrogen column density for given coordinates.
    :param float ra: Right Ascension of object
    :param float dec: Declination of object
    :return : Average and weighted average nH values (in units of cm^-2)
    :rtype: ndarray
    """
    # Apparently minimal type-checking is the Python way, but for some reason this heasoft command fails if
    # integers are passed, so I'll convert them, let them ValueError if people pass weird types.
    ra = float(ra)
    dec = float(dec)

    heasoft_cmd = 'nh 2000 {ra} {dec}'.format(ra=ra, dec=dec)

    out, err = Popen(heasoft_cmd, stdout=PIPE, stderr=PIPE, shell=True).communicate()
    # Catch errors from stderr
    if err.decode("UTF-8") != '':
        # Going to assume top line of error most important, and strip out the error type from the string
        msg = err.decode("UTF-8").split('\n')[0].split(':')[-1].strip(' ')
        print(out.decode("UTF-8"))  # Sometimes this also has useful information
        raise HeasoftError(msg)

    heasoft_output = out.decode("utf-8")
    lines = heasoft_output.split('\n')
    try:
        average_nh = lines[-3].split(' ')[-1]
        weighed_av_nh = lines[-2].split(' ')[-1]
    except IndexError:
        raise HeasoftError("HEASOFT nH command output is not as expected")

    try:
        average_nh = float(average_nh)
        weighed_av_nh = float(weighed_av_nh)
    except ValueError:
        raise HeasoftError("HEASOFT nH command scraped output cannot be converted to float")

    # Returns both the average and weighted average nH values, as output by HEASOFT nH tool.
    nh_vals = array([average_nh, weighed_av_nh])
    return nh_vals


# TODO Maybe switch this over to taking a source object as an argument instead of coords
def simple_xmm_match(src_ra: float, src_dec: float, half_width: float = 15.0) -> DataFrame:
    """
    Returns ObsIDs within a square of +-half width from the input ra and dec. The default half_width is
    15 arcminutes, which approximately corresponds to the size of the XMM FOV.
    :param float src_ra: RA coordinate of the source, in degrees.
    :param float src_dec: DEC coordinate of the source, in degrees.
    :param float half_width: Half width of square to search in, in arc minutes.
    :return: The ObsID, RA_PNT, and DEC_PNT of matching XMM observations.
    :rtype: DataFrame
    """
    half_width = half_width / 60
    matches = CENSUS[(CENSUS["RA_PNT"] <= src_ra+half_width) & (CENSUS["RA_PNT"] >= src_ra-half_width) &
                     (CENSUS["DEC_PNT"] <= src_dec+half_width) & (CENSUS["DEC_PNT"] >= src_dec-half_width)]
    return matches


# TODO Want this to do a check as to where it falls wrt chip gaps ideally
def full_xmm_match():
    raise NotImplemented("More complex XMM matching is not implemented yet.")

# TODO Some unit objects for XMM coordinate systems perhaps, angular radius calculation,
#  other things that haven't yet occured to me.


