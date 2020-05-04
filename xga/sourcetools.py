#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 04/05/2020, 12:18. Copyright (c) David J Turner

from subprocess import Popen, PIPE
from numpy import array, ndarray

from xga.utils import HeasoftError


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

    nh_vals = array([average_nh, weighed_av_nh])
    return nh_vals


# TODO Here shall live the odds and ends, stuff that doesn't fit anywhere else.
#  Some unit objects for XMM coordinate systems perhaps, angular radius calculation,
#  other things that haven't yet occured to me.


