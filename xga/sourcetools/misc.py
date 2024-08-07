#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 14/04/2023, 09:06. Copyright (c) The Contributors
import warnings
from copy import deepcopy
from subprocess import Popen, PIPE
from typing import Union, List

from astropy.coordinates import SkyCoord
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from numpy import array, ndarray, pi

from .. import DEFAULT_COSMO
from ..exceptions import HeasoftError
from ..models import BaseModel1D


def nh_lookup(coord_pair: Quantity) -> ndarray:
    """
    Uses HEASOFT to lookup hydrogen column density for given coordinates.

    :param Quantity coord_pair: An astropy quantity with RA and DEC of interest.
    :return: Average and weighted average nH values (in units of cm$^{-2}$)
    :rtype: ndarray
    """
    # Apparently minimal type-checking is the Python way, but for some reason this heasoft command fails if
    # integers are passed, so I'll convert them, let them TypeError if people pass weird types.
    pos_deg = coord_pair.to("deg")
    src_ra = float(pos_deg.value[0])
    src_dec = float(pos_deg.value[1])

    heasoft_cmd = 'nh 2000 {ra} {dec}'.format(ra=src_ra, dec=src_dec)

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
        nh_vals = Quantity(array([average_nh, weighed_av_nh]) / 10 ** 22, "10^22 cm^-2")
    except ValueError:
        if any(["nH is from the closest pixel to the input position" in line for line in lines]):
            dist = [e for e in lines[12].split(' ') if e != ''][2]
            warnings.warn("nH is from the closest pixel to the input position, that is {d} "
                          "degrees away. Both returned nH values will be the same.".format(d=dist))
            try:
                nh_val = float([e for e in lines[12].split(' ') if e != ''][3])
                nh_vals = Quantity(array([nh_val, nh_val]) / 10 ** 22, "10^22 cm^-2")
            except ValueError:
                raise HeasoftError("HEASOFT nH command scraped output cannot be converted to float")
        else:
            raise HeasoftError("HEASOFT nH command scraped output cannot be converted to float")
    # Returns both the average and weighted average nH values, as output by HEASOFT nH tool.
    return nh_vals


def rad_to_ang(rad: Quantity, z: float, cosmo: Cosmology = DEFAULT_COSMO) -> Quantity:
    """
    Converts radius in length units to radius on sky in degrees.

    :param Quantity rad: Radius for conversion.
    :param Cosmology cosmo: An instance of an astropy cosmology, the default is a flat LambdaCDM concordance model.
    :param float z: The _redshift of the source.
    :return: The radius in degrees.
    :rtype: Quantity
    """
    d_a = cosmo.angular_diameter_distance(z)
    ang_rad = (rad.to("Mpc") / d_a).to('').value * (180 / pi)
    return Quantity(ang_rad, 'deg')


def ang_to_rad(ang: Quantity, z: float, cosmo: Cosmology = DEFAULT_COSMO) -> Quantity:
    """
    The counterpart to rad_to_ang, this converts from an angle to a radius in kpc.

    :param Quantity ang: Angle to be converted to radius.
    :param Cosmology cosmo: An instance of an astropy cosmology, the default is a flat LambdaCDM concordance model.
    :param float z: The _redshift of the source.
    :return: The radius in kpc.
    :rtype: Quantity
    """
    d_a = cosmo.angular_diameter_distance(z)
    rad = (ang.to("deg").value * (pi / 180) * d_a).to("kpc")
    return rad


def name_to_coord(name: str):
    """
    I'd like it to be known that I hate papers and resources that either only give the name of an object
    or its sexagesimal coordinates - however it happens upsettingly often so here we are. This function will
    take a standard format name (e.g. XMMXCS J041853.9+555333.7) and return RA and DEC in degrees.

    :param name:
    """
    raise NotImplementedError("I started this and will finish it at some point, but I got bored.")
    if " " in name:
        survey, coord_str = name.split(" ")
        coord_str = coord_str[1:]
    elif "J" in name:
        survey, coord_str = name.sdplit("J")
    else:
        num_search = [d.isdigit() for d in name].index(True)
        survey = name[:num_search]
        coord_str = name[num_search:]

    if "+" in coord_str:
        ra, dec = coord_str.split("+")
    elif "-" in coord_str:
        ra, dec = coord_str.split("-")
        dec = "-" + dec
    else:
        raise ValueError("There doesn't seem to be a + or - in the object name.")


def coord_to_name(coord_pair: Quantity, survey: str = None) -> str:
    """
    This was originally just written in the init of BaseSource, but I figured I should split it out
    into its own function really. This will take a coordinate pair, and optional survey name, and spit
    out an object name in the standard format.

    :return: Source name based on coordinates.
    :rtype: str
    """
    s = SkyCoord(ra=coord_pair[0], dec=coord_pair[1])
    crd_str = s.to_string("hmsdms").replace("h", "").replace("m", "").replace("s", "").replace("d", "")
    ra_str, dec_str = crd_str.split(" ")
    # A bug popped up where a conversion ended up with no decimal point and the return part got
    #  really upset - so this adds one if there isn't one
    if "." not in ra_str:
        ra_str += "."
    if "." not in dec_str:
        dec_str += "."

    if survey is None:
        name = "J" + ra_str[:ra_str.index(".") + 2] + dec_str[:dec_str.index(".") + 2]
    else:
        name = survey + "J" + ra_str[:ra_str.index(".") + 2] + dec_str[:dec_str.index(".") + 2]

    return name


def model_check(sources, model: Union[str, List[str], BaseModel1D, List[BaseModel1D]]) \
        -> Union[List[BaseModel1D], List[str]]:
    """
    Very simple function that checks if a passed set of models is appropriately structured for the number of sources
    that have been passed. I can't imagine why a user would need this directly, its only here as these checks
    have to be performed in multiple places in sourcetools.

    :param List[BaseSource]/BaseSample/BaseSource sources: The source(s).
    :param str/List[str]/BaseModel1D/List[BaseModel1D] model: The model(s).
    :return: A list of model instances, or names of models.
    :rtype: Union[List[BaseModel1D], List[str]]
    """

    # This is when there is a single model instance or model name given for a single source. Obviously this is
    #  fine, but we need to put it in a list because the functions that use this want everything to be iterable
    if isinstance(model, (str, BaseModel1D)) and len(sources) == 1:
        model = [model]
    # Here we deal with a single model name for a SET of sources - as the fit method will use strings to declare
    #  model instances we just make a list of the same length as the sample we're analysing (full of the same string)
    elif isinstance(model, str) and len(sources) != 1:
        model = [model]*len(sources)
    # Here we deal with a single model INSTANCE for a SET of sources - this is slightly more complex, as we don't want
    #  to just fill a list full of a bunch of pointers to the same memory address (instance). As such we store copies
    #  of the model in the list, one for each source
    elif isinstance(model, BaseModel1D) and len(sources) != 1:
        model = [deepcopy(model) for s_ind in range(len(sources))]
    # These next conditionals just catch when the user has done something silly - you can figure it out from the
    #  error messages
    elif isinstance(model, list) and len(model) != len(sources):
        raise ValueError("If you pass a list of model names (or model instances), then that list must be the same"
                         " length as the number of sources passed for analysis.")
    elif isinstance(model, list) and not all([isinstance(m, (str, BaseModel1D)) for m in model]):
        raise TypeError("If you pass a list, then every element must be either a string model name, or a "
                        "model instance.")
    elif not isinstance(model, (list, BaseModel1D, str)):
        raise TypeError("The model argument must either be a string model name, a single instance of a model, a list"
                        " of model names, or a list of model instances.")

    return model











