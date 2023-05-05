#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

from typing import Union

import numpy as np
from astropy.units import Quantity, UnitConversionError


def shell_ann_vol_intersect(shell_radii: Union[np.ndarray, Quantity], ann_radii: Union[np.ndarray, Quantity]) \
        -> Union[np.ndarray, Quantity]:
    """
    This function calculates the volume intersection matrix of a set of circular annuli and a
    set of spherical shells. It is assumed that the annuli and shells have the same x and y origin. The
    intersection is derived using simple geometric considerations, have a look in the appendix of DOI 10.1086/300836.

    :param ndarray/Quantity shell_radii: The radii of the spherical shells.
    :param ndarray/Quantity ann_radii: The radii of the circular annuli (DOES NOT need to be the same
        length as shell_radii).
    :return: A 2D array containing the volumes of intersections between the circular annuli defined by
        i_ann and o_ann, and the spherical shells defined by i_sph and o_sph. Annular radii are along the 'x' axis
        and shell radii are along the 'y' axis.
    :rtype: Union[np.ndarray, Quantity]
    """
    if all([type(shell_radii) == Quantity, type(ann_radii) == Quantity]) and shell_radii.unit != ann_radii.unit:
        raise UnitConversionError("If quantities are passed, they must be in the same units.")
    elif all([type(shell_radii) == Quantity, type(ann_radii) == Quantity]):
        pass
    elif all([type(shell_radii) == np.ndarray, type(ann_radii) == np.ndarray]):
        pass
    else:
        raise TypeError("shell_radii and ann_radii must either both be astropy quantities or numpy arrays, "
                        "you cannot mix the two")

    i_ann, i_sph = np.meshgrid(ann_radii[0:-1], shell_radii[0:-1])
    o_ann, o_sph = np.meshgrid(ann_radii[1:], shell_radii[1:])
    # The main term which makes use of the radii of the shells and annuli.
    # The use of clip enforces that none of the terms can be less than 0, as we don't care
    #  about those intersections, you can't have a negative volume. The None passed to clip is just to tell it
    #  that there is no upper limit that we wish to enforce.
    main_term = np.power(np.clip((o_sph ** 2 - i_ann ** 2), 0, None), 3 / 2) - \
                np.power(np.clip((o_sph ** 2 - o_ann ** 2), 0, None), 3 / 2) + \
                np.power(np.clip((i_sph ** 2 - o_ann ** 2), 0, None), 3 / 2) - \
                np.power(np.clip((i_sph ** 2 - i_ann ** 2), 0, None), 3 / 2)

    # Multiply by the necessary constants and return.
    return (4 / 3) * np.pi * main_term


def shell_volume(inn_radius: Quantity, out_radius: Quantity) -> Union[Quantity, np.ndarray]:
    """
    Silly little function that calculates the volume of a spherical shell with inner radius inn_radius and outer
    radius out_radius.

    :param Quantity/np.ndarray inn_radius: The inner radius of the spherical shell.
    :param Quantity/np.ndarray out_radius: The outer radius of the spherical shell.
    :return: The volume of the specified shell
    :rtype: Union[Quantity, np.ndarray]
    """
    if all([type(inn_radius) == Quantity, type(out_radius) == Quantity]) and inn_radius.unit != out_radius.unit:
        raise UnitConversionError("If quantities are passed, they must be in the same units.")
    elif all([type(inn_radius) == Quantity, type(out_radius) == Quantity]):
        pass
    elif all([type(inn_radius) == np.ndarray, type(out_radius) == np.ndarray]):
        pass
    else:
        raise TypeError("inn_radius and out_radius must either both be astropy quantities or numpy arrays, "
                        "you cannot mix the two")

    outer_vol = (4/3) * np.pi * out_radius**3
    inner_vol = (4/3) * np.pi * inn_radius**3

    return outer_vol - inner_vol


# def temp_onion(proj_prof: ProjectedGasTemperature1D) -> GasTemperature3D:
#     """
#     This function will generate deprojected, three-dimensional, gas temperature profile from a projected profile using
#     the 'onion peeling' deprojection method. The function is an implementation of a fairly old technique, though it
#     has been used recently in https://doi.org/10.1051/0004-6361/201731748. For a more in depth discussion of this
#     technique and its uses I would currently recommend https://doi.org/10.1051/0004-6361:20020905.
#
#     :param ProjectedGasTemperature1D proj_prof: A projected cluster temperature profile, which the
#         user wants to use to infer the 3D temperature profile.
#     :return: The deprojected temperature profile.
#     :rtype: GasTemperature3D
#     """
#
#     pass






