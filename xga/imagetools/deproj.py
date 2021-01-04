#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 04/01/2021, 19:36. Copyright (c) David J Turner

from typing import Union

import numpy as np


# TODO Should this function give an option for including different origins for the shells and annuli?
def sphere_circann_vol_intersec(shell_radii: np.ndarray, ann_radii: np.ndarray) -> np.ndarray:
    """
    This function calculates the volume intersection matrix of a set of circular annuli and a
    set of spherical shells. It is assumed that the annuli and shells have the same x and y origin. The
    intersection is derived using simple geometric considerations, have a look in the appendix of DOI 10.1086/300836.

    :param Union[float,ndarray] shell_radii: The radii of the spherical shells.
    :param Union[float,ndarray] ann_radii: The radii of the circular annuli (DOES NOT need to be the same
        length as shell_radii).
    :return: A 2D array containing the volumes of intersections between the circular annuli defined by
        i_ann and o_ann, and the spherical shells defined by i_sph and o_sph. Annular radii are along the 'x' axis
        and shell radii are along the 'y' axis.
    :rtype: ndarray
    """
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










