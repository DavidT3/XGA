#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/06/2020, 13:52. Copyright (c) David J Turner

from astropy.units import Quantity
from numpy import ogrid, ndarray, arctan2, pi, repeat, newaxis, array, greater


def annular_mask(cen_x: int, cen_y: int, inn_rad: ndarray, out_rad: ndarray, len_x: int, len_y: int,
                 start_ang: Quantity = Quantity(0, 'deg'), stop_ang: Quantity = Quantity(360, 'deg')) -> ndarray:
    """
    A handy little function to generate annular (or circular) masks in the form of numpy arrays.
    It produces the mask for a given shape of image, centered at supplied coordinates, and with inner and
    outer radii supplied by the user also. Angular limits can also be supplied to give the mask an annular
    dependence. This function should be properly vectorised, and accepts inner and outer radii in
    the form of arrays.
    The result will be an len_y, len_x, N dimensional array, where N is equal to the length of inn_rad.
    :param int cen_x: Numpy array x-coordinate of the center for this mask.
    :param int cen_y: Numpy array y-coordinate of the center for this mask.
    :param ndarray inn_rad: Pixel radius for the inner part of the annular mask.
    :param ndarray out_rad: Pixel radius for the outer part of the annular mask.
    :param Quantity start_ang: Lower angular limit for the mask.
    :param Quantity stop_ang: Upper angular limit for the mask.
    :param int len_x: Length in the x direction of the array/image this mask is for.
    :param int len_y: Length in the y direction of the array/image this mask is for.
    :return: The generated mask array.
    :rtype: ndarray
    """
    # Making use of the astropy units module, check that we are being pass actual angle values
    if start_ang.unit not in ['deg', 'rad']:
        raise ValueError("start_angle unit type {} is not an accepted angle unit, "
                         "please use deg or rad.".format(start_ang.unit))
    elif stop_ang.unit not in ['deg', 'rad']:
        raise ValueError("stop_angle unit type {} is not an accepted angle unit, "
                         "please use deg or rad.".format(stop_ang.unit))
    # Enforcing some common sense rules on the angles
    elif start_ang >= stop_ang:
        raise ValueError("start_ang cannot be greater than or equal to stop_ang.")
    elif start_ang > Quantity(360, 'deg') or stop_ang > Quantity(360, 'deg'):
        raise ValueError("start_ang and stop_ang cannot be greater than 360 degrees.")
    elif stop_ang < Quantity(0, 'deg'):
        raise ValueError("stop_ang cannot be less than 0 degrees.")
    else:
        # Don't want to pass astropy objects to numpy functions, but do need the angles in radians
        start_ang = start_ang.to('rad').value
        stop_ang = stop_ang.to('rad').value

    # Check that if the inner and outer radii are arrays, they're the same length
    if isinstance(inn_rad, (ndarray, list)) and len(inn_rad) != len(out_rad):
        raise ValueError("inn_rad and out_rad are not the same length")
    elif isinstance(inn_rad, list) and len(inn_rad) == len(out_rad):
        # If it is a list, just quickly transform to a numpy array
        inn_rad = array(inn_rad)
        out_rad = array(out_rad)

    # This sets up the cartesian coordinate grid of x and y values
    arr_y, arr_x = ogrid[:len_y, :len_x]

    # Go to polar coordinates
    rec_x = arr_x - cen_x
    rec_y = arr_y - cen_y
    # Leave this as r**2 to avoid square rooting and involving floats
    init_r_squared = rec_x**2 + rec_y**2

    # arctan2 does just perform arctan on two values, but then uses the signs of those values to
    # decide the quadrant of the output
    init_arr_theta = (arctan2(rec_x, rec_y) - start_ang) % (2*pi)  # Normalising to 0-2pi range

    # If the radius limits are an array, the arrays that describe the space we have constructed are copied
    #  into a third dimension - This allows masks for different radii to be generated in a vectorised fashion
    if isinstance(inn_rad, ndarray):
        arr_r_squared = repeat(init_r_squared[:, :, newaxis], len(inn_rad), axis=2)
        arr_theta = repeat(init_arr_theta[:, :, newaxis], len(inn_rad), axis=2)
    else:
        arr_r_squared = init_r_squared
        arr_theta = init_arr_theta

    # This will deal properly with inn_rad and out_rads that are arrays
    if greater(inn_rad, out_rad).any():
        raise ValueError("inn_rad value cannot be greater than out_rad")
    else:
        rad_mask = (arr_r_squared < out_rad ** 2) & (arr_r_squared >= inn_rad ** 2)

    # Finally, puts a cut on the allowed angle, and combined the radius and angular cuts into the final mask
    ang_mask = arr_theta <= (stop_ang - start_ang)
    ann_mask = rad_mask * ang_mask

    # Returns the annular mask(s), in the form of a len_y, len_x, N dimension np array
    return ann_mask




