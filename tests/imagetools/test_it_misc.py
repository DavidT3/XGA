#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/04/2023, 23:33. Copyright (c) The Contributors
import pytest
from astropy.cosmology import Planck15
from astropy.units import Quantity
from numpy import zeros
from numpy.testing import assert_array_equal

from xga.imagetools.misc import pix_deg_scale, sky_deg_scale, pix_rad_to_physical, physical_rad_to_pix, \
    data_limits, edge_finder
from xga.products.phot import Image
from .. import A907_LOC, A907_IM_PN_INFO

OFF_PIX = Quantity([20, 20], 'pix')


@pytest.mark.simple
@pytest.mark.data
@pytest.mark.parametrize("test_coord, test_wcs, test_offset, expected", [(A907_LOC, Image(*A907_IM_PN_INFO).radec_wcs,
                                                                          Quantity(1, 'arcmin'),
                                                                          Quantity(0.0012083328398207695, 'deg/pix')),
                                                                         (A907_LOC, Image(*A907_IM_PN_INFO).radec_wcs,
                                                                          Quantity(0.001, 'arcmin'),
                                                                          Quantity(0.0012083330447300933, 'deg/pix')),
                                                                         (OFF_PIX, Image(*A907_IM_PN_INFO).radec_wcs,
                                                                          Quantity(1, 'arcmin'),
                                                                          Quantity(0.0012082903638784826, 'deg/pix'))
                                                                         ])
def test_pix_deg_scale(test_coord, test_wcs, test_offset, expected):
    """
    Testing that the calculation of scale between the pixels and degrees of an image is correct. Tests include
    different coordinate units and different offsets.
    """
    assert pix_deg_scale(test_coord, test_wcs, test_offset) == expected


@pytest.mark.simple
@pytest.mark.data
@pytest.mark.parametrize("test_im, test_coord, test_offset, expected", [(Image(*A907_IM_PN_INFO), A907_LOC,
                                                                         Quantity(1, 'arcmin'),
                                                                         1.388888321633066e-05),
                                                                        (Image(*A907_IM_PN_INFO), A907_LOC,
                                                                         Quantity(0.001, 'arcmin'),
                                                                         1.3888885571577373e-05),
                                                                        (Image(*A907_IM_PN_INFO), OFF_PIX,
                                                                         Quantity(1, 'arcmin'),
                                                                         1.3888394987109003e-05)])
def test_sky_deg_scale(test_im, test_coord, test_offset, expected):
    """
    Testing that the calculation of scale between the degrees and sky coordinates of an XMM image is correct. Tests
    include different coordinate units and different offsets. I check only the values here because the way pytest
    runs tests evidently doesn't define xmm_sky.
    """
    assert sky_deg_scale(test_im, test_coord, test_offset).value == expected


@pytest.mark.simple
@pytest.mark.data
@pytest.mark.parametrize("test_im, test_rad, out_unit, expected", [(Image(*A907_IM_PN_INFO), Quantity(103, 'pix'), 'deg',
                                                                    Quantity(0.12445828250153926, 'deg')),
                                                                   (Image(*A907_IM_PN_INFO), Quantity(103, 'pix'), 'kpc',
                                                                    Quantity(1275.5389422602298, 'kpc'))])
def test_pix_rad_to_physical(test_im, test_rad, out_unit, expected):
    """
    Testing conversion of radius in pixels to physical units (both proper and angular).
    """
    ret_val = pix_rad_to_physical(test_im, test_rad, out_unit, A907_LOC, 0.16, Planck15)
    assert ret_val == expected


# TODO: Figure out why the output has .000000000000000000001 (or something like that) added to it
@pytest.mark.simple
@pytest.mark.data
@pytest.mark.xfail
@pytest.mark.parametrize("test_im, test_rad, expected", [(Image(*A907_IM_PN_INFO), Quantity(0.12445828250153926, 'deg'),
                                                          Quantity(103, 'pix')),
                                                         (Image(*A907_IM_PN_INFO), Quantity(1275.5389422602298, 'kpc'),
                                                          Quantity(103, 'pix'))])
def test_physical_rad_to_pix(test_im, test_rad, expected):
    """
    Testing conversion of radius in physical units to a radius in pixels.
    """
    ret_val = physical_rad_to_pix(test_im, test_rad, A907_LOC, 0.16, Planck15)
    assert ret_val == expected


@pytest.mark.simple
def test_data_lims_zeros():
    """
    Testing that this function fails properly in the absence of any data in an array.
    """
    test_arr = zeros((512, 512))
    with pytest.raises(ValueError):
        data_limits(test_arr)


@pytest.mark.simple
def test_data_lims_array():
    """
    Testing that a numpy array with some data in it returns the correct limits.
    """
    test_arr = zeros((512, 512))
    test_arr[34, 67] = 2
    test_arr[500, 400] = 1
    test_arr[4, 200] = 3

    assert data_limits(test_arr) == ([66, 401], [3, 501])


@pytest.mark.simple
@pytest.mark.data
def test_data_lims_array():
    """
    Testing that an image input results in the correct limits.
    """
    test_im = Image(*A907_IM_PN_INFO)
    assert data_limits(test_im) == ([25, 436], [67, 482])


# TODO Maybe re-write this so it generates a random rectangle for testing
@pytest.mark.simple
@pytest.mark.parametrize("border, keep_corners", [(False, False), (False, True), (True, True), (True, False)])
def test_edge_finder_array(border, keep_corners):
    """
    Tests that the edge finder works for a simple square in an array. Uses a function from the numpy
    testing framework to check whether the two arrays are the same. Checks for different combinations
    """

    # Sets up an array of zeros and ones, where the ones are in a solid rectangle shape
    test_arr = zeros((100, 100))
    test_arr[30:60, 35:65] = 1
    test_arr[30:60, 35:95] = 1
    # Runs the edge finding function with the passed parameters
    edgy = edge_finder(test_arr, keep_corners=keep_corners, border=border)

    # This is what we construct the expected array in
    expected = zeros((100, 100))
    if not border:
        expected[30, 35:95] = 1
        expected[59, 35:95] = 1
        expected[30:60, 35] = 1
        expected[30:60, 94] = 1
    else:
        # The current behaviour of when a border is generated means that there will be no corners
        #  (see issue #623), though I may change this in the future
        expected[29, 35:95] = 1
        expected[60, 35:95] = 1
        expected[30:60, 34] = 1
        expected[30:60, 95] = 1

    if keep_corners and not border:
        expected[30, 35] += 1
        expected[59, 35] += 1
        expected[30, 94] += 1
        expected[59, 94] += 1

    # Use this numpy function to check whether the arrays are equal
    assert_array_equal(edgy, expected)








