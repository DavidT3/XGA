#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 12/10/2021, 09:51. Copyright (c) David J Turner

import pytest
from astropy.cosmology import Planck15
from astropy.units import Quantity

from xga.products.phot import Image
from xga.imagetools.misc import pix_deg_scale, sky_deg_scale, pix_rad_to_physical
from .. import A907_LOC, A907_PN_INFO

OFF_PIX = Quantity([20, 20], 'pix')


@pytest.mark.simple
@pytest.mark.data
@pytest.mark.parametrize("test_coord, test_wcs, test_offset, expected", [(A907_LOC, Image(*A907_PN_INFO).radec_wcs,
                                                                          Quantity(1, 'arcmin'),
                                                                          Quantity(0.0012083328398207695, 'deg/pix')),
                                                                         (A907_LOC, Image(*A907_PN_INFO).radec_wcs,
                                                                          Quantity(0.001, 'arcmin'),
                                                                          Quantity(0.0012083330447300933, 'deg/pix')),
                                                                         (OFF_PIX, Image(*A907_PN_INFO).radec_wcs,
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
@pytest.mark.parametrize("test_im, test_coord, test_offset, expected", [(Image(*A907_PN_INFO), A907_LOC,
                                                                         Quantity(1, 'arcmin'),
                                                                         1.388888321633066e-05),
                                                                        (Image(*A907_PN_INFO), A907_LOC,
                                                                         Quantity(0.001, 'arcmin'),
                                                                         1.3888885571577373e-05),
                                                                        (Image(*A907_PN_INFO), OFF_PIX,
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
@pytest.mark.parametrize("test_im, test_rad, out_unit, expected", [(Image(*A907_PN_INFO), Quantity(100, 'pix'), 'deg', 1)])
def test_pix_rad_to_physical(test_im, test_rad, out_unit, expected):
    ret_val = pix_rad_to_physical(test_im, test_rad, out_unit, A907_LOC, 0.16, Planck15)
    print(ret_val)
    assert ret_val == expected


