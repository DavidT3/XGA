#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

import pytest
from astropy.cosmology import Planck15
from astropy.units import Quantity

from xga.sourcetools.misc import nh_lookup, rad_to_ang, ang_to_rad, name_to_coord, coord_to_name
from .. import A907_LOC


@pytest.mark.heasoft
def test_nh_value():
    """
    Tests whether the nh_lookup tool is correctly interfacing with nh command and retrieving values. It possible
    that these values will change depending on the heasoft version (as different hydrogen maps are used), so a failure
    due to that reason isn't critical.
    """
    ret_val = nh_lookup(A907_LOC)
    assert ret_val[0].value == 0.0534 and ret_val[1].value == 0.0545


@pytest.mark.simple
def test_rad2ang():
    """
    Tests whether the conversion of a proper radius to angular radius is correct with a given redshift and cosmology.
    Not sure why I did this one as I can't imagine it ever being anything but true.
    """
    ret_val = rad_to_ang(Quantity(1000, 'kpc'), 0.3, Planck15)
    assert ret_val == Quantity(0.06046667460469272, 'deg')


@pytest.mark.simple
def test_ang2rad():
    """
    Tests whether the conversion of a angular radius to proper radius is correct with a given redshift and cosmology.
    Not sure why I did this one as I can't imagine it ever being anything but true.
    """
    ret_val = ang_to_rad(Quantity(0.06046667460469272, 'deg'), 0.3, Planck15)
    assert ret_val == Quantity(1000., 'kpc')


# I expect this to fail right now because I haven't finished it, I need to replace this once I have done it
@pytest.mark.xfail
@pytest.mark.simple
def test_name2coord():
    name_to_coord("XMMXCS J041853.9+555333.7")


@pytest.mark.simple
@pytest.mark.parametrize("test_coord, test_survey, expected", [(A907_LOC, 'XMMXCS', 'XMMXCSJ095822.1-110334.9'),
                                                               (A907_LOC, None, 'J095822.1-110334.9')])
def test_coord2name(test_coord, test_survey, expected):
    assert coord_to_name(test_coord, test_survey) == expected



