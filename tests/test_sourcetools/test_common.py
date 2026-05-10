#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 6:08 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga import NUM_CORES
from xga.products.profile import GasDensity3D, GasTemperature3D
from xga.sources import GalaxyCluster
from xga.sourcetools._common import _get_all_telescopes, _setup_global, \
    _setup_inv_abel_dens_onion_temp
from .. import get_test_source


class TestSetupFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    def test_get_all_telescopes_list_input(self):
        res = _get_all_telescopes([self.src])
        assert set(res) == {'erass', 'xmm'}

    def test_get_all_telescopes_source_input(self):
        res = _get_all_telescopes(self.src)
        assert set(res) == {'erass', 'xmm'}

    def test_setup_global(self):
        res = _setup_global(self.src, Quantity(600, 'kpc'), Quantity(600, 'kpc'), 'angr', True,
                            5, None, None, NUM_CORES, 4, True, ['xmm', 'erass'])

        assert isinstance(res[0][0], GalaxyCluster)
        assert type(res[1][0]) == Quantity
        assert type(res[2]) == dict
        assert set(res[2].keys()) == {'erass', 'xmm'}
        assert len(res[2]['erass']) == 1

    def test_setup_inv_abel_dens_onion_temp(self):
        res = _setup_inv_abel_dens_onion_temp(self.src, Quantity(600, 'kpc'),
                                              'beta', 'king', 'vikhlinin_temp',
                                              Quantity(600, 'kpc'),
                                              stacked_spectra=True)

        assert type(res[0]) == list
        assert len(res[0]) == 1
        assert type(res[1]) == dict
        assert type(res[1][repr(self.src)]['erass']) == GasDensity3D
        assert type(res[1][repr(self.src)]['xmm']) == GasDensity3D
        assert type(res[2]) == dict
        assert type(res[2][repr(self.src)]['erass']) == GasTemperature3D
        assert type(res[2][repr(self.src)]['xmm']) == GasTemperature3D
        assert type(res[3]) == dict
        assert type(res[3][repr(self.src)]) == str
        assert res[3][repr(self.src)] == 'king'
        assert type(res[4]) == dict
        assert type(res[4][repr(self.src)]) == str
        assert res[4][repr(self.src)] == 'vikhlinin_temp'



