import unittest
import sys
import os

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.products.profile import GasDensity3D, GasTemperature3D
from xga.sourcetools._common import _get_all_telescopes, _setup_global, \
                                    _setup_inv_abel_dens_onion_temp
from xga import NUM_CORES


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .. import SRC_ALL_TELS

class TestSetupFuncs(unittest.TestCase):
    def test_get_all_telescopes_list_input(self):
        res = _get_all_telescopes([SRC_ALL_TELS])
        assert set(res) == set(['erosita', 'xmm'])

    def test_get_all_telescopes_source_input(self):
        res = _get_all_telescopes(SRC_ALL_TELS)
        assert set(res) == set(['erosita', 'xmm'])

    def test_setup_global(self):
        res = _setup_global(SRC_ALL_TELS, Quantity(600, 'kpc'), Quantity(600, 'kpc'), 'angr', True,
                            5, None, None, NUM_CORES, 4, True, ['xmm', 'erosita'])
        print('in test_setup_global')
        print(res)
        
        assert res[0][0] == GalaxyCluster
        assert type(res[1][0]) == Quantity
        assert type(res[2]) == dict
        assert set(res[2].keys()) == set(['erosita', 'xmm'])
        assert len(res[2]['erosita']) == 1 

    def test_setup_inv_abel_dens_onion_temp(self):
        res = _setup_inv_abel_dens_onion_temp(SRC_ALL_TELS, Quantity(600, 'kpc'), 
                                              'beta', 'king', 'vikhlinin_temp', 
                                              Quantity(600, 'kpc'), 
                                              stacked_spectra=True)
        print('in test_setup_inv_abel_dens_onion_temp')
        print(res)
        
        assert type(res[0]) == list
        assert len(res[0]) == 1
        assert type(res[1]) == dict
        assert type(res[1][repr(SRC_ALL_TELS)]['erosita']) == GasDensity3D
        assert type(res[1][repr(SRC_ALL_TELS)]['xmm']) == GasDensity3D
        assert type(res[2]) == dict
        assert type(res[2][repr(SRC_ALL_TELS)]['erosita']) == GasTemperature3D
        assert type(res[2][repr(SRC_ALL_TELS)]['xmm']) == GasTemperature3D
        assert type(res[3]) == dict
        assert type(res[3][repr(SRC_ALL_TELS)]) == str
        assert res[3][repr(SRC_ALL_TELS)] == 'king'
        assert type(res[4]) == dict
        assert type(res[4][repr(SRC_ALL_TELS)]) == str
        assert res[3][repr(SRC_ALL_TELS)] == 'vikhlinin_temp'
