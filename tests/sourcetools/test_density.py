import unittest

import os
import shutil
import numpy as np

from astropy.units import Quantity

from .. import set_up_test_config, restore_og_cfg

# I know it is horrible to write code in the middle of importing modules, but this needs to 
# happen before xga is imported, as we are moving config files
set_up_test_config()


# Now when xga is imported it will make a new census with the test_data
import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.sourcetools.density import _dens_setup, _run_sb, inv_abel_fitted_model, \
                                    ann_spectra_apec_norm
from xga.products.profile import SurfaceBrightness1D, GasDensity3D

class TestSetupFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(226.0318, -2.8046, 0.2093, r500=Quantity(500, 'kpc'),
                    name="1eRASS_J150407.6-024816", use_peak=False,
                    telescope='erosita',
                    search_distance={'erosita': Quantity(3.6, 'deg')})

    @classmethod
    def tearDownClass(cls):
        """
        This is run once after all the tests.
        """
        # This function restores the user's original config file and deletes the test one made
        restore_og_cfg()
        # Then we will delete all the products that xga has made so there aren't loads of big files
        # in the package
#        shutil.rmtree('tests/test_data/xga_output')

    def test_dens_setup(self):
        res = _dens_setup(self.test_src, Quantity(600, 'kpc'), Quantity(100, 'kpc'), 'angr', 
                    Quantity(0.5, 'keV'), Quantity(2, 'keV'), stacked_spectra=True)
        
        assert type(res[0][0]) == GalaxyCluster
        assert type(res[1][0]) == Quantity
        assert set(res[2].keys()) == set(['erosita', 'xmm'])
        assert set(res[3].keys()) == set(['erosita', 'xmm'])

    def test_run_sb(self):
        res = _run_sb(self.test_src, 'erosita', Quantity(600, 'kpc'), False, Quantity(0.5, 'keV'), 
                Quantity(2, 'keV'), False, None, None, None, None, 1, 0.0, None, None)
        
        assert type(res) == SurfaceBrightness1D
    
    def test_inv_abel_fitted_model(self):
        res = inv_abel_fitted_model(self.test_src, 'beta', use_peak=False, psf_corr=False, 
                              stacked_spectra=True)

        assert type(res['erosita'][0]) == GasDensity3D
        assert type(res['xmm'][0]) == GasDensity3D
    
    def test_ann_spectra_apec_norm(self):
        res = ann_spectra_apec_norm(self.test_src, Quantity(600, 'kpc'), stacked_spectra=True)

        assert type(res['erosita'][0]) == GasDensity3D
        assert type(res['xmm'][0]) == GasDensity3D