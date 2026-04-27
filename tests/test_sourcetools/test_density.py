#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:19 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

from astropy.units import Quantity

from xga.products.profile import SurfaceBrightness1D, GasDensity3D
from xga.sources import GalaxyCluster
from xga.sourcetools.density import _dens_setup, _run_sb, inv_abel_fitted_model, \
    ann_spectra_apec_norm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import get_test_source

class TestDensityFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

#    def test_dens_setup(self):
#        res = _dens_setup(self.src, 'angr',Quantity(0.5, 'keV'), Quantity(2, 'keV'), stacked_spectra=True)
#        assert type(res[0][0]) == GalaxyCluster
#        assert set(res[2].keys()) == set(['erosita', 'xmm'])
#        assert set(res[3].keys()) == set(['erosita', 'xmm'])

#    def test_run_sb(self):
#        res = _run_sb(self.src, 'erosita', Quantity(600, 'kpc'), False, Quantity(0.5, 'keV'), 
#                Quantity(2, 'keV'), False, None, None, None, None, 1, 0.0, None, None)
        
#        assert type(res) == SurfaceBrightness1D
    
    def test_inv_abel_fitted_model(self):
        inv_abel_fitted_model(self.src, 'beta', use_peak=False, psf_corr=False, stacked_spectra=True)

#        assert type(res['erosita'][0]) == GasDensity3D
#        assert type(res['xmm'][0]) == GasDensity3D

#   def test_inv_abel_fitted_model_stacked_spectrum_F(self):
#        
#        res = inv_abel_fitted_model(self.src, 'beta', use_peak=False, psf_corr=False, stacked_spectra=False)
#        print(res)
#        assert type(res['xmm'][0]) == GasDensity3D
#        assert type(res['erosita'][0]) == GasDensity3D
    
#    def test_ann_spectra_apec_norm(self):
#        res = ann_spectra_apec_norm(self.src, Quantity(600, 'kpc'), stacked_spectra=True)

#        assert type(res['erosita'][0]) == GasDensity3D
#        assert type(res['xmm'][0]) == GasDensity3D


