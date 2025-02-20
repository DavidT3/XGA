import unittest

from astropy.units import Quantity

import xga
from xga.sources import GalaxyCluster
from xga.generate.esass.phot import evtool_image, expmap
from xga.generate.sas.phot import evselect_image, eexpmap, emosaic
from xga.sourcetools.density import _dens_setup, _run_sb, inv_abel_fitted_model, \
                                    ann_spectra_apec_norm
from xga.products.profile import SurfaceBrightness1D, GasDensity3D

from .. import SRC_ALL_TELS

class TestDensityFuncs(unittest.TestCase):
    def test_dens_setup(self):
        res = _dens_setup(SRC_ALL_TELS, Quantity(600, 'kpc'), Quantity(100, 'kpc'), 'angr', 
                    Quantity(0.5, 'keV'), Quantity(2, 'keV'), stacked_spectra=True)
        
        assert type(res[0][0]) == GalaxyCluster
        assert type(res[1][0]) == Quantity
        assert set(res[2].keys()) == set(['erosita', 'xmm'])
        assert set(res[3].keys()) == set(['erosita', 'xmm'])

    def test_run_sb(self):
        res = _run_sb(SRC_ALL_TELS, 'erosita', Quantity(600, 'kpc'), False, Quantity(0.5, 'keV'), 
                Quantity(2, 'keV'), False, None, None, None, None, 1, 0.0, None, None)
        
        assert type(res) == SurfaceBrightness1D
    
    def test_inv_abel_fitted_model(self):
        res = inv_abel_fitted_model(SRC_ALL_TELS, 'beta', use_peak=False, psf_corr=False, 
                              stacked_spectra=True)

        assert type(res['erosita'][0]) == GasDensity3D
        assert type(res['xmm'][0]) == GasDensity3D
    
    def test_ann_spectra_apec_norm(self):
        res = ann_spectra_apec_norm(SRC_ALL_TELS, Quantity(600, 'kpc'), stacked_spectra=True)

        assert type(res['erosita'][0]) == GasDensity3D
        assert type(res['xmm'][0]) == GasDensity3D