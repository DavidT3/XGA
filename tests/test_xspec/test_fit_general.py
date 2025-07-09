import unittest
import sys
import os

from astropy.units import Quantity

import xga
from xga.samples import ClusterSample
from xga.xspec.fit.general import single_temp_apec

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import CLUSTER_SMP

class TestBaseSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_smp = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values, 
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values, 
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erosita': Quantity(3.6, 'deg')})
    
    def test_Lx_w_stacked_spectra(self):
        single_temp_apec(self.test_smp, 'r500', stacked_spectra=True, spectrum_checking=False)
        Lx = self.test_smp.Lx('r500', 'erosita', stacked_spectra=True)

        assert len(Lx) == 2
        assert isinstance(Lx, Quantity)