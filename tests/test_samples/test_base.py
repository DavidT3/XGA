import unittest
import numpy as np
import os 
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from astropy.units import Quantity

from xga.samples import ClusterSample

from .. import SRC_INFO, SUPP_SRC_INFO, CLUSTER_SMP

class TestBaseSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_smp = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values, 
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values, 
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erosita': Quantity(3.6, 'deg')})

    def test_smp_names(self):
        assert set([SRC_INFO['name'].replace("_", "-"), SUPP_SRC_INFO['name']]) == set(self.test_smp.names)

    def test_smp_redshifts(self):
        assert set([SRC_INFO['z'], SUPP_SRC_INFO['z']]) == set(self.test_smp.redshifts)

    def test_smp_telescopes(self):
        assert set(['erosita', 'xmm']) == set(self.test_smp.telescopes)
    
    def test_smp_src_telescopes(self):
        assert set(self.test_smp.src_telescopes[SRC_INFO['name'].replace("_", "-")]) == set(['xmm', 'erosita'])
        assert set(self.test_smp.src_telescopes[SUPP_SRC_INFO['name']]) == set(['xmm', 'erosita'])
