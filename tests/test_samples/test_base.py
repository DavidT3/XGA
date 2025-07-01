import unittest
import numpy as np
import os 
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from astropy.units import Quantity

from xga.samples import ClusterSample

from .. import SRC_INFO, SUPP_SRC_INFO

class TestBaseSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_smp = ClusterSample([SRC_INFO['ra'], SUPP_SRC_INFO['ra']],
                                     [SRC_INFO['dec'], SUPP_SRC_INFO['dec']],
                                     [SRC_INFO['z'], SUPP_SRC_INFO['z']],
                                     r500=Quantity([500, 500], 'kpc'),
                                     name=[SRC_INFO['name'], SUPP_SRC_INFO['name']],
                                     use_peak=False)

    def test_smp_names(self):
        assert set([SRC_INFO['name'], SUPP_SRC_INFO['name']]) == set(self.test_smp.names)

    def test_smp_redshifts(self):
        assert set([SRC_INFO['z'], SUPP_SRC_INFO['z']]) == set(self.test_smp.redshifts)

    def test_smp_telescopes(self):
        assert set(['erosita', 'xmm']) == set(self.test_smp.telescopes)
    
    def test_smp_src_telescopes(self):
        assert self.test_smp.src_telescopes[SRC_INFO['name']] == set(['xmm', 'erosita'])
        assert self.test_smp.src_telescopes[SUPP_SRC_INFO['name']] == set(['xmm', 'erosita'])
