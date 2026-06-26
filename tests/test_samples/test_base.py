#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/5/26, 11:52 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.samples import ClusterSample
from .. import SRC_INFO, SUPP_SRC_INFO, CLUSTER_SMP


class TestBaseSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_smp = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values,
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values,
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erass': Quantity(3.6, 'deg')})

    def test_smp_names(self):
        assert {SRC_INFO['name'].replace("_", "-"), SUPP_SRC_INFO['name']} == set(self.test_smp.names)

    def test_smp_redshifts(self):
        assert {SRC_INFO['z'], SUPP_SRC_INFO['z']} == set(self.test_smp.redshifts)

    def test_smp_telescopes(self):
        assert {'erass', 'xmm'} == set(self.test_smp.telescopes)

    def test_smp_src_telescopes(self):
        assert set(self.test_smp.src_telescopes[SRC_INFO['name'].replace("_", "-")]) == {'xmm', 'erass'}
        assert set(self.test_smp.src_telescopes[SUPP_SRC_INFO['name']]) == {'xmm', 'erass'}
