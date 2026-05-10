#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 7:19 PM. Copyright (c) The Contributors.

import unittest

import numpy as np
from astropy.units import Quantity

from xga.samples import ClusterSample
from xga.xspec.fit.general import single_temp_apec
from .. import CLUSTER_SMP, get_test_source


class TestXSPECSingleSource(unittest.TestCase):
    """
    This class houses tests that use XGA's XSPEC functionality on source class instances, as opposed to
    sample class instances.
    """

    @classmethod
    def setUpClass(cls):
        cls.test_src = get_test_source('all')

    def test_global_single_temp_apec(self):
        par_combs = [
            {"stacked_spectra": True},
            {"stacked_spectra": False},
        ]

        out_rad = Quantity(500, 'kpc')

        for cur_pc in par_combs:
            with self.subTest(msg=f"Test global single_temp_apec with {cur_pc}", **cur_pc):

                single_temp_apec(self.test_src, out_rad, **cur_pc)

                for cur_tel in self.test_src.telescopes:
                    cur_tx = self.test_src.get_temperature(out_rad, cur_tel, **cur_pc)


class TestBaseSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_smp = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values,
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values,
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erass': Quantity(3.6, 'deg')})

        cls.smp_odd_tels = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values,
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values,
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erass': Quantity(3.6, 'deg')})
        cls.smp_odd_tels[0].disassociate_obs('erass')

    def test_Lx_w_stacked_spectra(self):
        single_temp_apec(self.test_smp, 'r500', stacked_spectra=True, spectrum_checking=False)
        Lx = self.test_smp.Lx('r500', 'erass', stacked_spectra=True)

        assert len(Lx) == 2
        assert isinstance(Lx, Quantity)

    def test_Lx_Tx_w_odd_telescope_sample(self):
        """
        Testing that for samples where sources don't have the same telescopes assigned, the Lx and Tx
        can be retrieved.
        """
        single_temp_apec(self.smp_odd_tels, 'r500', stacked_spectra=True, spectrum_checking=False)

        cur_lx = self.smp_odd_tels.Lx('r500', 'erass', stacked_spectra=True)

        assert isinstance(cur_lx, Quantity)
        assert len(cur_lx) == 2
        assert np.isnan(cur_lx[0][0].value)

