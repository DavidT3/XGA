#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 10:37 AM. Copyright (c) The Contributors.

import os
import sys
import unittest

import numpy as np
from astropy.units import Quantity

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

        cls.smp_odd_tels = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values,
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values,
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erosita': Quantity(3.6, 'deg')})
        cls.smp_odd_tels[0].disassociate_obs('erosita')

    def test_Lx_w_stacked_spectra(self):
        single_temp_apec(self.test_smp, 'r500', stacked_spectra=True, spectrum_checking=False)
        Lx = self.test_smp.Lx('r500', 'erosita', stacked_spectra=True)

        assert len(Lx) == 2
        assert isinstance(Lx, Quantity)

    def test_Lx_Tx_w_odd_telescope_sample(self):
        """
        Testing that for samples were sources dont have the same telescopes assigned, the Lx and Tx
        can be retrieved.
        """
        single_temp_apec(self.smp_odd_tels, 'r500', stacked_spectra=True, spectrum_checking=False)

        Lx = self.smp_odd_tels.Lx('r500', 'erosita', stacked_spectra=True)

        assert len(Lx) == 2
        assert np.isnan(Lx[0][0].value)
        assert isinstance(Lx, Quantity)

