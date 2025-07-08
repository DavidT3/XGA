import unittest
import os
import sys

from astropy.units import Quantity

import xga
from xga.samples import ClusterSample
from xga.generate.esass.phot import evtool_image, expmap

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .. import SRC_ALL_TELS, SRC_INFO, CLUSTER_SMP

class TestEsassPhotFuncs(unittest.TestCase):
    def test_evtool_image(self):
        evtool_image(SRC_ALL_TELS, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        im = SRC_ALL_TELS.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')[0]
        if isinstance(im, list):
            for i in im:
                assert i.telescope == 'erosita'
                assert i.energy_bounds[0] == Quantity(0.4, 'keV')
                assert i.energy_bounds[1] == Quantity(3, 'keV')
        else:
            assert im.telescope == 'erosita'
            assert im.energy_bounds[0] == Quantity(0.4, 'keV')
            assert im.energy_bounds[1] == Quantity(3, 'keV')


    def test_evtool_image_combined_obs(self):
        evtool_image(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        im = SRC_ALL_TELS.get_combined_images(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert im.telescope == 'erosita'
        assert im.energy_bounds[0] == Quantity(0.5, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')

    def test_expmap(self):
        expmap(SRC_ALL_TELS, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        exp = SRC_ALL_TELS.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        for e in exp:
            assert e.telescope == 'erosita'
            assert e.energy_bounds[0] == Quantity(0.4, 'keV')
            assert e.energy_bounds[1] == Quantity(3, 'keV')

    def test_expmap_combined_obs(self):
        expmap(SRC_ALL_TELS, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        exp = SRC_ALL_TELS.get_combined_expmaps(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert exp.telescope == 'erosita'
        assert exp.energy_bounds[0] == Quantity(0.5, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')
    
    def test_evtool_image_w_sample_w_odd_telescopes(self):
        """
        There was an old bug that occured when product generation functions were run with samples
        with sources that didnt all have the same telescopes. So this is testing that this bug has
        been fixed!
        """
        test_smp = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values, 
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values, 
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erosita': Quantity(3.6, 'deg')})

        print(test_smp[0].telescopes)
        print(test_smp[1].telescopes)
        test_smp[0].disassociate_obs('erosita')
        evtool_image(test_smp)

if __name__ == "__main__":
     unittest.main()
