#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/29/26, 9:39 AM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity
from xga.generate.esass.phot import evtool_image, expmap
from xga.samples import ClusterSample

from .. import get_test_source, CLUSTER_SMP
from ..utils import require_esass


class TestEsassPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('erosita')

    @require_esass
    def test_evtool_image(self):
        evtool_image(self.src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        im = self.src.get_images(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
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

    @require_esass
    def test_evtool_image_combined_obs(self):
        evtool_image(self.src, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        im = self.src.get_combined_images(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert im.telescope == 'erosita'
        assert im.energy_bounds[0] == Quantity(0.5, 'keV')
        assert im.energy_bounds[1] == Quantity(3, 'keV')

    @require_esass
    def test_expmap(self):
        expmap(self.src, Quantity(0.4, 'keV'), Quantity(3, 'keV'))

        exp = self.src.get_expmaps(lo_en=Quantity(0.4, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        for e in exp:
            assert e.telescope == 'erosita'
            assert e.energy_bounds[0] == Quantity(0.4, 'keV')
            assert e.energy_bounds[1] == Quantity(3, 'keV')

    @require_esass
    def test_expmap_combined_obs(self):
        expmap(self.src, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        exp = self.src.get_combined_expmaps(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'), 
                                      telescope='erosita')

        assert exp.telescope == 'erosita'
        assert exp.energy_bounds[0] == Quantity(0.5, 'keV')
        assert exp.energy_bounds[1] == Quantity(3, 'keV')
    
    @require_esass
    def test_evtool_image_w_sample_w_odd_telescopes(self):
        """
        There was an old bug that occured when product generation functions were run with samples
        with sources that didn't all have the same telescopes. So this is testing that this bug has
        been fixed!
        """
        test_smp = ClusterSample(CLUSTER_SMP["ra"].values, CLUSTER_SMP["dec"].values, 
                                 CLUSTER_SMP["z"].values, CLUSTER_SMP["name"].values, 
                                 r500=Quantity(CLUSTER_SMP["r500"].values, 'kpc'), use_peak=False,
                                 search_distance={'erosita': Quantity(3.6, 'deg')})

        test_smp[0].disassociate_obs('erosita')
        evtool_image(test_smp)

if __name__ == "__main__":
     unittest.main()



