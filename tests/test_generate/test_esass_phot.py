#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 6:19 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.exceptions import NoProductAvailableError
from xga.generate.esass.phot import evtool_image, expmap
from xga.products import Image
from xga.samples import ClusterSample
from .. import get_test_source, CLUSTER_SMP
from ..utils import require_esass


class TestEsassPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('erass')

        # Set the lower and upper energy bounds of the phot products generated and retrieved
        #  in the tests implemented in this class
        cls._phot_lo_en = Quantity(0.4, "keV")
        cls._phot_hi_en = Quantity(3.0, "keV")

    @require_esass
    def test_evtool_image(self):
        """
        Generates and retrieves one image per eROSITA ObsID (tile), with TMs combined, associated
        with the source - then retrieves the image.
        """
        # Generate one image per ObsID, with TMs combined
        evtool_image(self.src, self._phot_lo_en, self._phot_hi_en)

        try:
            im = self.src.get_images(lo_en=self._phot_lo_en,
                                     hi_en=self._phot_hi_en,
                                     telescope='erass',
                                     inst='combined')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        # We should have retrieved multiple images, so the return should be a list
        assert type(im) == list

        # Cycle through and check some properties of what was returned
        for cur_im in im:
            assert type(cur_im) == Image
            assert cur_im.telescope == 'erass'
            assert cur_im.energy_bounds[0] == self._phot_lo_en
            assert cur_im.energy_bounds[1] == self._phot_hi_en

    @require_esass
    def test_evtool_image_combined_obs(self):
        """
        Generates and retrieves eROSITA images with all available ObsIDs and TMs combined.
        """
        # Generate combined-Obs combined-TM image
        evtool_image(self.src, self._phot_lo_en, self._phot_lo_en, combine_obs=True)

        try:
            im = self.src.get_combined_images(lo_en=self._phot_lo_en,
                                              hi_en=self._phot_hi_en,
                                              telescope='erass')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        # Single combined-obs combined-TM image should have been generated/fetched
        assert type(im) == Image
        assert im.telescope == 'erass'
        assert im.energy_bounds[0] == self._phot_lo_en
        assert im.energy_bounds[1] == self._phot_hi_en

    @require_esass
    def test_expmap(self):
        """
        Test generation and retrieval of eROSITA exposure maps for each Obs associated with
        the source, with TMs combined.
        """
        # Generate new eROSITA exposure maps, one per ObsID, with TMs combined
        expmap(self.src, self._phot_lo_en, self._phot_hi_en)

        try:
            exp = self.src.get_expmaps(lo_en=self._phot_lo_en,
                                       hi_en=self._phot_hi_en,
                                       telescope='erass',
                                       inst='combined')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised.")

        # We know that multiple exposure maps should be retrieved, so check that
        #  the return is a list
        assert type(exp) == list

        # Cycle through and check the exposure maps
        for cur_ex in exp:
            assert cur_ex.telescope == 'erass'
            assert cur_ex.energy_bounds[0] == self._phot_lo_en
            assert cur_ex.energy_bounds[1] == self._phot_hi_en

    @require_esass
    def test_expmap_combined_obs(self):
        expmap(self.src, Quantity(0.5, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        exp = self.src.get_combined_expmaps(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(3, 'keV'),
                                      telescope='erass')

        assert exp.telescope == 'erass'
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
                                 search_distance={'erass': Quantity(3.6, 'deg')})

        test_smp[0].disassociate_obs('erass')
        evtool_image(test_smp)

if __name__ == "__main__":
     unittest.main()



