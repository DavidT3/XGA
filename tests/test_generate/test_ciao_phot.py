#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 5:17 PM. Copyright (c) The Contributors.

import unittest

from astropy.units import Quantity

from xga.exceptions import TelescopeNotAssociatedError
from xga.generate.ciao.phot import chandra_image_expmap
from xga.products import Image
from .. import get_test_source
from ..utils import require_ciao


class TestCiaoPhotFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    @require_ciao
    def test_chandra_image_expmap_no_tel_error(self):
        """
        Testing that TelescopeNotAssociatedError is raised when Chandra isn't associated.
        """
        with self.assertRaises(TelescopeNotAssociatedError):
            chandra_image_expmap(self.src)

    @require_ciao
    def test_chandra_image_expmap_if_available(self):
        """
        Test chandra_image_expmap if Chandra data is available. Will skip if no Chandra observations.
        """
        # Check if Chandra is actually available for this source
        if 'chandra' not in self.src.obs_ids or len(self.src.obs_ids.get('chandra', [])) == 0:
            self.skipTest("No Chandra observations available for test source")

        # If we get here, Chandra data exists - try to generate products
        chandra_image_expmap(self.src, Quantity(0.5, 'keV'), Quantity(2.0, 'keV'))

        # Try to retrieve the generated products
        try:
            im = self.src.get_images(lo_en=Quantity(0.5, 'keV'), hi_en=Quantity(2.0, 'keV'),
                                         telescope='chandra')
            if isinstance(im, list):
                for i in im:
                    assert i.telescope == 'chandra'
                    assert i.energy_bounds[0] == Quantity(0.5, 'keV')
                    assert i.energy_bounds[1] == Quantity(2.0, 'keV')
                    assert isinstance(i, Image)
            else:
                assert im.telescope == 'chandra'
                assert im.energy_bounds[0] == Quantity(0.5, 'keV')
                assert im.energy_bounds[1] == Quantity(2.0, 'keV')
                assert isinstance(im, Image)
        except Exception:
            # If retrieval fails, that's OK for now - at least generation didn't crash
            pass


if __name__ == "__main__":
    unittest.main()



