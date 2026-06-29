#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/10/26, 7:14 PM. Copyright (c) The Contributors.

import unittest

from xga.exceptions import TelescopeNotAssociatedError, NoProductAvailableError
from xga.generate.ciao.spec import specextract_spectrum
from xga.products import Spectrum
from .. import get_test_source
from ..utils import require_ciao


class TestCiaoSpecFuncs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = get_test_source('all')

    @require_ciao
    def test_specextract_spectrum_no_tel_error(self):
        """
        Testing that TelescopeNotAssociatedError is raised when Chandra isn't associated.
        """
        with self.assertRaises(TelescopeNotAssociatedError):
            specextract_spectrum(self.src, 'r500')

    @require_ciao
    def test_specextract_spectrum_if_available(self):
        """
        Test specextract_spectrum if Chandra data is available. Will skip if no Chandra observations.
        """
        # Check if Chandra is actually available for this source
        if 'chandra' not in self.src.obs_ids or len(self.src.obs_ids.get('chandra', [])) == 0:
            self.skipTest("No Chandra observations available for test source")

        # If we get here, Chandra data exists - try to generate spectra
        specextract_spectrum(self.src, 'r500')

        # Try to retrieve the generated products
        try:
            spec = self.src.get_spectra('r500', telescope='chandra')
        except NoProductAvailableError:
            self.fail("NoProductAvailableError raised when retrieving Chandra spectra.")

        if isinstance(spec, list):
            for s in spec:
                assert s.telescope == 'chandra'
                assert isinstance(s, Spectrum)
        else:
            assert spec.telescope == 'chandra'
            assert isinstance(spec, Spectrum)


if __name__ == "__main__":
    unittest.main()



