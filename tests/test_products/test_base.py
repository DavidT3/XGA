#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 9/25/26, 5:46 PM. Copyright (c) The Contributors.

import os
import unittest

from astropy.io import fits

from xga.products import BaseProduct

from .. import EXTERNAL_TEST_DATA_PATH
from . import HTTPS_ROOT, S3_ROOT, TEST_EVTS


class TestBaseProductFileExists(unittest.TestCase):
    """
    Tests that the BaseProduct class' file-exists checking facility works across local and remote files.
    """

    @classmethod
    def setUpClass(cls):
        rel_info = TEST_EVTS["XMM_PN"]
        rel_url = os.path.join(HTTPS_ROOT, rel_info["path"])

        # We download and decompress an events list to make sure there is a local file to try out the
        #  file exists checks on.
        test_ext_data_dir = os.path.join(EXTERNAL_TEST_DATA_PATH, "TestBaseProductFileExists")
        os.makedirs(test_ext_data_dir, exist_ok=True)

        cls.loc_evt_path = os.path.join(test_ext_data_dir, os.path.basename(rel_url))
        if not os.path.exists(cls.loc_evt_path):
            with fits.open(rel_url) as evto:
                evto.writeto(cls.loc_evt_path)

        # Now we define the URI and URL versions of the same file path
        #  In fact, we already have the URL
        cls.url_evt_path = rel_url
        #  And the S3 URI version is also simple to construct.
        cls.s3_evt_path = os.path.join(S3_ROOT, rel_info["path"])

        # We also define local, URL, and S3 paths that are deliberately broken, to check that those
        #  products are marked as unusable.
        cls.broken_loc_evt_path = cls.loc_evt_path + ".broken.gz"
        cls.broken_url_evt_path = cls.url_evt_path + ".broken.gz"
        cls.broken_s3_evt_path = cls.s3_evt_path + ".broken.gz"

    def test_local_exist(self) -> None:
        """Tests whether defining a BaseProduct with a local file and check_exists=True will work."""
        cur_test_evt = BaseProduct(self.loc_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_test_evt.usable, True)

    def test_url_exist(self) -> None:
        """Tests whether defining a BaseProduct with an HTTP/HTTPS path and check_exists=True will work."""
        cur_test_evt = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_test_evt.usable, True)

    def test_s3_uri_exist(self) -> None:
        """Tests whether defining a BaseProduct with an S3 URI path and check_exists=True will work."""
        cur_test_evt = BaseProduct(self.s3_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_test_evt.usable, True)

    def test_local_not_exist(self) -> None:
        """Tests whether defining a BaseProduct with a fake local file path and check_exists=True will behave properly."""
        cur_broken_test_evt = BaseProduct(self.broken_loc_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_broken_test_evt.usable, False)
        self.assertEqual(cur_broken_test_evt.not_usable_reasons, ["ProductPathDoesNotExist"])

    def test_url_not_exist(self) -> None:
        """Tests whether defining a BaseProduct with a fake HTTP/HTTPS path and check_exists=True will behave properly."""
        cur_broken_test_evt = BaseProduct(self.broken_url_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_broken_test_evt.usable, False)
        self.assertEqual(cur_broken_test_evt.not_usable_reasons, ["ProductPathDoesNotExist"])

    def test_s3_uri_not_exist(self) -> None:
        """Tests whether defining a BaseProduct with a fake S3 URI path and check_exists=True will behave properly."""
        cur_broken_test_evt = BaseProduct(self.broken_s3_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_broken_test_evt.usable, False)
        self.assertEqual(cur_broken_test_evt.not_usable_reasons, ["ProductPathDoesNotExist"])


if __name__ == "__main__":
    unittest.main()
