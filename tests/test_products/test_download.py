#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 9/29/26, 4:25 PM. Copyright (c) The Contributors.

import os
import tempfile
import unittest

from xga.products import BaseProduct
from xga.products.download import download_products

from . import HTTPS_ROOT, S3_ROOT, TEST_EVTS


class TestDownloadProducts(unittest.TestCase):
    """
    Tests for the download_products batch downloading utility.
    """

    @classmethod
    def setUpClass(cls):
        rel_info_pn = TEST_EVTS["XMM_PN"]
        rel_info_m1 = TEST_EVTS["XMM_MOS1"]
        cls.url_evt_path = os.path.join(HTTPS_ROOT, rel_info_pn["path"])
        cls.s3_evt_path = os.path.join(S3_ROOT, rel_info_pn["path"])
        cls.s3_evt_path_2 = os.path.join(S3_ROOT, rel_info_m1["path"])

        # Shared directory to avoid excessive tempdir creation
        cls.shared_td = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.shared_td.cleanup()

    def test_download_products_url(self) -> None:
        """Tests download_products with HTTP/HTTPS URLs."""
        p1 = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        res = download_products([p1], save_path=self.shared_td.name, disable_progress=True)
        self.assertIn(p1, res)
        self.assertTrue(os.path.exists(res[p1]))
        self.assertTrue(p1.downloaded)
        self.assertEqual(p1.path, res[p1])

    def test_download_products_s3_batch(self) -> None:
        """Tests download_products with multiple S3 URIs in a batch."""
        p1 = BaseProduct(self.s3_evt_path, "", "", "", "", "", check_exists=True)
        p2 = BaseProduct(self.s3_evt_path_2, "", "", "", "", "", check_exists=True)
        res = download_products([p1, p2], save_path=self.shared_td.name, disable_progress=False)
        self.assertIn(p1, res)
        self.assertIn(p2, res)
        self.assertTrue(os.path.exists(res[p1]))
        self.assertTrue(os.path.exists(res[p2]))
        self.assertTrue(p1.downloaded)
        self.assertTrue(p2.downloaded)

    def test_download_products_already_local(self) -> None:
        """Tests download_products skipping already local products."""
        loc_path = os.path.join(self.shared_td.name, "dummy.fits")
        with open(loc_path, "w") as f:
            f.write("dummy")
        p_local = BaseProduct(loc_path, "", "", "", "", "", check_exists=False)
        res = download_products([p_local], save_path=self.shared_td.name)
        self.assertEqual(res[p_local], loc_path)

    def test_download_products_fallback(self) -> None:
        """Tests fallback mechanism when one product in a batch has an invalid remote path."""
        p_good = BaseProduct(self.s3_evt_path, "", "", "", "", "", check_exists=True)
        p_bad = BaseProduct(self.s3_evt_path + ".nonexistent.fits", "", "", "", "", "", check_exists=False)

        res = download_products([p_good, p_bad], save_path=self.shared_td.name, disable_progress=True)
        self.assertIn(p_good, res)
        self.assertIn(p_bad, res)
        self.assertTrue(os.path.exists(res[p_good]))
        self.assertTrue(p_good.downloaded)
        self.assertFalse(p_bad.usable)
        self.assertIsInstance(res[p_bad], Exception)


if __name__ == "__main__":
    unittest.main()
