#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 9/30/26, 10:29 AM. Copyright (c) The Contributors.

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

    def test_download_products_non_existent_dir_with_sep(self) -> None:
        """Tests download_products when the destination directory doesn't exist yet but has a trailing separator."""
        new_dir = os.path.join(self.shared_td.name, "new_dir_sep") + os.sep
        p1 = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        res = download_products([p1], save_path=new_dir, disable_progress=True)
        self.assertTrue(os.path.isdir(new_dir))
        self.assertTrue(os.path.exists(res[p1]))
        self.assertEqual(os.path.dirname(res[p1]) + os.sep, new_dir)

    def test_download_products_multiple_to_non_existent_dir(self) -> None:
        """Tests that multiple products to a non-existent dir (with sep) don't overwrite each other."""
        new_dir = os.path.join(self.shared_td.name, "multi_dir") + os.sep
        p1 = BaseProduct(self.s3_evt_path, "", "", "", "", "", check_exists=True)
        p2 = BaseProduct(self.s3_evt_path_2, "", "", "", "", "", check_exists=True)
        res = download_products([p1, p2], save_path=new_dir, disable_progress=True)

        self.assertTrue(os.path.isdir(new_dir))
        self.assertTrue(os.path.exists(res[p1]))
        self.assertTrue(os.path.exists(res[p2]))
        self.assertNotEqual(res[p1], res[p2])
        self.assertEqual(os.path.dirname(res[p1]) + os.sep, new_dir)
        self.assertEqual(os.path.dirname(res[p2]) + os.sep, new_dir)

    def test_download_products_redundant_skip(self) -> None:
        """Tests that download_products skips already downloaded local files and updates properties."""
        p1 = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        # First download to establish the file
        download_products([p1], save_path=self.shared_td.name, disable_progress=True)
        self.assertTrue(p1.downloaded)
        loc_path = p1.path

        # Create a fresh product object pointing at the same remote, but not yet 'downloaded'
        p2 = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        self.assertFalse(p2.downloaded)

        # Run download_products again - it should find the existing file and skip network call
        res = download_products([p2], save_path=self.shared_td.name, disable_progress=True)
        self.assertIn(p2, res)
        self.assertTrue(p2.downloaded)
        self.assertTrue(p2.local_file)
        self.assertEqual(p2.path, loc_path)
        self.assertEqual(res[p2], loc_path)

    def test_download_products_redownload(self) -> None:
        """Tests the redownload argument in download_products."""
        p1 = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        # First download to establish the file
        download_products([p1], save_path=self.shared_td.name, disable_progress=True)
        self.assertTrue(p1.downloaded)
        loc_path = p1.path

        # Create a fresh product object pointing at the same remote
        p2 = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)

        # Run with redownload=True
        res = download_products([p2], save_path=self.shared_td.name, redownload=True, disable_progress=True)
        self.assertIn(p2, res)
        self.assertTrue(p2.downloaded)
        self.assertEqual(p2.path, loc_path)


if __name__ == "__main__":
    unittest.main()
