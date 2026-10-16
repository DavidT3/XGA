#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 9/29/26, 4:25 PM. Copyright (c) The Contributors.

import os
import shutil
import tempfile
import unittest

from astropy.io import fits
from fsspec.core import url_to_fs

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

    @classmethod
    def tearDownClass(cls):
        # We clean up the local file we created for the existence checks.
        test_ext_data_dir = os.path.join(EXTERNAL_TEST_DATA_PATH, "TestBaseProductFileExists")
        if os.path.exists(test_ext_data_dir):
            shutil.rmtree(test_ext_data_dir)

    def test_local_exist(self) -> None:
        """Tests whether defining a BaseProduct with a local file and check_exists=True will work."""
        cur_test_evt = BaseProduct(self.loc_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(
            cur_test_evt.usable,
            True,
            f"Expected usable to be True, instead False. Reason given - {cur_test_evt.not_usable_reasons}.",
        )

    def test_url_exist(self) -> None:
        """Tests whether defining a BaseProduct with an HTTP/HTTPS path and check_exists=True will work."""
        cur_test_evt = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(
            cur_test_evt.usable,
            True,
            f"Expected usable to be True, instead False. Reason given - {cur_test_evt.not_usable_reasons}.",
        )

    def test_s3_uri_exist(self) -> None:
        """Tests whether defining a BaseProduct with an S3 URI path and check_exists=True will work."""
        cur_test_evt = BaseProduct(self.s3_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(
            cur_test_evt.usable,
            True,
            f"Expected usable to be True, instead False. Reason given - {cur_test_evt.not_usable_reasons}.",
        )

    def test_local_not_exist(self) -> None:
        """Tests whether defining a BaseProduct with a fake local file path and check_exists=True will behave properly."""
        cur_broken_test_evt = BaseProduct(self.broken_loc_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_broken_test_evt.usable, False, "Expected usable to be False, instead True.")
        self.assertEqual(
            cur_broken_test_evt.not_usable_reasons,
            ["ProductPathDoesNotExist"],
            f"Expected not_usable_reasons to be ['ProductPathDoesNotExist'], instead {cur_broken_test_evt.not_usable_reasons}.",
        )

    def test_url_not_exist(self) -> None:
        """Tests whether defining a BaseProduct with a fake HTTP/HTTPS path and check_exists=True will behave properly."""
        cur_broken_test_evt = BaseProduct(self.broken_url_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_broken_test_evt.usable, False, "Expected usable to be False, instead True.")
        self.assertEqual(
            cur_broken_test_evt.not_usable_reasons,
            ["ProductPathDoesNotExist"],
            f"Expected not_usable_reasons to be ['ProductPathDoesNotExist'], instead {cur_broken_test_evt.not_usable_reasons}.",
        )

    def test_s3_uri_not_exist(self) -> None:
        """Tests whether defining a BaseProduct with a fake S3 URI path and check_exists=True will behave properly."""
        cur_broken_test_evt = BaseProduct(self.broken_s3_evt_path, "", "", "", "", "", check_exists=True)
        self.assertEqual(cur_broken_test_evt.usable, False, "Expected usable to be False, instead True.")
        self.assertEqual(
            cur_broken_test_evt.not_usable_reasons,
            ["ProductPathDoesNotExist"],
            f"Expected not_usable_reasons to be ['ProductPathDoesNotExist'], instead {cur_broken_test_evt.not_usable_reasons}.",
        )


class TestBaseProductDownload(unittest.TestCase):
    """
    Tests for the BaseProduct.download() method across URL and S3 sources.
    """

    @classmethod
    def setUpClass(cls):
        rel_info = TEST_EVTS["XMM_PN"]
        cls.url_evt_path = os.path.join(HTTPS_ROOT, rel_info["path"])
        cls.s3_evt_path = os.path.join(S3_ROOT, rel_info["path"])

        # We set up a shared temporary directory for these tests, to avoid creating many of them
        cls.shared_td = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.shared_td.cleanup()

    def test_download_url(self) -> None:
        """Tests BaseProduct.download with an HTTP/HTTPS URL."""
        prod = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        prod.download(save_path=self.shared_td.name)
        self.assertTrue(prod.downloaded)
        self.assertTrue(os.path.exists(prod.path))

    def test_download_s3(self) -> None:
        """Tests BaseProduct.download with an S3 URI."""
        prod = BaseProduct(self.s3_evt_path, "", "", "", "", "", check_exists=True)
        prod.download(save_path=self.shared_td.name)
        self.assertTrue(prod.downloaded)
        self.assertTrue(os.path.exists(prod.path))

    def test_download_with_explicit_fs(self) -> None:
        """Tests BaseProduct.download when a remote_file_sys is explicitly passed."""
        fs, _ = url_to_fs(self.s3_evt_path, anon=True)
        prod = BaseProduct(self.s3_evt_path, "", "", "", "", "", check_exists=True)
        prod.download(save_path=self.shared_td.name, remote_file_sys=fs)
        self.assertTrue(prod.downloaded)
        self.assertTrue(os.path.exists(prod.path))

    def test_directory_detection(self) -> None:
        """Tests that the download method correctly identifies directories even with dots in the name."""
        # We create a directory name that has a dot in it, to test the robust detection logic.
        # We add a trailing slash to indicate it's a directory
        dir_name = os.path.join(self.shared_td.name, "test_v1.0") + os.sep
        # We don't create the directory yet, as we want to see if the download method will
        prod = BaseProduct(self.url_evt_path, "", "", "", "", "", check_exists=True)
        prod.download(save_path=dir_name)

        self.assertTrue(os.path.isdir(dir_name))
        self.assertTrue(os.path.exists(os.path.join(dir_name, os.path.basename(self.url_evt_path))))
        self.assertTrue(prod.downloaded)


if __name__ == "__main__":
    unittest.main()
