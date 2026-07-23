#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/23/26, 3:23 PM. Copyright (c) The Contributors.

import os
import unittest

import numpy as np
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel
from astropy.units import Quantity

from xga.imagetools import general_smooth
from xga.products.phot import Image, ExpMap, RateMap
from .. import MISC_OUTPUT_TESTS


class TestGeneralSmoothFunction(unittest.TestCase):
    """A set of tests of XGA's `general_smooth` imagetools function."""

    @classmethod
    def setUpClass(cls):
        """Exists to prepare the XGA Image instance we'll use to test `general_smooth(...)`."""
        # Store the URL in a class attribute just in case
        # Might want to reconsider how much I'm using remote data for tests of product
        #  classes. If files move or the tests are running on a machine with no
        #  internet connection, then there will be artificial failures.
        cls.demo_im_url = ("https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/0843441101/"
                           "PPS/P0843441101M1S001IMAGE_8000.FTZ")
        cls.demo_ex_url = ("https://heasarc.gsfc.nasa.gov/FTP/xmm/data/rev0/0843441101/"
                           "PPS/P0843441101M1S001EXPMP_8000.FTZ")

        # Set up the XGA Image we'll be testing on
        cls.demo_im = Image(cls.demo_im_url, "0843441101", "mos1", "", "",
                            "", Quantity(0.2, 'keV'), Quantity(12., 'keV'),
                            telescope='xmm')

        # Set up the XGA ExpMap we'll use to make a RateMap
        cls.demo_ex = ExpMap(cls.demo_ex_url, "0843441101", "mos1", "", "",
                            "", Quantity(0.2, 'keV'), Quantity(12., 'keV'),
                            telescope='xmm')

        # And then the RateMap itself
        cls.demo_rt = RateMap(cls.demo_im, cls.demo_ex)

    def test_1D_kernel_fail(self):
        """Tests that a 1D Gaussian kernel triggers an exception."""
        # Define a 1D kernel
        cur_kern = Gaussian1DKernel(2)

        # Now we call the smoothing function, which should reject this kernel, as 1D
        #  kernels cannot be applied to 2D imaging data.
        with self.assertRaises(ValueError, msg="The 'general_smooth' function failed to "
                                               "raise an exception when passed a 1D kernel."):
            general_smooth(self.demo_im, cur_kern)

    def test_smooth_image(self):
        """Checks the general_smooth function by running on the demo image with various configurations."""
        # Set up an output directory for this test - visualizations of the images will
        #  be written to disk there as PNGs.
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        # Define a 2D Gaussian Kernel
        cur_kern = Gaussian2DKernel(2)

        # Now start various subtests using different configurations of the general_smooth function
        with self.subTest(check="Default smoothing"):
            # Just passing the image and smoothing kernel
            def_smth_im = general_smooth(self.demo_im, cur_kern)
            # Save view as a PNG
            def_smth_im.save_view(os.path.join(test_out_path, "default_smooth.png"))

        # Using the FFT convolution method
        with self.subTest(check="FFT smoothing"):
            fft_smth_im = general_smooth(self.demo_im, cur_kern, fft=True)
            fft_smth_im.save_view(os.path.join(test_out_path, "FFT_smooth.png"))

        # Switching off kernel renormalization
        with self.subTest(check="Unnormalized kernel smoothing"):
            nonorm_smth_im = general_smooth(self.demo_im, cur_kern, norm_kernel=False)
            nonorm_smth_im.save_view(os.path.join(test_out_path, "nonorm_smooth.png"))

        # Switching off kernel renormalization
        with self.subTest(check="Unnormalized kernel FFT smoothing"):
            nonorm_fft_smth_im = general_smooth(self.demo_im, cur_kern, fft=True, norm_kernel=False)
            nonorm_fft_smth_im.save_view(os.path.join(test_out_path, "nonorm_fft_smooth.png"))

        # Applying a mask and smoothing
        with self.subTest(check="Masked smoothing"):
            mask = np.zeros(self.demo_im.data.shape, dtype=int)
            mask[270:378, 270:378] = 1
            masked_smth_im = general_smooth(self.demo_im, cur_kern, mask=mask)
            masked_smth_im.save_view(os.path.join(test_out_path, "masked_smooth.png"))

    def test_smooth_ratemap_image(self):
        """
        Checks the general_smooth function by running on the demo ratemap, using the
        ratemap_smooth_im=True method.
        """
        # Set up an output directory for this test - visualizations of the ratemaps will
        #  be written to disk there as PNGs.
        test_out_path = os.path.join(MISC_OUTPUT_TESTS, self.id())
        os.makedirs(test_out_path, exist_ok=True)

        # Define a 2D Gaussian Kernel
        cur_kern = Gaussian2DKernel(2)

        # Now start various subtests using different configurations of the general_smooth function
        with self.subTest(check="Default smoothing ratemap [im method]"):
            # Just passing the image and smoothing kernel
            def_smth_rt = general_smooth(self.demo_rt, cur_kern, ratemap_smooth_im=True)
            # Save view as a PNG
            def_smth_rt.save_view(os.path.join(test_out_path, "default_smooth_rt_im_method.png"))

        # Using the FFT convolution method
        with self.subTest(check="FFT smoothing ratemap [im method]"):
            fft_smth_rt = general_smooth(self.demo_rt, cur_kern, fft=True, ratemap_smooth_im=True)
            fft_smth_rt.save_view(os.path.join(test_out_path, "FFT_smooth_rt_im_method.png"))

        # Switching off kernel renormalization
        with self.subTest(check="Unnormalized kernel smoothing ratemap [im method]"):
            nonorm_smth_rt = general_smooth(self.demo_rt, cur_kern, norm_kernel=False, ratemap_smooth_im=True)
            nonorm_smth_rt.save_view(os.path.join(test_out_path, "nonorm_smooth_rt_im_method.png"))

        # Switching off kernel renormalization
        with self.subTest(check="Unnormalized kernel FFT smoothing ratemap [im method]"):
            nonorm_fft_smth_rt = general_smooth(self.demo_rt, cur_kern, fft=True, norm_kernel=False,
                                                ratemap_smooth_im=True)
            nonorm_fft_smth_rt.save_view(os.path.join(test_out_path, "nonorm_fft_smooth_rt_im_method.png"))

        # Applying a mask and smoothing
        with self.subTest(check="Masked smoothing ratemap [im method]"):
            mask = np.zeros(self.demo_rt.data.shape, dtype=int)
            mask[270:378, 270:378] = 1
            masked_smth_rt = general_smooth(self.demo_rt, cur_kern, mask=mask, ratemap_smooth_im=True)
            masked_smth_rt.save_view(os.path.join(test_out_path, "masked_smooth_rt_im_method.png"))

if __name__ == "__main__":
    unittest.main()