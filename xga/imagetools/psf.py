#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/07/2020, 11:37. Copyright (c) David J Turner

from typing import List

from astropy.units import Quantity

from xga.sas import evselect_image, psfgen
from xga.sources import BaseSource
from xga.utils import NUM_CORES


def rl_psf(sources: List[BaseSource], iterations: int = 15, psf_model: str = "ELLBETA",
           lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV'),
           bins: int = 4, num_cores: int = NUM_CORES):
    """
    An implementation of the Richardson-Lucy (doi:10.1364/JOSA.62.000055) PSF deconvolution algorithm that
    also takes into account the spatial variance of the XMM Newton PSF. The sources passed into this
    function will have all images matching the passed energy range deconvolved, the image objects will have the
    result stored in them alongside the original data, and a combined image will be generated. I view this
    method as quite crude, but it does seem to work, and I may implement a more intelligent way of doing
    PSF deconvolutions later.
    :param List[BaseSource] sources: A single source object, or list of source objects.
    :param int iterations: The number of deconvolution iterations performed by the Richardson-Lucy algorithm.
    :param str psf_model: Which model of PSF should be used for this deconvolution. The default is ELLBETA,
    the best available.
    :param Quantity lo_en: The lower energy bound of the images to be deconvolved.
    :param Quantity hi_en: The upper energy bound of the images to be deconvolved.
    :param int bins: Number of bins that the X and Y axes will be divided into when generating a PSFGrid.
    :param int num_cores: The number of cores to use (if running locally), the default is set to 90%
    of available cores in your system.
    """
    # These just make sure that the images to be deconvolved and the PSFs to deconvolve them
    #  with have actually been generated. If they are already associated with the sources then this won't
    #  do anything.
    evselect_image(sources, lo_en, hi_en, num_cores=num_cores)
    psfgen(sources, bins, psf_model, num_cores)

    print("YA BOI")



