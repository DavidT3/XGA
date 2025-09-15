#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 07/07/2025, 13:22. Copyright (c) The Contributors

from typing import Union, List
from warnings import warn

from astropy.units import Quantity

from xga import NUM_CORES
from xga.generate.esass import combine_phot_prod
from xga.generate.multitelescope._common import check_tel_associated
from xga.generate.sas import emosaic
from xga.samples import BaseSample
from xga.sources import BaseSource


def all_telescope_combined_images(sources: Union[BaseSource, BaseSample], lo_en: Quantity, hi_en: Quantity,
                                  telescope: Union[str, List[str]] = None, num_cores: int = NUM_CORES):
    """
    A convenience function for generating individual-telescope combined images (this function does not combine
    data from multiple telescopes). It acts as a simple wrapper for the different generate functions that make
    combined images using telescope-specific backend software.

    :param BaseSource/BaseSample sources: The source/sample for which we will generate individual-telescope
        combined images.
    :param Quantity lo_en: Lower energy bound of the combined images.
    :param Quantity hi_en: Upper energy bound of the combined images.
    :param str/List[str]/None telescope: Telescope name or list of telescope names for which to generate
        images. Default is None, in which case all associated telescopes will be used.
    :param int num_cores: Number of CPU cores to use. Default is set to 90% of available.
    """
    rel_tels = check_tel_associated(sources, telescope)

    if 'xmm' in rel_tels:
        emosaic(sources, 'image', lo_en, hi_en, num_cores=num_cores)

    if 'erosita' in rel_tels or 'erass' in rel_tels:
        combine_phot_prod(sources, 'image', lo_en, hi_en, num_cores=num_cores)

    if 'chandra' in rel_tels:
        warn("Combined images cannot yet be generated from Chandra observations.", stacklevel=2)
        pass


def all_telescope_combined_expmaps(sources: Union[BaseSource, BaseSample], lo_en: Quantity, hi_en: Quantity,
                                   telescope: Union[str, List[str]] = None, num_cores: int = NUM_CORES):
    """
    A convenience function for generating individual-telescope combined exposure maps (this function does
    not combine data from multiple telescopes). It acts as a simple wrapper for the different generate
    functions that make combined exposure maps using telescope-specific backend software.

    :param BaseSource/BaseSample sources: The source/sample for which we will generate individual-telescope
        combined exposure maps.
    :param Quantity lo_en: Lower energy bound of the combined exposure maps.
    :param Quantity hi_en: Upper energy bound of the combined exposure maps.
    :param str/List[str]/None telescope: Telescope name or list of telescope names for which to generate
        exposure maps. Default is None, in which case all associated telescopes will be used.
    :param int num_cores: Number of CPU cores to use. Default is set to 90% of available.
    """

    rel_tels = check_tel_associated(sources, telescope)

    if 'xmm' in rel_tels:
        emosaic(sources, 'expmap', lo_en, hi_en, num_cores=num_cores)

    if 'erosita' in rel_tels or 'erass' in rel_tels:
        combine_phot_prod(sources, 'expmap', lo_en, hi_en, num_cores=num_cores)

    if 'chandra' in rel_tels:
        warn("Combined exposure maps cannot yet be generated from Chandra observations.", stacklevel=2)
        pass