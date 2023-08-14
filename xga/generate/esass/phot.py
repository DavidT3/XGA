from typing import Union

from astropy.units import Quantity

from .. import NUM_CORES
from .run import esass_call
from ...sources import BaseSource
from ...sources.base import NullSource
from ...samples.base import BaseSample

@esass_call
def evtool_image(sources: Union[BaseSource, NullSource, BaseSample], lo_en: Quantity = Quantity(0.2, 'keV'),
                 hi_en: Quantity = Quantity(10, 'keV'), num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A convenient Python wrapper for a configuration of the eSASS evtool command that makes images.
    Images will be generated for every observation associated with every source passed to this function.
    If images in the requested energy band are already associated with the source,
    they will not be generated again.

    :param BaseSource/NullSource/BaseSample sources: A single source object, or a sample of sources.
    :param Quantity lo_en: The lower energy limit for the image, in astropy energy units.
    :param Quantity hi_en: The upper energy limit for the image, in astropy energy units.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    pass
