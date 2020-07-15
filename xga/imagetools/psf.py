#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/07/2020, 10:42. Copyright (c) David J Turner

from typing import List

from astropy.units import Quantity

from xga.sources import BaseSource
from xga.utils import NUM_CORES


def rl_psf(sources: List[BaseSource], iterations: int = 15, lo_en: Quantity = Quantity(0.5, 'keV'),
           hi_en: Quantity = Quantity(2.0, 'keV'), bins: int = 4, num_cores: int = NUM_CORES):
    pass



