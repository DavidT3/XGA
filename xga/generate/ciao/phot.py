#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by Ray Wang (wangru46@msu.edu) 21/11/2024, 12:08. Copyright (c) The Contributors

import os
from random import randint
from shutil import rmtree
from typing import Union

import numpy as np
from astropy.units import Quantity, UnitConversionError

# from .misc import evtool_combine_evts
from .run import ciao_call
from ... import OUTPUT, NUM_CORES
from ...exceptions import TelescopeNotAssociatedError, NoProductAvailableError
from ...products import BaseProduct
from ...products.misc import EventList
from ...samples.base import BaseSample
from ...sources import BaseSource
from ...sources.base import NullSource