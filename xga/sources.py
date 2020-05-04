#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/04/2020, 21:44. Copyright (c) David J Turner

import numpy as np
from xga.sourcetools import nhlookup


class BaseSource:
    def __init__(self, ra, dec, redshift):
        self.ra_dec = np.array(ra, dec)
        self.redshift = redshift


class ExtendedSource(BaseSource):
    def __init__(self):
        pass


class PointSource(BaseSource):
    def __init__(self):
        pass


class GalaxyCluster(ExtendedSource):
    def __init__(self):
        pass






