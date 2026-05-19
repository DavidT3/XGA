#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/19/26, 3:30 PM. Copyright (c) The Contributors.
import numpy as np
import pandas as pd
from astropy.units import Quantity

from xga.sources import GalaxyCluster
from .source_info import SRC_INFO, SUPP_SRC_INFO, EXPECTED_ERO_OBS, EXPECTED_XMM_OBS

# Making a df to make a sample from
column_names = ['name', 'ra', 'dec', 'z', 'r500']
cluster_data = np.array([[SRC_INFO['name'], SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], 500],
                         [SUPP_SRC_INFO['name'], SUPP_SRC_INFO['ra'], SUPP_SRC_INFO['dec'], SUPP_SRC_INFO['z'], 500]])

CLUSTER_SMP = pd.DataFrame(data=cluster_data, columns=column_names)
CLUSTER_SMP[['ra', 'dec', 'z', 'r500']] = CLUSTER_SMP[['ra', 'dec', 'z', 'r500']].astype(float)


# We use a factory pattern to provide the test sources, this is because they are expensive to instantiate
#  and we don't want to do it at import time, as the configuration might not be set up yet.
_CACHED_SOURCES = {}

def get_test_source(telescope: str = 'all', shared: bool = True, load_fits: bool = True) -> GalaxyCluster:
    """
    A factory function to provide test sources. This is used to avoid instantiating them at import time.

    :param str telescope: The telescope for which we want a source. Options are 'all', 'xmm', 'erass'.
    :param bool shared: Whether to return a shared (cached) instance or a fresh one.
    :param bool load_fits: Only applied if shared=False, and controls whether the fresh GalaxyCluster
        instance is set to load existing XSPEC fit information from disk.
    :return: The requested source.
    :rtype: GalaxyCluster
    """
    global _CACHED_SOURCES

    # Reset the load_fits argument to True if shared=True
    load_fits = True if shared else load_fits

    # If a shared instance is requested, and we have one cached, return it
    if shared and telescope in _CACHED_SOURCES:
        return _CACHED_SOURCES[telescope]

    # Otherwise we set up the requested source instance for testing.
    if telescope == 'all':
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            search_distance={'erass': Quantity(3.6, 'deg')},
                            load_profiles=False, load_fits=load_fits)
    elif telescope == 'xmm':
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            telescope='xmm', load_profiles=False, load_fits=load_fits)
    elif telescope == 'erass' or telescope == 'erosita':
        src = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                            name=SRC_INFO['name'], use_peak=False,
                            telescope='erass',
                            search_distance={'erass': Quantity(3.6, 'deg')},
                            load_profiles=False, load_fits=load_fits)
    else:
        raise ValueError(f"Unknown mission name: {telescope}")

    # If a shared instance was requested, cache it
    if shared:
        _CACHED_SOURCES[telescope] = src

    return src
