#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/22/26, 9:08 AM. Copyright (c) The Contributors.
"""Shared test fixtures, paths, and factory functions for the XGA test suite."""

import os

import numpy as np
import pandas as pd
from astropy.units import Quantity

from xga import OUTPUT
from xga.samples import ClusterSample
from xga.sources import GalaxyCluster

from .source_info import (
    EXPECTED_ERO_OBS as EXPECTED_ERO_OBS,
)
from .source_info import (
    EXPECTED_XMM_OBS as EXPECTED_XMM_OBS,
)
from .source_info import (
    SRC_INFO as SRC_INFO,
)
from .source_info import (
    SUPP_SRC_INFO as SUPP_SRC_INFO,
)

MISC_OUTPUT_TESTS = os.path.join(OUTPUT, "test_figures")
os.makedirs(MISC_OUTPUT_TESTS, exist_ok=True)

EXTERNAL_TEST_DATA_PATH = os.path.join(OUTPUT, "test_data", "external_data")
os.makedirs(EXTERNAL_TEST_DATA_PATH, exist_ok=True)

# Making a df to make a sample from
column_names = ["name", "ra", "dec", "z", "r500"]
cluster_data = np.array(
    [
        [SRC_INFO["name"], SRC_INFO["ra"], SRC_INFO["dec"], SRC_INFO["z"], 500],
        [SUPP_SRC_INFO["name"], SUPP_SRC_INFO["ra"], SUPP_SRC_INFO["dec"], SUPP_SRC_INFO["z"], 500],
    ]
)

CLUSTER_SMP = pd.DataFrame(data=cluster_data, columns=column_names)
CLUSTER_SMP[["ra", "dec", "z", "r500"]] = CLUSTER_SMP[["ra", "dec", "z", "r500"]].astype(float)


# We use a factory pattern to provide the test sources, this is because they are expensive to instantiate
#  and we don't want to do it at import time, as the configuration might not be set up yet.
_CACHED_SOURCES: dict[str, GalaxyCluster] = {}
_CACHED_SAMPLES: dict[str, ClusterSample] = {}


def get_test_source(telescope: str = "all", shared: bool = True, load_fits: bool = True) -> GalaxyCluster:
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
    ra_val = float(SRC_INFO["ra"])  # type: ignore[arg-type]
    dec_val = float(SRC_INFO["dec"])  # type: ignore[arg-type]
    z_val = float(SRC_INFO["z"])  # type: ignore[arg-type]
    name_val = str(SRC_INFO["name"])
    if telescope == "all":
        src = GalaxyCluster(
            ra_val,
            dec_val,
            z_val,
            r500=Quantity(500, "kpc"),
            name=name_val,
            use_peak=False,
            search_distance={"erass": Quantity(3.6, "deg")},
            load_profiles=False,
            load_fits=load_fits,
        )
    elif telescope == "xmm":
        src = GalaxyCluster(
            ra_val,
            dec_val,
            z_val,
            r500=Quantity(500, "kpc"),
            name=name_val,
            use_peak=False,
            telescope="xmm",
            load_profiles=False,
            load_fits=load_fits,
        )
    elif telescope in ["erass", "erosita"]:
        src = GalaxyCluster(
            ra_val,
            dec_val,
            z_val,
            r500=Quantity(500, "kpc"),
            name=name_val,
            use_peak=False,
            telescope="erass",
            search_distance={"erass": Quantity(3.6, "deg")},
            load_profiles=False,
            load_fits=load_fits,
        )
    else:
        raise ValueError(f"Unknown mission name: {telescope}")

    # If a shared instance was requested, cache it
    if shared:
        _CACHED_SOURCES[telescope] = src

    return src


def get_test_sample(telescope: str = "all", shared: bool = True, load_fits: bool = True) -> ClusterSample:
    """
    A factory function to provide test ClusterSample instances without instantiating them at import time.

    :param str telescope: The telescope for which we want a sample. Options are 'all', 'xmm', 'erass'.
    :param bool shared: Whether to return a shared (cached) instance or a fresh one.
    :param bool load_fits: Only applied if shared=False, and controls whether the fresh ClusterSample
        instance is set to load existing XSPEC fit information from disk.
    :return: The requested ClusterSample instance.
    :rtype: ClusterSample
    """
    global _CACHED_SAMPLES

    # Reset the load_fits argument to True if shared=True
    load_fits = True if shared else load_fits

    # If a shared instance is requested, and we have one cached, return it
    if shared and telescope in _CACHED_SAMPLES:
        return _CACHED_SAMPLES[telescope]

    # Otherwise we set up the requested ClusterSample instance for testing
    if telescope == "all":
        smp = ClusterSample(
            CLUSTER_SMP["ra"].to_numpy(),
            CLUSTER_SMP["dec"].to_numpy(),
            CLUSTER_SMP["z"].to_numpy(),
            CLUSTER_SMP["name"].to_numpy(),
            r500=Quantity(CLUSTER_SMP["r500"].to_numpy(), "kpc"),
            use_peak=False,
            search_distance={"erass": Quantity(3.6, "deg")},
            load_fits=load_fits,
        )
    elif telescope == "xmm":
        smp = ClusterSample(
            CLUSTER_SMP["ra"].to_numpy(),
            CLUSTER_SMP["dec"].to_numpy(),
            CLUSTER_SMP["z"].to_numpy(),
            CLUSTER_SMP["name"].to_numpy(),
            r500=Quantity(CLUSTER_SMP["r500"].to_numpy(), "kpc"),
            use_peak=False,
            telescope="xmm",
            load_fits=load_fits,
        )
    elif telescope in ["erass", "erosita"]:
        smp = ClusterSample(
            CLUSTER_SMP["ra"].to_numpy(),
            CLUSTER_SMP["dec"].to_numpy(),
            CLUSTER_SMP["z"].to_numpy(),
            CLUSTER_SMP["name"].to_numpy(),
            r500=Quantity(CLUSTER_SMP["r500"].to_numpy(), "kpc"),
            use_peak=False,
            telescope="erass",
            search_distance={"erass": Quantity(3.6, "deg")},
            load_fits=load_fits,
        )
    else:
        raise ValueError(f"Unknown mission name: {telescope}")

    if shared:
        _CACHED_SAMPLES[telescope] = smp

    return smp
