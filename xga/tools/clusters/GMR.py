#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 30/05/2024, 16:48. Copyright (c) The Contributors

from typing import Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from astropy.cosmology import Cosmology
from astropy.units import Quantity

from xga import DEFAULT_COSMO, NUM_CORES
from xga.samples import ClusterSample

# This just sets the data columns that MUST be present in the sample data passed by the user
LT_REQUIRED_COLS = ['ra', 'dec', 'name', 'redshift']


def gas_mass_radius_pipeline(sample_data: pd.DataFrame, delta: int, baryon_frac: Union[float, np.ndarray], model: str,
                             start_aperture: Quantity, use_peak: bool = False,
                             peak_find_method: str = "hierarchical", convergence_frac: float = 0.1, min_iter: int = 3,
                             max_iter: int = 10, freeze_nh: bool = True, freeze_met: bool = True,
                             freeze_temp: bool = False, start_temp: Quantity = Quantity(3.0, 'keV'),
                             lo_en: Quantity = Quantity(0.3, "keV"), hi_en: Quantity = Quantity(7.9, "keV"),
                             group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                             over_sample: float = None, back_inn_rad_factor: float = 1.05,
                             back_out_rad_factor: float = 1.5, save_samp_results_path: str = None,
                             save_rad_history_path: str = None, cosmo: Cosmology = DEFAULT_COSMO,
                             timeout: Quantity = Quantity(1, 'hr'), num_cores: int = NUM_CORES) \
        -> Tuple[ClusterSample, pd.DataFrame, pd.DataFrame]:

    # I want the sample to be passed in as a DataFrame, so I can easily extract the information I need
    if not isinstance(sample_data, pd.DataFrame):
        raise TypeError("The sample_data argument must be a Pandas DataFrame, with the following columns; "
                        "{}".format(', '.join(LT_REQUIRED_COLS)))

    # Also have to make sure that the required information exists in the dataframe, otherwise obviously this tool
    #  is not going to work
    if not set(LT_REQUIRED_COLS).issubset(sample_data.columns):
        raise KeyError("Not all required columns ({}) are present in the sample_data "
                       "DataFrame.".format(', '.join(LT_REQUIRED_COLS)))

    if (sample_data['name'].str.contains(' ') | sample_data['name'].str.contains('_')).any():
        warn("One or more cluster name has been modified. Empty spaces (' ') are removed, and underscores ('_') are "
             "replaced with hyphens ('-').", stacklevel=2)
        sample_data['name'] = sample_data['name'].apply(lambda x: x.replace(" ", "").replace("_", "-"))

    # I'm going to make sure that the user isn't allowed to request that it not iterate at all
    if min_iter < 2:
        raise ValueError("The minimum number of iterations set by 'min_iter' must be 2 or more.")

    # Also have to make sure the user hasn't something daft like make min_iter larger than max_iter
    if max_iter <= min_iter:
        raise ValueError("The max_iter value ({mai}) is less than or equal to the min_iter value "
                         "({mii}).".format(mai=max_iter, mii=min_iter))

    o_dens = 'r' + str(delta)
    # Overdensity radius argument for the declaration of the sample
    o_dens_arg = {o_dens: start_aperture}

    # Keeps track of the current iteration number
    iter_num = 0

    # Set up the ClusterSample to be used for this process (I did consider setting up a new one each time but that
    #  adds overhead, and I think that this way should work fine).
    samp = ClusterSample(sample_data['ra'].values, sample_data['dec'].values, sample_data['redshift'].values,
                         sample_data['name'].values, use_peak=use_peak, peak_find_method=peak_find_method,
                         clean_obs_threshold=0.7, clean_obs_reg=o_dens, load_fits=False, cosmology=cosmo,
                         back_inn_rad_factor=back_inn_rad_factor, back_out_rad_factor=back_out_rad_factor, **o_dens_arg)

    # As it is possible some clusters in the sample_data dataframe don't actually have X-ray data, we copy
    #  the sample_data and cut it down, so it only contains entries for clusters that were loaded in the sample at the
    #  beginning of this process
    loaded_samp_data = sample_data.copy()
    loaded_samp_data = loaded_samp_data[loaded_samp_data['name'].isin(samp.names)]

    # This is a boolean array of whether the current radius has been accepted or not - starts off False
    acc_rad = np.full(len(samp), False)

    # In this dictionary we're going to keep a record of the radius history for all clusters for each step. The
    #  keys are names, and the initial setup will have the start aperture as the first entry in the list of
    #  radii for each cluster
    rad_hist = {n: [start_aperture.value] for n in samp.names}

    # This will hopefully be eventually replaced with GalaxyCluster instances storing their own overdensity radius
    #  errors, but for now we use a quantity to keep track of the uncertainties we calculate for the radii. We
    #  initially set it up as None because then we can create an appropriately sized quantity after the first run
    #  of spectrum generation, taking into account any systems that failed for some reason
    cur_rad_errs = None

    # This while loop (never thought I'd be using one of them in XGA!) will keep going either until all radii have been
    #  accepted OR until we reach the maximum number  of iterations
    while acc_rad.sum() != len(samp) and iter_num < max_iter:
        pass