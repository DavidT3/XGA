#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 17/04/2023, 12:21. Copyright (c) The Contributors
import numpy as np
import pandas as pd
from astropy.cosmology import Cosmology
from astropy.units import Quantity, Unit, UnitConversionError

from xga import DEFAULT_COSMO, NUM_CORES
from xga.products import ScalingRelation
from xga.relations.clusters.RT import arnaud_r500
# This just sets the data columns that MUST be present in the sample data passed by the user
from xga.samples import ClusterSample
from xga.xspec import single_temp_apec

LT_REQUIRED_COLS = ['ra', 'dec', 'name', 'redshift']


def luminosity_temperature_pipeline(sample_data: pd.DataFrame, start_aperture: Quantity, convergence_frac: float = 0.1,
                                    min_iter: int = 3, max_iter: int = 10, rad_temp_rel: ScalingRelation = arnaud_r500,
                                    cosmo: Cosmology = DEFAULT_COSMO, timeout: Quantity = Quantity(1, 'hr'),
                                    num_cores: int = NUM_CORES):

    # I want the sample to be passed in as a DataFrame, so I can easily extract the information I need
    if not isinstance(sample_data, pd.DataFrame):
        raise TypeError("The sample_data argument must be a Pandas DataFrame, with the following columns; "
                        "{}".format(', '.join(LT_REQUIRED_COLS)))

    # Also have to make sure that the required information exists in the dataframe, otherwise obviously this tool
    #  is not going to work
    if not set(LT_REQUIRED_COLS).issubset(sample_data.columns):
        raise KeyError("Not all required columns ({}) are present in the sample_data "
                       "DataFrame.".format(', '.join(LT_REQUIRED_COLS)))

    # A key part of this process is a relation between the temperature we measure, and the overdensity radius. As
    #  scaling relations can be between basically any two parameters, and I want this relation object to be an XGA
    #  scaling relation instance, I need to check some things with the rad_temp_rel passed by the user
    if not isinstance(rad_temp_rel, ScalingRelation):
        raise TypeError("The rad_temp_rel argument requires an XGA ScalingRelation instance.")
    elif not rad_temp_rel.x_unit.is_equivalent(Unit('keV')):
        raise UnitConversionError("This pipeline requires a radius-temperature relation, but the x-unit of the "
                                  "rad_temp_rel relation is {bu}. It cannot be converted to "
                                  "keV.".format(bu=rad_temp_rel.x_unit.to_string()))
    elif not rad_temp_rel.y_unit.is_equivalent(Unit('kpc')):
        raise UnitConversionError("This pipeline requires a radius-temperature relation, but the y-unit of the "
                                  "rad_temp_rel relation is {bu}. It cannot be converted to "
                                  "kpc.".format(bu=rad_temp_rel.y_unit.to_string()))

    # I'm going to make sure that the user isn't allowed to request that it not iterate at all
    if min_iter < 2:
        raise ValueError("The minimum number of iterations set by 'min_iter' must be 2 or more.")

    # Also have to make sure the user hasn't something daft like make min_iter larger than max_iter
    if max_iter <= min_iter:
        raise ValueError("The max_iter value ({mai}) is less than or equal to the min_iter value "
                         "({mii}).".format(mai=max_iter, mii=min_iter))

    # Just want to ensure this dataframe is separate (in memory) from the data that has been passed in
    cur_sample_data = sample_data.copy()
    # And will keep a copy that won't be modified, probably to return alongside a ClusterSample
    all_sample_data = sample_data.copy()

    # Keeps track of the current iteration number
    iter_num = 0

    # Set up the ClusterSample to be used for this process (I did consider setting up a new one each time but that
    #  adds overhead, and I think that this way should work fine).
    samp = ClusterSample(sample_data['ra'].values, sample_data['dec'].values, sample_data['redshift'].values,
                         sample_data['name'].values, r500=start_aperture, use_peak=False, clean_obs_threshold=0.7,
                         load_fits=True)

    # This is a boolean array of whether the current radius has been accepted or not - starts off False
    acc_rad = np.full(len(samp), False)

    # Here we keep track of the clusters which have had some sort of failure during the iterative process
    fail_names = []

    # This while loop (never thought I'd be using one of them in XGA!) will keep going either until all radii have been
    #  accepted OR until we reach the maximum number  of iterations
    while acc_rad.sum() != len(samp) or iter_num < max_iter:

        # We generate and fit spectra for the current value of R500
        single_temp_apec(samp, samp.r500, one_rmf=False, num_cores=num_cores, timeout = timeout)

        # Just reading out the temperatures, not the uncertainties at the moment
        txs = samp.Tx(samp.r500, quality_checks=False)[:, 0]

        # This uses the scaling relation to predict R500 from the measured temperatures
        pr_rs = rad_temp_rel.predict(txs)

        # It is possible that some of these radius entries are going to be NaN - the result of NaN temperature values
        #  passed through the 'predict' method of the scaling relation. As such we identify any NaN results, flag
        #  those clusters by storing their name in the 'fail_names' list, and remove the radii from the pr_rs array
        #  as we're going to do the same for the clusters in the sample
        bad_pr_rs = np.where(np.isnan(pr_rs))[0]
        # pr_rs[bad_pr_rs] = samp.r500[bad_pr_rs]
        pr_rs = samp.r500[bad_pr_rs]
        fail_names += list(samp.names[bad_pr_rs])
        # I am also actually going to remove the clusters with NaN results from the sample - if the NaN was caused
        #  by something like a fit not converging then it's going to keep trying over and over again and that could
        #  slow everything down.
        for name in samp.names[bad_pr_rs]:
            del samp[name]

        # The basis of this method is that we measure a temperature, starting in some user-specified fixed aperture,
        #  and then use that to predict an overdensity radius (something far more useful than a fixed aperture). This
        #  process is repeated until the radius fraction converges to within the user-specified limit.
        # It should also be noted that each cluster is made to iterate at least `min_iter` times, nothing will be
        #  allowed to just accept the first result
        rad_rat = pr_rs / samp.r500
        print(rad_rat)

        # Make a copy of the currently set radius values from the sample - these will then be modified with the
        #  new predicted values if the particular cluster's radius isn't already considered 'accepted' - i.e. it
        #  reached the required convergence in a previous iteration
        new_rads = samp.r500.copy()
        print(new_rads)
        # The clusters which DON'T have previously accepted radii have their radii updated from those predicted from
        #  temperature
        new_rads[~acc_rad] = pr_rs[~acc_rad]
        print(new_rads)
        # Use the new radius value inferred from the temperature + scaling relation and add it to the ClusterSample (or
        #  just re-adding the same value as is already here if that radius has converged and been accepted).
        samp.r500 = new_rads

        # If there have been enough iterations, then we need to start checking whether any of the radii have
        #  converged to within the user-specified fraction. If they have then we accept them and those radii won't
        #  be changed the next time around.
        if iter_num > min_iter:
            acc_rad *= (rad_rat > (1 - convergence_frac)) & (rad_rat < (1 + convergence_frac))
        print(acc_rad.sum(), '\n')
        # Got to increment the counter otherwise the while loop may go on and on forever :O
        iter_num += 1

    return samp







