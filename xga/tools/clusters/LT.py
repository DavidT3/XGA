#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 18/04/2023, 22:35. Copyright (c) The Contributors
from warnings import warn

import numpy as np
import pandas as pd
from astropy.cosmology import Cosmology
from astropy.units import Quantity, Unit, UnitConversionError

from xga import DEFAULT_COSMO, NUM_CORES
from xga.exceptions import ModelNotAssociatedError
from xga.products import ScalingRelation
from xga.relations.clusters.RT import arnaud_r500
# This just sets the data columns that MUST be present in the sample data passed by the user
from xga.samples import ClusterSample
from xga.xspec import single_temp_apec

LT_REQUIRED_COLS = ['ra', 'dec', 'name', 'redshift']


def luminosity_temperature_pipeline(sample_data: pd.DataFrame, start_aperture: Quantity, convergence_frac: float = 0.1,
                                    min_iter: int = 3, max_iter: int = 10, rad_temp_rel: ScalingRelation = arnaud_r500,
                                    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                                    core_excised: bool = False, cosmo: Cosmology = DEFAULT_COSMO,
                                    timeout: Quantity = Quantity(1, 'hr'), num_cores: int = NUM_CORES,
                                    save_samp_results_path: str = None, save_rad_history_path: str = None):
    """


    :param pd.DataFrame sample_data:
    :param Quantity start_aperture:
    :param float convergence_frac:
    :param int min_iter:
    :param int max_iter:
    :param ScalingRelation rad_temp_rel:
    :param Quantity lum_en:
    :param bool core_excised:
    :param Cosmology cosmo:
    :param Quantity timeout:
    :param int num_cores:
    :param str save_samp_results_path:
    :param str save_rad_history_path:
    :return:
    :rtype:
    """
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
             "replaced with hyphens ('-').")
        sample_data['name'] = sample_data['name'].apply(lambda x: x.replace(" ", "").replace("_", "-"))

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

    # Trying to determine the targeted overdensity based on the name of the scaling relation y-axis label
    y_name = rad_temp_rel.y_name.lower()
    if 'r' in y_name and '2500' in y_name:
        o_dens = 'r2500'
    elif 'r' in y_name and '500' in y_name:
        o_dens = 'r500'
    elif 'r' in y_name and '200' in y_name:
        o_dens = 'r200'
    else:
        raise ValueError("The y-axis label of the scaling relation ({ya}) does not seem to contain 2500, 500, or "
                         "200; it has not been possible to determine the overdensity.".format(ya=rad_temp_rel.y_name))

    # Overdensity radius argument for the declaration of the sample
    o_dens_arg = {o_dens: start_aperture}

    # Just a little warning to a user who may have made a silly decision
    if core_excised and o_dens == 'r2500':
        warn("You may not measure reliable core-excised results when iterating on R2500 - the radii can be small "
             " enough that multiplying by 0.15 for an inner radius will result in too small of a "
             "radius.", stacklevel=2)

    # Just want to ensure this dataframe is separate (in memory) from the data that has been passed in
    all_sample_data = sample_data.copy()

    # Keeps track of the current iteration number
    iter_num = 0

    # Set up the ClusterSample to be used for this process (I did consider setting up a new one each time but that
    #  adds overhead, and I think that this way should work fine).
    samp = ClusterSample(sample_data['ra'].values, sample_data['dec'].values, sample_data['redshift'].values,
                         sample_data['name'].values, use_peak=False, clean_obs_threshold=0.7, load_fits=True,
                         cosmology=cosmo, **o_dens_arg)

    # This is a boolean array of whether the current radius has been accepted or not - starts off False
    acc_rad = np.full(len(samp), False)

    # In this dictionary we're going to keep a record of the radius history for all clusters for each step. The
    #  keys are names, and the initial setup will have the start aperture as the first entry in the list of
    #  radii for each cluster
    rad_hist = {n: [start_aperture] for n in samp.names}

    # This while loop (never thought I'd be using one of them in XGA!) will keep going either until all radii have been
    #  accepted OR until we reach the maximum number  of iterations
    while acc_rad.sum() != len(samp) and iter_num < max_iter:

        # We generate and fit spectra for the current value of the overdensity radius
        single_temp_apec(samp, samp.get_radius(o_dens), one_rmf=False, num_cores=num_cores, timeout=timeout,
                         lum_en=lum_en)

        # Just reading out the temperatures, not the uncertainties at the moment
        txs = samp.Tx(samp.get_radius(o_dens), quality_checks=False)[:, 0]

        # This uses the scaling relation to predict the overdensity radius from the measured temperatures
        pr_rs = rad_temp_rel.predict(txs, samp.redshifts, samp.cosmo)

        # It is possible that some of these radius entries are going to be NaN - the result of NaN temperature values
        #  passed through the 'predict' method of the scaling relation. As such we identify any NaN results and
        #  remove the radii from the pr_rs array as we're going to do the same for the clusters in the sample
        bad_pr_rs = np.where(np.isnan(pr_rs))[0]
        # pr_rs[bad_pr_rs] = samp.r500[bad_pr_rs]
        pr_rs = np.delete(pr_rs, bad_pr_rs)
        acc_rad = np.delete(acc_rad, bad_pr_rs)
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
        rad_rat = pr_rs / samp.get_radius(o_dens)

        # Make a copy of the currently set radius values from the sample - these will then be modified with the
        #  new predicted values if the particular cluster's radius isn't already considered 'accepted' - i.e. it
        #  reached the required convergence in a previous iteration
        new_rads = samp.get_radius(o_dens).copy()
        # The clusters which DON'T have previously accepted radii have their radii updated from those predicted from
        #  temperature
        new_rads[~acc_rad] = pr_rs[~acc_rad]
        # Use the new radius value inferred from the temperature + scaling relation and add it to the ClusterSample (or
        #  just re-adding the same value as is already here if that radius has converged and been accepted).
        if o_dens == 'r500':
            samp.r500 = new_rads
        elif o_dens == 'r2500':
            samp.r2500 = new_rads
        elif o_dens == 'r200':
            samp.r200 = new_rads

        # If there have been enough iterations, then we need to start checking whether any of the radii have
        #  converged to within the user-specified fraction. If they have then we accept them and those radii won't
        #  be changed the next time around.
        if iter_num >= min_iter:
            acc_rad = ((rad_rat > (1 - convergence_frac)) & (rad_rat < (1 + convergence_frac))) | acc_rad

        rad_hist = {n: vals + [samp[n].get_radius(o_dens, 'kpc').value] if n in samp.names else vals
                    for n, vals in rad_hist.items()}
        # Got to increment the counter otherwise the while loop may go on and on forever :O
        iter_num += 1

    # At this point we've exited the loop - the final radii have been decided on. However, we cannot guarantee that
    #  the final radii have had spectra generated/fit for them, so we run single_temp_apec again one last time
    single_temp_apec(samp, samp.get_radius(o_dens), one_rmf=False, lum_en=lum_en)

    # We also check to see whether the user requested core-excised measurements also be performed. If so then we'll
    #  just multiply the current radius by 0.15 and use that for the inner radius.
    if core_excised:
        single_temp_apec(samp, samp.get_radius(o_dens), samp.get_radius(o_dens)*0.15, one_rmf=False, lum_en=lum_en)

    # Now to assemble the final sample information dataframe - note that the sample does have methods for the bulk
    #  retrieval of temperature and luminosity values, but they aren't so useful here because I know that some of the
    #  original entries in sample_data might have been deleted from the sample object itself
    for row_ind, row in sample_data.iterrows():
        # We're iterating through the rows of the sample information passed in, because we want there to be an
        #  entry even if the LT pipeline didn't succeed. As such we have to check if the current row's cluster
        #  is actually still a part of the sample
        if row['name'] in samp.names:
            # Grab the relevant source out of the ClusterSample object
            rel_src = samp[row['name']]
            rel_rad = rel_src.get_radius(o_dens, 'kpc')
            # These will be to store the read-out temperature and luminosity values, and their corresponding
            #  column names for the dataframe
            vals = [rel_src.get_radius(o_dens, 'kpc').value]
            cols = [o_dens.upper()]

            # We have to use try-excepts here, because even at this stage it is possible that we have a failed
            #  spectral fit to contend with - if there are no successful fits then the entry for the current
            #  cluster will be NaN
            try:
                # The temperature measured within the overdensity radius, with its - and + uncertainties are read out
                vals += list(rel_src.get_temperature(rel_rad).value)
                # We add columns with informative names
                cols += ['Tx' + o_dens[1:] + p_fix for p_fix in ['', '-', '+']]

                # Cycle through every available luminosity, this will return all luminosities in all energy bands
                #  requested by the user with lum_en
                for lum_name, lum in rel_src.get_luminosities(rel_rad).items():
                    # The luminosity and its uncertainties gets added to the values list
                    vals += list(lum.value)
                    # Then the column names get added
                    cols += ['Lx' + o_dens[1:] + lum_name.split('bound')[-1] + p_fix for p_fix in ['', '-', '+']]

            except ModelNotAssociatedError:
                # For the temperature that apparently failed
                # vals += [np.NaN, np.NaN, np.NaN]
                pass

            # Now we repeat the above process, but only if we know the user requested core-excised values as well
            if core_excised:
                try:
                    # Adding temperature value and uncertainties
                    vals += list(rel_src.get_temperature(rel_rad, inner_radius=0.15*rel_rad).value)
                    # Corresponding column names (with ce now included to indicate core-excised).
                    cols += ['Tx' + o_dens[1:] + 'ce' + p_fix for p_fix in ['', '-', '+']]

                    # The same process again for core-excised luminosities
                    lce_res = rel_src.get_luminosities(rel_rad, inner_radius=0.15*rel_rad)
                    for lum_name, lum in lce_res.items():
                        vals += list(lum.value)
                        cols += ['Lx' + o_dens[1:] + 'ce' + lum_name.split('bound')[-1] + p_fix
                                 for p_fix in ['', '-', '+']]

                except ModelNotAssociatedError:
                    # For the core-excised temperature that apparently failed
                    # vals += [np.NaN, np.NaN, np.NaN]
                    pass

            # We know that at least the radius will always be there to be added to the dataframe, so we add the
            #  information in vals and cols
            sample_data.loc[row_ind, cols] = vals

    # If the user wants to save the resulting dataframe to disk then we do so
    if save_samp_results_path is not None:
        sample_data.to_csv(save_samp_results_path, index=False)

    # Finally, we put together the radius history throughout the iteration-convergence process
    radius_hist_df = pd.DataFrame.from_dict(rad_hist, orient='index')
    # And if the user wants this saved as well they can
    if save_rad_history_path is not None:
        radius_hist_df.to_csv(save_rad_history_path, index=False)

    return samp, sample_data, radius_hist_df







