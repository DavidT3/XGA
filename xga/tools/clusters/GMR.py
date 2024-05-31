#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 31/05/2024, 12:38. Copyright (c) The Contributors

from typing import Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from astropy.cosmology import Cosmology
from astropy.units import Quantity
from xga import DEFAULT_COSMO, NUM_CORES
from xga.exceptions import SASGenerationError
from xga.imagetools.psf import rl_psf
from xga.samples import ClusterSample
from xga.sas import evselect_spectrum
from xga.sourcetools.density import inv_abel_fitted_model
from xga.xspec import single_temp_apec

# This just sets the data columns that MUST be present in the sample data passed by the user
LT_REQUIRED_COLS = ['ra', 'dec', 'name', 'redshift']


# TODO DECIDE - freeze_temp: bool = False,
def gas_mass_radius_pipeline(sample_data: pd.DataFrame, delta: int, baryon_frac: Union[float, np.ndarray],
                             sb_model: str, dens_model: str,
                             start_aperture: Quantity, use_peak: bool = False,
                             peak_find_method: str = "hierarchical", convergence_frac: float = 0.1, min_iter: int = 3,
                             max_iter: int = 10, psf_corr: bool = True, psf_model: str = "ELLBETA", psf_bins: int = 4,
                             psf_algo: str = "rl", psf_iter: int = 15, freeze_nh: bool = True, freeze_met: bool = True,
                             failover_temp: Quantity = Quantity(3.0, 'keV'),
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

    # TODO Not sure whether this will stay the way it is, as I have not figured out whether I should let it re-measure
    #  a temperature each iteration (will make it take much longer).
    try:
        # Run the spectrum generation for the current values of the over density radius
        evselect_spectrum(samp, samp.get_radius(o_dens), num_cores=num_cores, one_rmf=False, group_spec=group_spec,
                          min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)
        # If the end of evselect_spectrum doesn't throw a SASGenerationError then we know we're all good, so we
        #  define the not_bad_gen_ind to just contain an index for all the clusters
        not_bad_gen_ind = np.nonzero(samp.names)
    except SASGenerationError as err:
        # Otherwise if something went wrong we can parse the error messages and extract the names of the sources
        #  for which the error occurred
        poss_bad_gen = list(set([me.message.split(' is the associated source')[0].split('- ')[-1]
                                 for i_err in err.message for me in i_err]))
        # Do also need to check that the entries in poss_bad_gen are actually source names - as XGA is raising
        #  the errors we're parsing, we SHOULD be able to rely on them being a certain format, but we had better
        #  be safe
        bad_gen = [en for en in poss_bad_gen if en in samp.names]
        if len(bad_gen) != len(poss_bad_gen):
            # If there are entries in poss_bad_gen that ARE NOT names in the sample, then something has gone wrong
            #  with the error parsing, and we need to warn the user.
            problem = [en for en in poss_bad_gen if en not in samp.names]
            warn("SASGenerationError parsing has recovered a string that is not a source name, a "
                 "problem source may not have been removed from the sample (contact the development team). The "
                 "offending strings are, {}".format(', '.join(problem)), stacklevel=2)

        # Just to be safe I'm adding a check to make sure bad_gen has entries
        if len(bad_gen) == 0:
            raise SASGenerationError("Failed to identify sources for which SAS spectrum generation failed.")

        # We define the indices that WON'T have been removed from the sample (so these can be used to address
        #  things like the pr_rs quantity we defined up top
        not_bad_gen_ind = np.nonzero(~np.isin(samp.names, bad_gen))
        acc_rad = acc_rad[not_bad_gen_ind]

        # Then we can cycle through those names and delete the sources from the sample (throwing a hopefully
        #  useful warning as well).
        for bad_name in bad_gen:
            if bad_name in samp.names:
                del samp[bad_name]
        warn("Some sources ({}) have been removed because of spectrum generation "
             "failures.".format(', '.join(bad_gen)), stacklevel=2)

    # We generate and fit spectra for the current value of the overdensity radius
    # TODO Decide -  lum_en=lum_en,
    single_temp_apec(samp, samp.get_radius(o_dens), freeze_nh=freeze_nh, freeze_met=freeze_met,
                     lo_en=lo_en, hi_en=hi_en, group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                     over_sample=over_sample, one_rmf=False, num_cores=num_cores, timeout=timeout,
                     start_temp=failover_temp)

    # Just reading out the temperatures, not the uncertainties at the moment
    tx_all = samp.Tx(samp.get_radius(o_dens), quality_checks=False, group_spec=group_spec,
                     min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)
    txs = tx_all[:, 0]
    tx_errs = tx_all[:, 1]

    nan_tx_ind = np.where(np.isnan(txs))[0]
    txs[nan_tx_ind] = failover_temp

    if psf_corr:
        rl_psf(samp, bins=10, iterations=psf_iter, psf_model=psf_model)

    # This while loop (never thought I'd be using one of them in XGA!) will keep going either until all radii have been
    #  accepted OR until we reach the maximum number  of iterations
    while acc_rad.sum() != len(samp) and iter_num < max_iter:
        print(txs)
        # TODO Decide whether conv_outer_radius will actually be the start aperture all the time
        dps = inv_abel_fitted_model(samp, sb_model, psf_corr=psf_corr, psf_bins=10, psf_iter=psf_iter,
                                    psf_model=psf_model, outer_radius=o_dens, use_peak=use_peak,
                                    conv_outer_radius=start_aperture, conv_temp=txs, show_warn=False)
        # TODO test out analytical king vs direct numerical inv. abel
        temp_new_rads = []
        for dp in dps:
            if dp is not None:
                rel_src = samp[dp.src_name]
                dp.fit(dens_model, show_warn=False)
                new_rad = dp.overdensity_radius(delta, dens_model, rel_src.redshift, rel_src.cosmo, baryon_frac)
                temp_new_rads.append(new_rad)
            else:
                temp_new_rads.append(Quantity(np.NaN, 'kpc'))

        temp_new_rads = Quantity(temp_new_rads)

        bad_pr_rs = np.where(np.isnan(temp_new_rads))[0]
        pr_rs = np.delete(temp_new_rads, bad_pr_rs)
        # pr_r_errs = np.delete(pr_r_errs, bad_pr_rs)
        acc_rad = np.delete(acc_rad, bad_pr_rs)

        # Have to remove the failures from this because it is used for the density profile measurements
        txs = np.delete(txs, bad_pr_rs)

        # I am also actually going to remove the clusters with NaN results from the sample - if the NaN was caused
        #  by something like a fit not converging then it's going to keep trying over and over again and that could
        #  slow everything down.
        # I make sure not to try to remove clusters which I've ALREADY removed further up because their spectral
        #  generation failed.
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
        # This dictionary is used to store the various radius steps that are made for each source
        rad_hist = {n: vals + [samp[n].get_radius(o_dens, 'kpc').value] if n in samp.names else vals
                    for n, vals in rad_hist.items()}

        # Got to increment the counter otherwise the while loop may go on and on forever :O
        iter_num += 1

    # TODO This is temporary because I want to set this running and go home
    return samp, rad_hist