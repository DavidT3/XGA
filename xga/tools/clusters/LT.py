#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 22/08/2023, 18:29. Copyright (c) The Contributors
from typing import Tuple
from warnings import warn

import numpy as np
import pandas as pd
from astropy.cosmology import Cosmology
from astropy.units import Quantity, Unit, UnitConversionError

from xga import DEFAULT_COSMO, NUM_CORES
from xga.exceptions import ModelNotAssociatedError, SASGenerationError
from xga.products import ScalingRelation
from xga.relations.clusters.RT import arnaud_r500
from xga.samples import ClusterSample
from xga.sas import evselect_spectrum
from xga.xspec import single_temp_apec

# This just sets the data columns that MUST be present in the sample data passed by the user
LT_REQUIRED_COLS = ['ra', 'dec', 'name', 'redshift']


def luminosity_temperature_pipeline(sample_data: pd.DataFrame, start_aperture: Quantity, use_peak: bool = False,
                                    peak_find_method: str = "hierarchical", convergence_frac: float = 0.1,
                                    min_iter: int = 3, max_iter: int = 10, rad_temp_rel: ScalingRelation = arnaud_r500,
                                    lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"),
                                    core_excised: bool = False, freeze_nh: bool = True, freeze_met: bool = True,
                                    lo_en: Quantity = Quantity(0.3, "keV"), hi_en: Quantity = Quantity(7.9, "keV"),
                                    group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                                    over_sample: float = None, save_samp_results_path: str = None,
                                    save_rad_history_path: str = None, cosmo: Cosmology = DEFAULT_COSMO,
                                    timeout: Quantity = Quantity(1, 'hr'), num_cores: int = NUM_CORES) \
        -> Tuple[ClusterSample, pd.DataFrame, pd.DataFrame]:
    """
     This is the XGA pipeline for measuring overdensity radii, and the temperatures and luminosities within the
     radii, for a sample of clusters. No knowledge of the overdensity radii of the clusters is required
     beforehand, only the position and redshift of the objects. A name is also required for each of them.

     The pipeline works by measuring a temperature from a spectrum generated with radius equal to the
     'start_aperture', and the using the radius temperature relation ('rad_temp_rel') to infer a value for the
     overdensity radius you are targeting. The cluster's overdensity radius is set equal to the new radius estimate
     and we repeat the process.

     A cluster radius measurement is accepted if the 'current' estimate of the radius is considered to be converged
     with the last estimate. For instance if 'convergence_frac' is set to 0.1, convergence occurs when a change of
     less than 10% from the last radius estimate is measured. The radii cannot be assessed for convergence until
     at least 'min_iter' iterations have been passed, and the iterative process will end if the number of iterations
     reaches 'max_iter'.

     This pipeline will only work for clusters that we can successfully measure temperatures for, which requires a
     minimum data quality - as such you may find that some do not achieve successful radius measurements with this
     pipeline. In these cases the pipeline should not error, but the failure will be recorded in the results and
     radius history dataframes returned from the function (and optionally written to CSV files). The pipeline will
     also gracefully handle SAS spectrum generation failures, removing the offending clusters from the sample being
     analysed and warning the user of the failure.

     As with all XGA sources and samples, the XGA luminosity-temperature pipeline DOES NOT require all objects
     passed in the sample_data to have X-ray observations. Those that do not will simply be filtered out.

     This pipeline will not read in previous XSPEC fits in its current form, though previously generated spectra
     will be read in.

    :param pd.DataFrame sample_data: A dataframe of information on the galaxy clusters. The columns 'ra', 'dec',
        'name', and 'redshift' are required for this pipeline to work.
    :param Quantity start_aperture: This is the radius used to generate the first set of spectra for each
        cluster, which in turn are fit to produce the first temperature estimate.
    :param bool use_peak: If True then XGA will measure an X-ray peak coordinate and use that as the centre for
        spectrum generation and fitting, and the peak coordinate will be included in the results dataframe/csv.
        If False then the coordinate in sample_data will be used. Default is False.
    :param str peak_find_method: Which peak finding method should be used (if use_peak is True). Default is
        'hierarchical' (uses XGA's hierarchical clustering peak finder), 'simple' may also be passed in which
        case the brightest unmasked pixel within the source region will be selected.
    :param float convergence_frac: This defines how close a current radii estimate must be to the last
        radii measurement for it to count as converged. The default value is 0.1, which means the current-to-last
        estimate ratio must be between 0.9 and 1.1.
    :param int min_iter: The minimum number of iterations before a radius can converge and be accepted. The
        default is 3.
    :param int max_iter: The maximum number of iterations before the loop exits and the pipeline moves on. This
        makes sure that the loop has an exit condition and won't continue on forever. The default is 10.
    :param ScalingRelation rad_temp_rel: The scaling relation used to convert a cluster temperature measurement
        for into an estimate of an overdensity radius. The y-axis must be radii, and the x-axis must be temperature.
        The pipeline will attempt to determine the overdensity radius you are attempting to measure for by checking
        the name of the y-axis; it must contain 2500, 500, or 200 to indicate the overdensity. The default is the
        R500-Tx Arnaud et al. 2005 relation.
    :param Quantity lum_en: The energy bands in which to measure luminosity. The default is
        Quantity([[0.5, 2.0], [0.01, 100.0]], 'keV'), corresponding to the 0.5-2.0keV and bolometric bands.
    :param bool core_excised: Should final measurements of temperature and luminosity be made with core-excision in
        addition to measurements within the overdensity radius specified by the scaling relation. This will involve
        multiplying the radii by 0.15 to determine the inner radius. Default is False.
    :param bool freeze_nh: Controls whether the hydrogen column density (nH) should be frozen during XSPEC fits to
        spectra, the default is True.
    :param bool freeze_met: Controls whether metallicity should be frozen during XSPEC fits to spectra, the default
        is False. Leaving metallicity free to vary tends to require more photons to achieve a good fit.
    :param Quantity lo_en: The lower energy limit for the data to be fitted by XSPEC. The default is 0.3 keV.
    :param Quantity hi_en: The upper energy limit for the data to be fitted by XSPEC. The default is 7.9 keV, but
        reducing this value may help achieve a good fit for suspected lower temperature systems.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param str save_samp_results_path: The path to save the final results (temperatures, luminosities, radii) to.
        The default is None, in which case no file will be created. This information is also returned from this
        function.
    :param str save_rad_history_path: The path to save the radii history for all clusters. This specifies what the
        estimated radius was for each cluster at each iteration step, in kpc. The default is None, in which case no
        file will be created. This information is also returned from this function.
    :param Cosmology cosmo: The cosmology to use for sample declaration, and thus for all analysis. The default
        cosmology is a flat LambdaCDM concordance model.
    :param Quantity timeout: This sets the amount of time an XSPEC fit can run before it is timed out, the default
        is 1 hour.
    :param int num_cores: The number of cores that can be used for spectrum generation and fitting. The default is
        90% of the cores detected on the system.
    :return: The GalaxyCluster sample object used for this analysis, the dataframe of results for all input
        objects (even those for which the pipeline was unsuccessful), and the radius history dataframe for the
        clusters.
    :rtype: Tuple[ClusterSample, pd.DataFrame, pd.DataFrame]
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

    # Keeps track of the current iteration number
    iter_num = 0

    # Set up the ClusterSample to be used for this process (I did consider setting up a new one each time but that
    #  adds overhead, and I think that this way should work fine).
    samp = ClusterSample(sample_data['ra'].values, sample_data['dec'].values, sample_data['redshift'].values,
                         sample_data['name'].values, use_peak=use_peak, peak_find_method=peak_find_method,
                         clean_obs_threshold=0.7, clean_obs_reg=o_dens, load_fits=False, cosmology=cosmo, **o_dens_arg)

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

    # This while loop (never thought I'd be using one of them in XGA!) will keep going either until all radii have been
    #  accepted OR until we reach the maximum number  of iterations
    while acc_rad.sum() != len(samp) and iter_num < max_iter:
        # We have a try-except looking for SAS generation errors - they will only be thrown once all the spectrum
        #  generation processes have finished, so we know that the spectra that didn't throw an error exist and
        #  are fine
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
                #  with the error parsing and we need to warn the user.
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
        single_temp_apec(samp, samp.get_radius(o_dens), one_rmf=False, num_cores=num_cores, timeout=timeout,
                         lum_en=lum_en, freeze_met=freeze_met, freeze_nh=freeze_nh, lo_en=lo_en, hi_en=hi_en,
                         group_spec=group_spec, min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)

        # Just reading out the temperatures, not the uncertainties at the moment
        txs = samp.Tx(samp.get_radius(o_dens), quality_checks=False, group_spec=group_spec, min_counts=min_counts,
                      min_sn=min_sn, over_sample=over_sample)[:, 0]

        # This uses the scaling relation to predict the overdensity radius from the measured temperatures
        pr_rs = rad_temp_rel.predict(txs, samp.redshifts, samp.cosmo)

        # It is possible that some of these radius entries are going to be NaN - the result of NaN temperature values
        #  passed through the 'predict' method of the scaling relation. As such we identify any NaN results and
        #  remove the radii from the pr_rs array as we're going to do the same for the clusters in the sample
        bad_pr_rs = np.where(np.isnan(pr_rs))[0]
        pr_rs = np.delete(pr_rs, bad_pr_rs)
        acc_rad = np.delete(acc_rad, bad_pr_rs)

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

        rad_hist = {n: vals + [samp[n].get_radius(o_dens, 'kpc').value] if n in samp.names else vals
                    for n, vals in rad_hist.items()}
        # Got to increment the counter otherwise the while loop may go on and on forever :O
        iter_num += 1

    # Throw a warning if the maximum number of iterations being reached was the reason the loop exited
    if iter_num == max_iter:
        warn("The radius measurement process reached the maximum number of iterations; as such one or more clusters "
             "may have unconverged radii.")

    # At this point we've exited the loop - the final radii have been decided on. However, we cannot guarantee that
    #  the final radii have had spectra generated/fit for them, so we run single_temp_apec again one last time
    single_temp_apec(samp, samp.get_radius(o_dens), one_rmf=False, lum_en=lum_en, num_cores=num_cores,
                     freeze_met=freeze_met, freeze_nh=freeze_nh, lo_en=lo_en, hi_en=hi_en, group_spec=group_spec,
                     min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)

    # We also check to see whether the user requested core-excised measurements also be performed. If so then we'll
    #  just multiply the current radius by 0.15 and use that for the inner radius.
    if core_excised:
        single_temp_apec(samp, samp.get_radius(o_dens), samp.get_radius(o_dens)*0.15, one_rmf=False, lum_en=lum_en,
                         num_cores=num_cores, freeze_met=freeze_met, freeze_nh=freeze_nh, lo_en=lo_en, hi_en=hi_en,
                         group_spec=group_spec, min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)

    # Now to assemble the final sample information dataframe - note that the sample does have methods for the bulk
    #  retrieval of temperature and luminosity values, but they aren't so useful here because I know that some of the
    #  original entries in sample_data might have been deleted from the sample object itself
    for row_ind, row in loaded_samp_data.iterrows():
        # We're iterating through the rows of the sample information passed in, because we want there to be an
        #  entry even if the LT pipeline didn't succeed. As such we have to check if the current row's cluster
        #  is actually still a part of the sample
        if row['name'] in samp.names:
            # Grab the relevant source out of the ClusterSample object
            rel_src = samp[row['name']]
            rel_rad = rel_src.get_radius(o_dens, 'kpc')

            # These will be to store the read-out temperature and luminosity values, and their corresponding
            #  column names for the dataframe - we make sure that the measure radius is present in the data
            vals = [rel_rad.value]
            cols = [o_dens]

            # If the user let XGA determine a peak coordinate for the cluster, we will need to add it to the results
            #  as all the spectra for the cluster were generated with that as the central point
            if use_peak:
                vals += [*rel_src.peak.value]
                cols += ['peak_ra', 'peak_dec']

            # We have to use try-excepts here, because even at this stage it is possible that we have a failed
            #  spectral fit to contend with - if there are no successful fits then the entry for the current
            #  cluster will be NaN
            try:
                # The temperature measured within the overdensity radius, with its - and + uncertainties are read out
                vals += list(rel_src.get_temperature(rel_rad, group_spec=group_spec, min_counts=min_counts,
                                                     min_sn=min_sn, over_sample=over_sample).value)
                # We add columns with informative names
                cols += ['Tx' + o_dens[1:] + p_fix for p_fix in ['', '-', '+']]

                # Cycle through every available luminosity, this will return all luminosities in all energy bands
                #  requested by the user with lum_en
                for lum_name, lum in rel_src.get_luminosities(rel_rad, group_spec=group_spec,
                                                              min_counts=min_counts, min_sn=min_sn,
                                                              over_sample=over_sample).items():
                    # The luminosity and its uncertainties gets added to the values list
                    vals += list(lum.value)
                    # Then the column names get added
                    cols += ['Lx' + o_dens[1:] + lum_name.split('bound')[-1] + p_fix for p_fix in ['', '-', '+']]

                # If we note that the metallicity and/or nH were left free to vary, we had better save those values
                #  as well!
                if not freeze_met:
                    met = rel_src.get_results(rel_rad, par='Abundanc', group_spec=group_spec, min_counts=min_counts,
                                              min_sn=min_sn, over_sample=over_sample)
                    vals += list(met)
                    cols += ['Zmet' + o_dens[1:] + p_fix for p_fix in ['', '-', '+']]

                if not freeze_nh:
                    nh = rel_src.get_results(rel_rad, par='nH', group_spec=group_spec, min_counts=min_counts,
                                             min_sn=min_sn, over_sample=over_sample)
                    vals += list(nh)
                    cols += ['nH' + o_dens[1:] + p_fix for p_fix in ['', '-', '+']]

            except ModelNotAssociatedError:
                pass

            # Now we repeat the above process, but only if we know the user requested core-excised values as well
            if core_excised:
                try:
                    # Adding temperature value and uncertainties
                    vals += list(rel_src.get_temperature(rel_rad, inner_radius=0.15*rel_rad, group_spec=group_spec,
                                                         min_counts=min_counts, min_sn=min_sn,
                                                         over_sample=over_sample).value)
                    # Corresponding column names (with ce now included to indicate core-excised).
                    cols += ['Tx' + o_dens[1:] + 'ce' + p_fix for p_fix in ['', '-', '+']]

                    # The same process again for core-excised luminosities
                    lce_res = rel_src.get_luminosities(rel_rad, inner_radius=0.15*rel_rad, group_spec=group_spec,
                                                       min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)
                    for lum_name, lum in lce_res.items():
                        vals += list(lum.value)
                        cols += ['Lx' + o_dens[1:] + 'ce' + lum_name.split('bound')[-1] + p_fix
                                 for p_fix in ['', '-', '+']]

                    # If we note that the metallicity and/or nH were left free to vary, we had better save those values
                    #  as well!
                    if not freeze_met:
                        metce = rel_src.get_results(rel_rad, par='Abundanc', inner_radius=0.15*rel_rad,
                                                    group_spec=group_spec, min_counts=min_counts, min_sn=min_sn,
                                                    over_sample=over_sample)
                        vals += list(metce)
                        cols += ['Zmet' + o_dens[1:] + 'ce' + p_fix for p_fix in ['', '-', '+']]

                    if not freeze_nh:
                        nhce = rel_src.get_results(rel_rad, par='nH', inner_radius=0.15*rel_rad, group_spec=group_spec,
                                                   min_counts=min_counts, min_sn=min_sn, over_sample=over_sample)
                        vals += list(nhce)
                        cols += ['nH' + o_dens[1:] + 'ce' + p_fix for p_fix in ['', '-', '+']]

                except ModelNotAssociatedError:
                    pass

            # We know that at least the radius will always be there to be added to the dataframe, so we add the
            #  information in vals and cols
            loaded_samp_data.loc[row_ind, cols] = vals

    # If the user wants to save the resulting dataframe to disk then we do so
    if save_samp_results_path is not None:
        loaded_samp_data.to_csv(save_samp_results_path, index=False)

    # Finally, we put together the radius history throughout the iteration-convergence process
    radius_hist_df = pd.DataFrame.from_dict(rad_hist, orient='index')

    # There is already an array detailing whether particular radii have been 'accepted' (i.e. converged) or not, but
    #  it only contains entries for those clusters which are still loaded in the ClusterSample - in the next part
    #  of this pipeline I assemble a radius history dataframe (for all clusters that were initially in the
    #  ClusterSample), and want the final column to declare if they converged or not.
    rad_hist_acc_rad = []
    for row_ind, row in loaded_samp_data.iterrows():
        if row['name'] in samp.names:
            # Did this radius converge?
            converged = acc_rad[np.argwhere(samp.names == row['name'])[0][0]]
        else:
            converged = False
        rad_hist_acc_rad.append(converged)

    # We add the final column which just tells the user whether the radius was converged or not
    radius_hist_df['converged'] = rad_hist_acc_rad

    # And if the user wants this saved as well they can
    if save_rad_history_path is not None:
        # This one I keep indexing set to True, because the names of the clusters are acting as the index for
        #  this dataframe
        radius_hist_df.to_csv(save_rad_history_path, index=True, index_label='name')

    return samp, loaded_samp_data, radius_hist_df







