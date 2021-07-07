#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 07/07/2021, 17:50. Copyright (c) David J Turner

from typing import Union, List

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity
from tqdm import tqdm

from .base import BaseSample
from ..exceptions import PeakConvergenceFailedError, ModelNotAssociatedError, ParameterNotAssociatedError, \
    NoProductAvailableError, NoValidObservationsError
from ..products.profile import GasDensity3D
from ..relations.fit import *
from ..sources.extended import GalaxyCluster


# Names are required for the ClusterSample because they'll be used to access specific cluster objects
class ClusterSample(BaseSample):
    """
    A sample class to be used for declaring and analysing populations of galaxy clusters, with many cluster-science
    specific functions, such as the ability to create common scaling relations.
    """
    def __init__(self, ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray, name: np.ndarray, r200: Quantity = None,
                 r500: Quantity = None, r2500: Quantity = None, richness: np.ndarray = None,
                 richness_err: np.ndarray = None, wl_mass: Quantity = None, wl_mass_err: Quantity = None,
                 custom_region_radius: Quantity = None, use_peak: bool = True,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5, cosmology=Planck15,
                 load_fits: bool = False, clean_obs: bool = True, clean_obs_reg: str = "r200",
                 clean_obs_threshold: float = 0.3, no_prog_bar: bool = False, psf_corr: bool = False):

        # I don't like having this here, but it does avoid a circular import problem
        from xga.sas import evselect_image, eexpmap, emosaic

        # Using the super defines BaseSources and stores them in the self._sources dictionary
        super().__init__(ra, dec, redshift, name, cosmology, load_products=True, load_fits=False,
                         no_prog_bar=no_prog_bar)

        # This part is super useful - it is much quicker to use the base sources to generate all
        #  necessary ratemaps, as we can do it in parallel for the entire sample, rather than one at a time as
        #  might be necessary for peak finding in the cluster init.
        evselect_image(self, peak_lo_en, peak_hi_en)
        eexpmap(self, peak_lo_en, peak_hi_en)
        emosaic(self, "image", peak_lo_en, peak_hi_en)
        emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        # Now that we've made those images the BaseSource objects aren't required anymore, we're about
        #  to define GalaxyClusters
        del self._sources
        self._sources = {}

        # We have this final names list in case so that we don't need to remove elements of self.names if one of the
        #  clusters doesn't pass the observation cleaning stage.
        final_names = []
        with tqdm(desc="Setting up Galaxy Clusters", total=len(self.names), disable=no_prog_bar) as dec_lb:
            for ind, r in enumerate(ra):
                # Just splitting out relevant values for this particular cluster so the object declaration isn't
                #  super ugly.
                d = dec[ind]
                z = redshift[ind]
                # The replace is there because source declaration removes spaces from any passed names,
                n = name[ind].replace(' ', '')
                # Declaring the BaseSample higher up weeds out those objects that aren't in any XMM observations
                #  So we want to check that the current object name is in the list of objects that have data
                if n in self.names:
                    # I know this code is a bit ugly, but oh well
                    if r200 is not None:
                        r2 = r200[ind]
                    else:
                        r2 = None
                    if r500 is not None:
                        r5 = r500[ind]
                    else:
                        r5 = None
                    if r2500 is not None:
                        r25 = r2500[ind]
                    else:
                        r25 = None
                    if custom_region_radius is not None:
                        cr = custom_region_radius[ind]
                    else:
                        cr = None

                    # Here we check the options that are allowed to be None
                    if richness is not None:
                        lam = richness[ind]
                        lam_err = richness_err[ind]
                    else:
                        lam = None
                        lam_err = None

                    if wl_mass is not None:
                        wlm = wl_mass[ind]
                        wlm_err = wl_mass_err[ind]
                    else:
                        wlm = None
                        wlm_err = None

                    # Will definitely load products (the True in this call), because I just made sure I generated a
                    #  bunch to make GalaxyCluster declaration quicker
                    try:
                        self._sources[n] = GalaxyCluster(r, d, z, n, r2, r5, r25, lam, lam_err, wlm, wlm_err, cr,
                                                         use_peak, peak_lo_en, peak_hi_en, back_inn_rad_factor,
                                                         back_out_rad_factor, cosmology, True, load_fits, clean_obs,
                                                         clean_obs_reg, clean_obs_threshold, False)
                        final_names.append(n)
                    except PeakConvergenceFailedError:
                        warn("The peak finding algorithm has not converged for {}, using user "
                             "supplied coordinates".format(n))
                        self._sources[n] = GalaxyCluster(r, d, z, n, r2, r5, r25, lam, lam_err, wlm, wlm_err, cr, False,
                                                         peak_lo_en, peak_hi_en, back_inn_rad_factor, back_out_rad_factor,
                                                         cosmology, True, load_fits, clean_obs, clean_obs_reg,
                                                         clean_obs_threshold, False)
                        final_names.append(n)

                    except NoValidObservationsError:
                        warn("After applying the criteria for the minimum amount of cluster required on an "
                             "observation, {} cannot be declared as all potential observations were removed".format(n))
                        # Note we don't append n to the final_names list here, as it is effectively being
                        #  removed from the sample
                        self._failed_sources[n] = "Failed ObsClean"
                dec_lb.update(1)

        self._names = final_names

        # And again I ask XGA to generate the merged images and exposure maps, in case any sources have been
        #  cleaned and had data removed
        if clean_obs:
            emosaic(self, "image", peak_lo_en, peak_hi_en)
            emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        # Updates with new peaks
        if clean_obs and use_peak:
            for n in self.names:
                # If the source in question has had data removed
                if self._sources[n].disassociated:
                    try:
                        en_key = "bound_{0}-{1}".format(peak_lo_en.to("keV").value,
                                                        peak_hi_en.to("keV").value)
                        rt = self._sources[n].get_products("combined_ratemap", extra_key=en_key)[0]
                        peak = self._sources[n].find_peak(rt)
                        self._sources[n].peak = peak[0]
                        self._sources[n].default_coord = peak[0]
                    except PeakConvergenceFailedError:
                        pass

        # I don't offer the user choices as to the configuration for PSF correction at the moment
        if psf_corr:
            from ..imagetools.psf import rl_psf
            rl_psf(self, lo_en=peak_lo_en, hi_en=peak_hi_en)

    @property
    def r200_snr(self) -> np.ndarray:
        """
        Fetches and returns the R200 signal to noises from the constituent sources.

        :return: The signal to noise ration calculated at the R200.
        :rtype: np.ndarray
        """
        snrs = []
        for s in self:
            try:
                snrs.append(s.get_snr("r200"))
            except ValueError:
                snrs.append(None)
        return np.array(snrs)

    @property
    def r500_snr(self) -> np.ndarray:
        """
        Fetches and returns the R500 signal to noises from the constituent sources.

        :return: The signal to noise ration calculated at the R500.
        :rtype: np.ndarray
        """
        snrs = []
        for s in self:
            try:
                snrs.append(s.get_snr("r500"))
            except ValueError:
                snrs.append(None)
        return np.array(snrs)

    @property
    def r2500_snr(self) -> np.ndarray:
        """
        Fetches and returns the R2500 signal to noises from the constituent sources.

        :return: The signal to noise ration calculated at the R2500.
        :rtype: np.ndarray
        """
        snrs = []
        for s in self:
            try:
                snrs.append(s.get_snr("r2500"))
            except ValueError:
                snrs.append(None)
        return np.array(snrs)

    @property
    def richness(self) -> Quantity:
        """
        Provides the richnesses of the clusters in this sample, if they were passed in on definition.

        :return: A unitless Quantity object of the richnesses and their error(s).
        :rtype: Quantity
        """
        rs = []
        for gcs in self._sources.values():
            rs.append(gcs.richness.value)

        rs = np.array(rs)

        # We're going to throw an error if all the richnesses are NaN, because obviously something is wrong
        check_rs = rs[~np.isnan(rs)]
        if len(check_rs) == 0:
            raise ValueError("All richnesses appear to be NaN.")

        return Quantity(rs)

    @property
    def wl_mass(self) -> Quantity:
        """
        Provides the weak lensing masses of the clusters in this sample, if they were passed in on definition.

        :return: A Quantity object of the WL masses and their error(s), in whatever units they were when
        they were passed in originally.
        :rtype: Quantity
        """
        wlm = []
        for gcs in self._sources.values():
            wlm.append(gcs.weak_lensing_mass.value)
            wlm_unit = gcs.weak_lensing_mass.unit

        wlm = np.array(wlm)

        # We're going to throw an error if all the weak lensing masses are NaN, because obviously something is wrong
        check_wlm = wlm[~np.isnan(wlm)]
        if len(check_wlm) == 0:
            raise ValueError("All weak lensing masses appear to be NaN.")

        return Quantity(wlm, wlm_unit)

    @property
    def r200(self) -> Quantity:
        """
        Returns all the R200 values passed in on declaration, but in units of kpc.

        :return: A quantity of R200 values.
        :rtype: Quantity
        """
        rads = []
        for gcs in self._sources.values():
            rad = gcs.get_radius('r200', 'kpc')
            if rad is None:
                rads.append(np.NaN)
            else:
                rads.append(rad.value)

        rads = np.array(rads)
        check_rads = rads[~np.isnan(rads)]
        if len(check_rads) == 0:
            raise ValueError("All R200 values appear to be NaN.")

        return Quantity(rads, 'kpc')

    @property
    def r500(self) -> Quantity:
        """
        Returns all the R500 values passed in on declaration, but in units of kpc.

        :return: A quantity of R500 values.
        :rtype: Quantity
        """
        rads = []
        for gcs in self._sources.values():
            rad = gcs.get_radius('r500', 'kpc')
            if rad is None:
                rads.append(np.NaN)
            else:
                rads.append(rad.value)

        rads = np.array(rads)
        check_rads = rads[~np.isnan(rads)]
        if len(check_rads) == 0:
            raise ValueError("All R500 values appear to be NaN.")

        return Quantity(rads, 'kpc')

    @property
    def r2500(self) -> Quantity:
        """
        Returns all the R2500 values passed in on declaration, but in units of kpc.

        :return: A quantity of R2500 values.
        :rtype: Quantity
        """
        rads = []
        for gcs in self._sources.values():
            rad = gcs.get_radius('r2500', 'kpc')
            if rad is None:
                rads.append(np.NaN)
            else:
                rads.append(rad.value)

        rads = np.array(rads)
        check_rads = rads[~np.isnan(rads)]
        if len(check_rads) == 0:
            raise ValueError("All R2500 values appear to be NaN.")

        return Quantity(rads, 'kpc')

    def Lx(self, outer_radius: Union[str, Quantity], model: str = 'constant*tbabs*apec',
           inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = Quantity(0.5, 'keV'),
           hi_en: Quantity = Quantity(2.0, 'keV'), group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
           over_sample: float = None, quality_checks: bool = True):
        """
        A get method for luminosities measured for the constituent sources of this sample. An error will be
        thrown if luminosities haven't been measured for the given region and model, no default model has been
        set, unlike the Tx method of ClusterSample. An extra condition that aims to only return 'good' data has
        been included, so that any Lx measurement with an uncertainty greater than value will be set to NaN, and
        a warning will be issued.

        This overrides the BaseSample method, but the only difference is that this has a default model, which
        is what single_temp_apec fits (constant*tbabs*apec).

        :param str model: The name of the fitted model that you're requesting the luminosities
            from (e.g. constant*tbabs*apec).
        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). You may also pass a quantity containing radius values,
            with one value for each source in this sample.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum. You may also pass a quantity containing radius values, with one value for each
            source in this sample.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param bool quality_checks: Whether the quality checks to make sure a returned value is good enough
            to use should be performed.
        :return: An Nx3 array Quantity where N is the number of sources. First column is the luminosity, second
            column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN
        :rtype: Quantity
        """
        return super().Lx(outer_radius, model, inner_radius, lo_en, hi_en, group_spec, min_counts, min_sn, over_sample,
                          quality_checks)

    def Tx(self, outer_radius: Union[str, Quantity] = 'r500', model: str = 'constant*tbabs*apec',
           inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True, min_counts: int = 5,
           min_sn: float = None, over_sample: float = None, quality_checks: bool = True):
        """
        A get method for temperatures measured for the constituent clusters of this sample. An error will be
        thrown if temperatures haven't been measured for the given region (the default is R_500) and model (default
        is the constant*tbabs*apec model which single_temp_apec fits to cluster spectra). Any clusters for which
        temperature fits failed will return NaN temperatures, and with temperature greater than 25keV is considered
        failed, any temperature with a negative error value is considered failed, any temperature where the Tx-low
        err is less than zero isn't returned, and any temperature where one of the errors is more than three times
        larger than the other is considered failed (if quality checks are on).

        :param str model: The name of the fitted model that you're requesting the results
            from (e.g. constant*tbabs*apec).
        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in
            region files), then any inner radius will be ignored. You may also pass a quantity containing radius
            values, with one value for each source in this sample. The default for this method is r500.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum. You may also pass a quantity containing radius values, with one value for each
            source in this sample.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param bool quality_checks: Whether the quality checks to make sure a returned value is good enough
            to use should be performed.
        :return: An Nx3 array Quantity where N is the number of clusters. First column is the temperature, second
            column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN.
        :rtype: Quantity
        """
        # Has to be here to prevent circular import unfortunately
        from ..sas.spec import region_setup

        if outer_radius != 'region':
            # This just parses the input inner and outer radii into something predictable
            inn_rads, out_rads = region_setup(self, outer_radius, inner_radius, True, '')[1:]
        else:
            raise NotImplementedError("Sorry region fitting is currently not supported")

        temps = []
        for src_ind, gcs in enumerate(self._sources.values()):
            try:
                # Fetch the temperature from a given cluster using the dedicated method
                gcs_temp = gcs.get_temperature(out_rads[src_ind], model, inn_rads[src_ind], group_spec, min_counts,
                                               min_sn, over_sample).value

                # If the measured temperature is 64keV I know that's a failure condition of the XSPEC fit,
                #  so its set to NaN
                if quality_checks and gcs_temp[0] > 25:
                    gcs_temp = np.array([np.NaN, np.NaN, np.NaN])
                    warn("A temperature of {m}keV was measured for {s}, anything over 30keV considered a failed "
                         "fit by XGA".format(s=gcs.name, m=gcs_temp))
                elif quality_checks and gcs_temp.min() < 0:
                    gcs_temp = np.array([np.NaN, np.NaN, np.NaN])
                    warn("A negative value was detected in the temperature array for {s}, this is considered a failed "
                         "measurement".format(s=gcs.name))
                elif quality_checks and ((gcs_temp[0] - gcs_temp[1]) <= 0):
                    gcs_temp = np.array([np.NaN, np.NaN, np.NaN])
                    warn("The temperature value - the lower error goes below zero for {s}, this makes the temperature"
                         " hard to use for scaling relations as values are often logged.".format(s=gcs.name))
                elif quality_checks and ((gcs_temp[1] / gcs_temp[2]) > 3 or (gcs_temp[1] / gcs_temp[2]) < 0.33):
                    gcs_temp = np.array([np.NaN, np.NaN, np.NaN])
                    warn("One of the temperature uncertainty values for {s} is more than three times larger than "
                         "the other, this means the fit quality is suspect.".format(s=gcs.name))
                elif quality_checks and ((gcs_temp[0] - gcs_temp[1:].mean()) < 0):
                    gcs_temp = np.array([np.NaN, np.NaN, np.NaN])
                    warn("The temperature value - the average error goes below zero for {s}, this makes the "
                         "temperature hard to use for scaling relations as values are often logged".format(s=gcs.name))
                temps.append(gcs_temp)

            except (ValueError, ModelNotAssociatedError, ParameterNotAssociatedError) as err:
                # If any of the possible errors are thrown, we print the error as a warning and replace
                #  that entry with a NaN
                warn(str(err))
                temps.append(np.array([np.NaN, np.NaN, np.NaN]))

        # Turn the list of 3 element arrays into an Nx3 array which is then turned into an astropy Quantity
        temps = Quantity(np.array(temps), 'keV')

        # We're going to throw an error if all the temperatures are NaN, because obviously something is wrong
        check_temps = temps[~np.isnan(temps)]
        if len(check_temps) == 0:
            raise ValueError("All temperatures appear to be NaN.")

        return temps

    def gas_mass(self, rad_name: str, dens_model: str, method: str, prof_outer_rad: Union[Quantity, str] = None,
                 pix_step: int = 1, min_snr: Union[float, int] = 0.0, psf_corr: bool = True,
                 psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15,
                 set_ids: List[int] = None, quality_checks: bool = True) -> Quantity:
        """
        A convenient get method for gas masses measured for the constituent clusters of this sample, though the
        arguments that can  be passed to retrieve gas density profiles are limited to tone-down the complexity.
        Largely this method assumes you have measured density profiles from combined surface brightness
        profiles, though if you have generated density profiles from annular spectra and know their set ids you may
        pass them. Any more nuanced access to density profiles will have to be done yourself.

        If the specified density model hasn't been fitted to the density profile, this method will run a fit using
        the default settings of the fit method of XGA profiles.

        A gas mass will be set to NaN if either of the uncertainties are larger than the gas mass value, if
        the gas mass value is less than 1e+9 solar masses, if the gas mass value is greater than 1e+16 solar
        masses, if quality checks are on.

        :param str rad_name: The name of the radius (e.g. r500) to calculate the gas mass within.
        :param str dens_model: The model fit to the density profiles to be used to calculate gas mass. If a fit
            doesn't already exist then one will be performed with default settings.
        :param str method: The method used to generate the density profile. For a profile created by fitting a model
            to a surface brightness profile this should be the name of the model, for a profile from annular spectra
            this should be 'spec', and for a profile generated directly from the data of a surface brightness profile
            this should be 'onion'.
        :param Quantity/str prof_outer_rad: The outer radii of the density profiles, either a single radius name or a
            Quantity containing an outer radius for each cluster. For instance if you defined a ClusterSample called
            srcs you could pass srcs.r500 here.
        :param int pix_step: The width of each annulus in pixels used to generate the profile, for profiles based on
            surface brightness.
        :param float min_snr: The minimum signal to noise imposed upon the profile, for profiles based on
            surface brightness.
        :param bool psf_corr: Is the brightness profile corrected for PSF effects, for profiles based on
            surface brightness.
        :param str psf_model: If PSF corrected, the PSF model used, for profiles based on surface brightness.
        :param int psf_bins: If PSF corrected, the number of bins per side, for profiles based on surface brightness.
        :param str psf_algo: If PSF corrected, the algorithm used, for profiles based on surface brightness.
        :param int psf_iter: If PSF corrected, the number of algorithm iterations, for profiles based on
            surface brightness.
        :param List[int] set_ids: A list of AnnularSpectra set IDs used to generate the density profiles, if you wish
            to use spectrum based density profiles here.
        :param bool quality_checks: Whether the quality checks to make sure a returned value is good enough
            to use should be performed.
        :return: An Nx3 array Quantity where N is the number of clusters. First column is the gas mass, second
            column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN.
        :rtype: Quantity
        """
        # Has to be here to prevent circular import unfortunately
        from ..sas.spec import region_setup

        gms = []
        if prof_outer_rad is not None:
            out_rad_vals = region_setup(self, prof_outer_rad, Quantity(0, 'deg'), True, '')[2]
        else:
            out_rad_vals = None

        if set_ids is not None and len(set_ids) != len(self):
            raise ValueError("If you wish to use density profiles generated from spectra you must supply a list of "
                             "set IDs with an entry for each cluster in this sample.")
        elif set_ids is None:
            set_ids = [None]*len(self)

        # Iterate through all of our Galaxy Clusters
        for gcs_ind, gcs in enumerate(self._sources.values()):
            gas_mass_rad = gcs.get_radius(rad_name, 'kpc')
            try:
                if prof_outer_rad is not None:
                    dens_profs = gcs.get_density_profiles(out_rad_vals[gcs_ind], method, None, None, radii=None,
                                                          pix_step=pix_step, min_snr=min_snr, psf_corr=psf_corr,
                                                          psf_model=psf_model, psf_bins=psf_bins, psf_algo=psf_algo,
                                                          psf_iter=psf_iter, set_id=set_ids[gcs_ind])
                else:
                    dens_profs = gcs.get_density_profiles(out_rad_vals, method, None, None, radii=None,
                                                          pix_step=pix_step, min_snr=min_snr, psf_corr=psf_corr,
                                                          psf_model=psf_model, psf_bins=psf_bins, psf_algo=psf_algo,
                                                          psf_iter=psf_iter, set_id=set_ids[gcs_ind])

                if isinstance(dens_profs, GasDensity3D):
                    if dens_model not in dens_profs.good_model_fits:
                        dens_profs.fit(dens_model)

                    try:
                        cur_gmass = dens_profs.gas_mass(dens_model, gas_mass_rad)[0]
                        if quality_checks and (cur_gmass[1] > cur_gmass[0] or cur_gmass[2] > cur_gmass[0]):
                            gms.append([np.NaN, np.NaN, np.NaN])
                        elif quality_checks and cur_gmass[0] < Quantity(1e+9, 'Msun'):
                            gms.append([np.NaN, np.NaN, np.NaN])
                            warn("{s}'s gas mass is less than 1e+12 solar masses")
                        elif quality_checks and cur_gmass[0] > Quantity(1e+16, 'Msun'):
                            gms.append([np.NaN, np.NaN, np.NaN])
                            warn("{s}'s gas mass is greater than 1e+16 solar masses")
                        else:
                            gms.append(cur_gmass.value)
                    except ModelNotAssociatedError:
                        gms.append([np.NaN, np.NaN, np.NaN])
                    except ValueError:
                        gms.append([np.NaN, np.NaN, np.NaN])
                        warn("{s}'s gas mass is negative")

                else:
                    warn("Somehow there multiple matches for {s}'s density profile, this is the developer's "
                         "fault.".format(s=gcs.name))
                    gms.append([np.NaN, np.NaN, np.NaN])

            except NoProductAvailableError:
                # If no dens_prof has been run or something goes wrong then NaNs are added
                gms.append([np.NaN, np.NaN, np.NaN])
                warn("{s} doesn't have a density profile associated, please look at "
                     "sourcetools.density.".format(s=gcs.name))

        gms = np.array(gms)

        # We're going to throw an error if all the gas masses are NaN, because obviously something is wrong
        check_gms = gms[~np.isnan(gms)]
        if len(check_gms) == 0:
            raise ValueError("All gas masses appear to be NaN.")

        return Quantity(gms, 'Msun')

    def hydrostatic_mass(self, rad_name: str, temp_model_name: str = None, dens_model_name: str = None,
                         quality_checks: bool = True) -> Quantity:
        """
        A simple method for fetching hydrostatic masses of this sample of clusters. This function is limited, and if
        you have generated multiple hydrostatic mass profiles you may have to use the get_hydrostatic_mass_profiles
        function of each source directly, or use the returned profiles from the function that generated them.

        If only one hydrostatic mass profile has been generated for each source, then you do not need to specify model
        names, but if the same temperature and density profiles have been used to make a hydrostatic mass profile but
        with different models then you may use them.

        A mass will be set to NaN if either of the uncertainties are larger than the mass value, if the mass value
        is less than 1e+12 solar masses, if the mass value is greater than 1e+16 solar masses (if quality checks
        are on), or if no hydrostatic mass profile is available.

        :param str rad_name: The name of the radius (e.g. r500) to calculate the hydrostatic mass within.
        :param str temp_model_name: The name of the model used to fit the temperature profile used to generate the
            required hydrostatic mass profile, default is None.
        :param str dens_model_name: The name of the model used to fit the density profile used to generate the
            required hydrostatic mass profile, default is None.
        :param bool quality_checks: Whether the quality checks to make sure a returned value is good enough
            to use should be performed.
        :return: An Nx3 array Quantity where N is the number of clusters. First column is the hydrostatic mass, second
            column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN.
        :rtype: Quantity
        """
        ms = []

        # Iterate through all of our Galaxy Clusters
        for gcs_ind, gcs in enumerate(self._sources.values()):
            actual_rad = gcs.get_radius(rad_name, 'kpc')
            try:
                mass_profs = gcs.get_hydrostatic_mass_profiles(temp_model_name=temp_model_name,
                                                               dens_model_name=dens_model_name)
                if isinstance(mass_profs, list):
                    raise ValueError("There are multiple matching hydrostatic mass profiles associated with {}, "
                                     "you will have to retrieve masses manually.")
                else:
                    try:
                        cur_mass = mass_profs.mass(actual_rad)[0]
                        if quality_checks and (cur_mass[1] > cur_mass[0] or cur_mass[2] > cur_mass[0]):
                            ms.append([np.NaN, np.NaN, np.NaN])
                            warn("{s}'s mass uncertainties are larger than the mass value.")
                        elif quality_checks and cur_mass[0] < Quantity(1e+12, 'Msun'):
                            ms.append([np.NaN, np.NaN, np.NaN])
                            warn("{s}'s mass is less than 1e+12 solar masses")
                        elif quality_checks and cur_mass[0] > Quantity(1e+16, 'Msun'):
                            ms.append([np.NaN, np.NaN, np.NaN])
                            warn("{s}'s mass is greater than 1e+16 solar masses")
                        else:
                            ms.append(cur_mass.value)
                    except ValueError:
                        warn("{s}'s mass is negative")
                        ms.append([np.NaN, np.NaN, np.NaN])

            except NoProductAvailableError:
                # If no dens_prof has been run or something goes wrong then NaNs are added
                ms.append([np.NaN, np.NaN, np.NaN])
                warn("{s} doesn't have a matching hydrostatic mass profile associated".format(s=gcs.name))

        ms = np.array(ms)
        # We're going to throw an error if all the masses are NaN, because obviously something is wrong
        check_ms = ms[~np.isnan(ms)]
        if len(check_ms) == 0:
            raise ValueError("All hydrostatic masses appear to be NaN.")

        return Quantity(ms, 'Msun')

    def gm_richness(self, rad_name: str, dens_model: str, prof_outer_rad: Union[Quantity, str], dens_method: str,
                    x_norm: Quantity = Quantity(60), y_norm: Quantity = Quantity(1e+12, 'solMass'),
                    fit_method: str = 'odr', start_pars: list = None, pix_step: int = 1,
                    min_snr: Union[float, int] = 0.0, psf_corr: bool = True, psf_model: str = "ELLBETA",
                    psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, set_ids: List[int] = None,
                    inv_efunc: bool = False) -> ScalingRelation:
        """
        This generates a Gas Mass vs Richness scaling relation for this sample of Galaxy Clusters.

        :param str rad_name: The name of the radius (e.g. r500) to measure gas mass within.
        :param str dens_model: The model fit to the density profiles to be used to calculate gas mass. If a fit
            doesn't already exist then one will be performed with default settings.
        :param Quantity/str prof_outer_rad: The outer radii of the density profiles, either a single radius name or a
            Quantity containing an outer radius for each cluster. For instance if you defined a ClusterSample called
            srcs you could pas srcs.r500 here.
        :param str dens_method: The method used to generate the density profile. For a profile created by fitting a
            model to a surface brightness profile this should be the name of the model, for a profile from
            annular spectra this should be 'spec', and for a profile generated directly from the data of a surface
            brightness profile this should be 'onion'.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str fit_method: The name of the fit method to use to generate the scaling relation.
        :param list start_pars: The start parameters for the fit run.
        :param int pix_step: The width of each annulus in pixels used to generate the profile, for profiles based on
            surface brightness.
        :param float min_snr: The minimum signal to noise imposed upon the profile, for profiles based on
            surface brightness.
        :param bool psf_corr: Is the brightness profile corrected for PSF effects, for profiles based on
            surface brightness.
        :param str psf_model: If PSF corrected, the PSF model used, for profiles based on surface brightness.
        :param int psf_bins: If PSF corrected, the number of bins per side, for profiles based on surface brightness.
        :param str psf_algo: If PSF corrected, the algorithm used, for profiles based on surface brightness.
        :param int psf_iter: If PSF corrected, the number of algorithm iterations, for profiles based on
            surface brightness.
        :param List[int] set_ids: A list of AnnularSpectra set IDs used to generate the density profiles, if you wish
            to use spectrum based density profiles here.
        :param bool inv_efunc: Should the inverse E(z) function be applied to the y-axis, if False then the
            non-inverse will be applied.
        :return: The XGA ScalingRelation object generated for this sample.
        :rtype: ScalingRelation
        """
        if rad_name in ['r200', 'r500', 'r2500']:
            rn = rad_name[1:]
        else:
            rn = rad_name

        if inv_efunc:
            y_name = "E(z)$^{-1}$M$_{g," + rn + "}$"
            e_factor = self.cosmo.inv_efunc(self.redshifts)
        else:
            y_name = "E(z)M$_{g," + rn + "}$"
            e_factor = self.cosmo.efunc(self.redshifts)

        # Just make sure fit method is lower case
        fit_method = fit_method.lower()

        # Read out the richness values into variables just for convenience sake
        r_data = self.richness[:, 0]
        r_errs = self.richness[:, 1]

        # Read out the gas mass values, and multiply by the inverse e function for each cluster
        gm_vals = self.gas_mass(rad_name, dens_model, prof_outer_rad, dens_method, pix_step, min_snr, psf_corr,
                                psf_model, psf_bins, psf_algo, psf_iter, set_ids)
        gm_vals *= e_factor[..., None]
        gm_data = gm_vals[:, 0]
        gm_err = gm_vals[:, 1:]

        if fit_method == 'curve_fit':
            scale_rel = scaling_relation_curve_fit(power_law, gm_data, gm_err, r_data, r_errs, y_norm, x_norm,
                                                   start_pars=start_pars, y_name=y_name, x_name=r"$\lambda$")
        elif fit_method == 'odr':
            scale_rel = scaling_relation_odr(power_law, gm_data, gm_err, r_data, r_errs, y_norm, x_norm,
                                             start_pars=start_pars, y_name=y_name, x_name=r"$\lambda$")
        elif fit_method == 'lira':
            scale_rel = scaling_relation_lira(gm_data, gm_err, r_data, r_errs, y_norm, x_norm,
                                              y_name=y_name, x_name=r"$\lambda$")
        elif fit_method == 'emcee':
            scaling_relation_emcee()
        else:
            raise ValueError('{e} is not a valid fitting method, please choose one of these: '
                             '{a}'.format(e=fit_method, a=' '.join(ALLOWED_FIT_METHODS)))

        return scale_rel

    # I don't allow the user to supply an inner radius here because I cannot think of a reason why you'd want to
    #  make a scaling relation with a core excised temperature.
    def gm_Tx(self, rad_name: str, dens_model: str, prof_outer_rad: Union[Quantity, str], dens_method: str,
              x_norm: Quantity = Quantity(4, 'keV'), y_norm: Quantity = Quantity(1e+12, 'solMass'),
              fit_method: str = 'odr', start_pars: list = None, model: str = 'constant*tbabs*apec',
              group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
              pix_step: int = 1, min_snr: Union[float, int] = 0.0, psf_corr: bool = True, psf_model: str = "ELLBETA",
              psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, set_ids: List[int] = None,
              inv_efunc: bool = False) -> ScalingRelation:
        """
        This generates a Gas Mass vs Tx scaling relation for this sample of Galaxy Clusters.

        :param str rad_name: The name of the radius (e.g. r500) to get values for.
        :param str dens_model: The model fit to the density profiles to be used to calculate gas mass. If a fit
            doesn't already exist then one will be performed with default settings.
        :param Quantity/str prof_outer_rad: The outer radii of the density profiles, either a single radius name or a
            Quantity containing an outer radius for each cluster. For instance if you defined a ClusterSample called
            srcs you could pas srcs.r500 here.
        :param str dens_method: The method used to generate the density profile. For a profile created by fitting a
            model to a surface brightness profile this should be the name of the model, for a profile from
            annular spectra this should be 'spec', and for a profile generated directly from the data of a surface
            brightness profile this should be 'onion'.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str fit_method: The name of the fit method to use to generate the scaling relation.
        :param list start_pars: The start parameters for the fit run.
        :param str model: The name of the model that the temperatures were measured with.
        :param bool group_spec: Whether the spectra that were fitted for the Tx values were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            Tx values were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            Tx values were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param int pix_step: The width of each annulus in pixels used to generate the profile, for profiles based on
            surface brightness.
        :param float min_snr: The minimum signal to noise imposed upon the profile, for profiles based on
            surface brightness.
        :param bool psf_corr: Is the brightness profile corrected for PSF effects, for profiles based on
            surface brightness.
        :param str psf_model: If PSF corrected, the PSF model used, for profiles based on surface brightness.
        :param int psf_bins: If PSF corrected, the number of bins per side, for profiles based on surface brightness.
        :param str psf_algo: If PSF corrected, the algorithm used, for profiles based on surface brightness.
        :param int psf_iter: If PSF corrected, the number of algorithm iterations, for profiles based on
            surface brightness.
        :param List[int] set_ids: A list of AnnularSpectra set IDs used to generate the density profiles, if you wish
            to use spectrum based density profiles here.
        :return: The XGA ScalingRelation object generated for this sample.
        :param bool inv_efunc: Should the inverse E(z) function be applied to the y-axis, if False then the
            non-inverse will be applied.
        :rtype: ScalingRelation
        """

        if rad_name in ['r200', 'r500', 'r2500']:
            rn = rad_name[1:]
        else:
            rn = rad_name

        if inv_efunc:
            e_factor = self.cosmo.inv_efunc(self.redshifts)
            y_name = r"E(z)$^{-1}$M$_{\rm{g}," + rn + "}$"
        else:
            e_factor = self.cosmo.efunc(self.redshifts)
            y_name = r"E(z)M$_{\rm{g}," + rn + "}$"

        # Just make sure fit method is lower case
        fit_method = fit_method.lower()

        # Read out the temperature values into variables just for convenience sake
        t_vals = self.Tx(rad_name, model, Quantity(0, 'deg'), group_spec, min_counts, min_sn, over_sample)
        t_data = t_vals[:, 0]
        t_errs = t_vals[:, 1:]

        # Read out the mass values, and multiply by the inverse e function for each cluster
        gm_vals = self.gas_mass(rad_name, dens_model, prof_outer_rad, dens_method, pix_step, min_snr, psf_corr,
                                psf_model, psf_bins, psf_algo, psf_iter, set_ids)
        gm_vals *= e_factor[..., None]
        gm_data = gm_vals[:, 0]
        gm_err = gm_vals[:, 1:]

        x_name = r"T$_{\rm{x}," + rn + '}$'
        if fit_method == 'curve_fit':
            scale_rel = scaling_relation_curve_fit(power_law, gm_data, gm_err, t_data, t_errs, y_norm, x_norm,
                                                   start_pars=start_pars, y_name=y_name, x_name=x_name)
        elif fit_method == 'odr':
            scale_rel = scaling_relation_odr(power_law, gm_data, gm_err, t_data, t_errs, y_norm, x_norm,
                                             start_pars=start_pars, y_name=y_name, x_name=x_name)
        elif fit_method == 'lira':
            scale_rel = scaling_relation_lira(gm_data, gm_err, t_data, t_errs, y_norm, x_norm,
                                              y_name=y_name, x_name=x_name)
        elif fit_method == 'emcee':
            scaling_relation_emcee()
        else:
            raise ValueError('{e} is not a valid fitting method, please choose one of these: '
                             '{a}'.format(e=fit_method, a=' '.join(ALLOWED_FIT_METHODS)))

        return scale_rel

    def Lx_richness(self, outer_radius: str = 'r500', x_norm: Quantity = Quantity(60),
                    y_norm: Quantity = Quantity(1e+44, 'erg/s'), fit_method: str = 'odr', start_pars: list = None,
                    model: str = 'constant*tbabs*apec', lo_en: Quantity = Quantity(0.5, 'keV'),
                    hi_en: Quantity = Quantity(2.0, 'keV'), inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                    group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                    over_sample: float = None, inv_efunc: bool = True) -> ScalingRelation:
        """
        This generates a Lx vs richness scaling relation for this sample of Galaxy Clusters. If you have run fits
        to find core excised luminosity, and wish to use it in this scaling relation, then please don't forget
        to supply an inner_radius to the method call.

        :param str outer_radius: The name of the radius (e.g. r500) to get values for.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str fit_method: The name of the fit method to use to generate the scaling relation.
        :param list start_pars: The start parameters for the fit run.
        :param str model: The name of the model that the luminosities were measured with.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum. You may also pass a quantity containing radius values, with one value for each
            source in this sample.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :return: The XGA ScalingRelation object generated for this sample.
        :param bool inv_efunc: Should the inverse E(z) function be applied to the y-axis, if False then the
            non-inverse will be applied.
        :rtype: ScalingRelation
        """
        if outer_radius in ['r200', 'r500', 'r2500']:
            rn = outer_radius[1:]
        else:
            raise ValueError("As this is a method for a whole population, please use a named radius such as "
                             "r200, r500, or r2500.")

        if inv_efunc:
            y_name = "E(z)$^{-1}$L$_{x," + rn + ',' + str(lo_en.value) + '-' + str(hi_en.value) + "}$"
            e_factor = self.cosmo.inv_efunc(self.redshifts)
        else:
            y_name = "E(z)L$_{x," + rn + ',' + str(lo_en.value) + '-' + str(hi_en.value) + "}$"
            e_factor = self.cosmo.efunc(self.redshifts)

        # Just make sure fit method is lower case
        fit_method = fit_method.lower()

        # Read out the richness values into variables just for convenience sake
        r_data = self.richness[:, 0]
        r_errs = self.richness[:, 1]

        # Read out the luminosity values, and multiply by the inverse e function for each cluster
        lx_vals = self.Lx(outer_radius, model, inner_radius, lo_en, hi_en, group_spec, min_counts, min_sn,
                          over_sample) * e_factor[..., None]
        lx_data = lx_vals[:, 0]
        lx_err = lx_vals[:, 1:]

        if fit_method == 'curve_fit':
            scale_rel = scaling_relation_curve_fit(power_law, lx_data, lx_err, r_data, r_errs, y_norm, x_norm,
                                                   start_pars=start_pars, y_name=y_name,
                                                   x_name=r"$\lambda$")
        elif fit_method == 'odr':
            scale_rel = scaling_relation_odr(power_law, lx_data, lx_err, r_data, r_errs, y_norm, x_norm,
                                             start_pars=start_pars, y_name=y_name, x_name=r"$\lambda$")
        elif fit_method == 'lira':
            scale_rel = scaling_relation_lira(lx_data, lx_err, r_data, r_errs, y_norm, x_norm,
                                              y_name=y_name, x_name=r"$\lambda$")
        elif fit_method == 'emcee':
            scaling_relation_emcee()
        else:
            raise ValueError('{e} is not a valid fitting method, please choose one of these: '
                             '{a}'.format(e=fit_method, a=' '.join(ALLOWED_FIT_METHODS)))

        return scale_rel

    def Lx_Tx(self, outer_radius: str = 'r500', x_norm: Quantity = Quantity(4, 'keV'),
              y_norm: Quantity = Quantity(1e+44, 'erg/s'), fit_method: str = 'odr', start_pars: list = None,
              model: str = 'constant*tbabs*apec', lo_en: Quantity = Quantity(0.5, 'keV'),
              hi_en: Quantity = Quantity(2.0, 'keV'), tx_inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
              lx_inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
              min_counts: int = 5, min_sn: float = None, over_sample: float = None,
              inv_efunc: bool = True) -> ScalingRelation:
        """
        This generates a Lx vs Tx scaling relation for this sample of Galaxy Clusters. If you have run fits
        to find core excised luminosity, and wish to use it in this scaling relation, then you can specify the inner
        radius of those spectra using lx_inner_radius, as well as ensuring that you use the temperature
        fit you want by setting tx_inner_radius.

        :param str outer_radius: The name of the radius (e.g. r500) to get values for.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str fit_method: The name of the fit method to use to generate the scaling relation.
        :param list start_pars: The start parameters for the fit run.
        :param str model: The name of the model that the luminosities and temperatures were measured with.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param str/Quantity tx_inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the temperature (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum. You may also pass a quantity containing radius values, with one value for each
            source in this sample.
        :param str/Quantity lx_inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the Lx. The same rules as tx_inner_radius apply, and this option
            is particularly useful if you have measured core-excised luminosity an wish to use it in a scaling relation.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param bool inv_efunc: Should the inverse E(z) function be applied to the y-axis, if False then the
            non-inverse will be applied.
        :return: The XGA ScalingRelation object generated for this sample.
        :rtype: ScalingRelation
        """

        if outer_radius in ['r200', 'r500', 'r2500']:
            rn = outer_radius[1:]
        else:
            raise ValueError("As this is a method for a whole population, please use a named radius such as "
                             "r200, r500, or r2500.")

        if lx_inner_radius.value != 0:
            lx_rn = "Core-Excised " + rn
        else:
            lx_rn = rn

        if inv_efunc:
            y_name = "E(z)$^{-1}$L$_{x," + lx_rn + ',' + str(lo_en.value) + '-' + str(hi_en.value) + "}$"
            e_factor = self.cosmo.inv_efunc(self.redshifts)
        else:
            y_name = "E(z)L$_{x," + lx_rn + ',' + str(lo_en.value) + '-' + str(hi_en.value) + "}$"
            e_factor = self.cosmo.efunc(self.redshifts)

        # Just make sure fit method is lower case
        fit_method = fit_method.lower()

        # Read out the luminosity values, and multiply by the inverse e function for each cluster
        lx_vals = self.Lx(outer_radius, model, lx_inner_radius, lo_en, hi_en, group_spec, min_counts, min_sn,
                          over_sample) * e_factor[..., None]
        lx_data = lx_vals[:, 0]
        lx_err = lx_vals[:, 1:]

        # Read out the temperature values into variables just for convenience sake
        t_vals = self.Tx(outer_radius, model, tx_inner_radius, group_spec, min_counts, min_sn, over_sample)
        t_data = t_vals[:, 0]
        t_errs = t_vals[:, 1:]

        x_name = r"T$_{x," + rn + '}$'
        if fit_method == 'curve_fit':
            scale_rel = scaling_relation_curve_fit(power_law, lx_data, lx_err, t_data, t_errs, y_norm, x_norm,
                                                   start_pars=start_pars, y_name=y_name,
                                                   x_name=x_name)
        elif fit_method == 'odr':
            scale_rel = scaling_relation_odr(power_law, lx_data, lx_err, t_data, t_errs, y_norm, x_norm,
                                             start_pars=start_pars, y_name=y_name, x_name=x_name)
        elif fit_method == 'lira':
            scale_rel = scaling_relation_lira(lx_data, lx_err, t_data, t_errs, y_norm, x_norm,
                                              y_name=y_name, x_name=x_name)
        elif fit_method == 'emcee':
            scaling_relation_emcee()
        else:
            raise ValueError('{e} is not a valid fitting method, please choose one of these: '
                             '{a}'.format(e=fit_method, a=' '.join(ALLOWED_FIT_METHODS)))

        return scale_rel

    def mass_Tx(self, outer_radius: str = 'r500', x_norm: Quantity = Quantity(4, 'keV'),
                y_norm: Quantity = Quantity(5e+14, 'Msun'), fit_method: str = 'odr', start_pars: list = None,
                model: str = 'constant*tbabs*apec', tx_inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                temp_model_name: str = None, dens_model_name: str = None, inv_efunc: bool = False) -> ScalingRelation:
        """
        A convenience function to generate a hydrostatic mass-temperature relation for this sample of galaxy clusters.

        :param str outer_radius: The outer radius of the region used to measure temperature and the radius
            out to which you wish to measure mass.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str fit_method: The name of the fit method to use to generate the scaling relation.
        :param list start_pars: The start parameters for the fit run.
        :param str model: The name of the model that the luminosities and temperatures were measured with.
        :param str/Quantity tx_inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the temperature (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum. You may also pass a quantity containing radius values, with one value for each
            source in this sample.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param str temp_model_name: The name of the model used to fit the temperature profile used to generate the
            required hydrostatic mass profile, default is None.
        :param str dens_model_name: The name of the model used to fit the density profile used to generate the
            required hydrostatic mass profile, default is None.
        :param bool inv_efunc: Should the inverse E(z) function be applied to the y-axis, if False then the
            non-inverse will be applied.
        :return: The XGA ScalingRelation object generated for this sample.
        :rtype: ScalingRelation
        """

        if outer_radius in ['r200', 'r500', 'r2500']:
            rn = outer_radius[1:]
        else:
            raise ValueError("As this is a method for a whole population, please use a named radius such as "
                             "r200, r500, or r2500.")

        if inv_efunc:
            y_name = r"E(z)$^{-1}$M$_{\rm{hydro," + rn + "}}$"
            e_factor = self.cosmo.inv_efunc(self.redshifts)
        else:
            y_name = r"E(z)M$_{\rm{hydro," + rn + "}}$"
            e_factor = self.cosmo.efunc(self.redshifts)

        # Just make sure fit method is lower case
        fit_method = fit_method.lower()

        # Read out the luminosity values, and multiply by the inverse e function for each cluster
        m_vals = self.hydrostatic_mass(outer_radius, temp_model_name, dens_model_name) * e_factor[..., None]
        m_data = m_vals[:, 0]
        m_err = m_vals[:, 1:]

        # Read out the temperature values into variables just for convenience sake
        t_vals = self.Tx(outer_radius, model, tx_inner_radius, group_spec, min_counts, min_sn, over_sample)
        t_data = t_vals[:, 0]
        t_errs = t_vals[:, 1:]

        x_name = r"T$_{\rm{x," + rn + '}}$'
        if fit_method == 'curve_fit':
            scale_rel = scaling_relation_curve_fit(power_law, m_data, m_err, t_data, t_errs, y_norm, x_norm,
                                                   start_pars=start_pars, y_name=y_name,
                                                   x_name=x_name)
        elif fit_method == 'odr':
            scale_rel = scaling_relation_odr(power_law, m_data, m_err, t_data, t_errs, y_norm, x_norm,
                                             start_pars=start_pars, y_name=y_name, x_name=x_name)
        elif fit_method == 'lira':
            scale_rel = scaling_relation_lira(m_data, m_err, t_data, t_errs, y_norm, x_norm,
                                              y_name=y_name, x_name=x_name)
        elif fit_method == 'emcee':
            scaling_relation_emcee()
        else:
            raise ValueError('{e} is not a valid fitting method, please choose one of these: '
                             '{a}'.format(e=fit_method, a=' '.join(ALLOWED_FIT_METHODS)))

        return scale_rel

    def mass_richness(self, outer_radius: str = 'r500', x_norm: Quantity = Quantity(60),
                      y_norm: Quantity = Quantity(5e+14, 'Msun'), fit_method: str = 'odr', start_pars: list = None,
                      temp_model_name: str = None, dens_model_name: str = None, inv_efunc: bool = False) -> ScalingRelation:
        """
        A convenience function to generate a hydrostatic mass-richness relation for this sample of galaxy clusters.

        :param str outer_radius: The name of the radius (e.g. r500) to get values for.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str fit_method: The name of the fit method to use to generate the scaling relation.
        :param list start_pars: The start parameters for the fit run.
        :param str temp_model_name: The name of the model used to fit the temperature profile used to generate the
            required hydrostatic mass profile, default is None.
        :param str dens_model_name: The name of the model used to fit the density profile used to generate the
            required hydrostatic mass profile, default is None.
        :param bool inv_efunc: Should the inverse E(z) function be applied to the y-axis, if False then the
            non-inverse will be applied.
        :return: The XGA ScalingRelation object generated for this sample.
        :rtype: ScalingRelation
        """
        if outer_radius in ['r200', 'r500', 'r2500']:
            rn = outer_radius[1:]
        else:
            raise ValueError("As this is a method for a whole population, please use a named radius such as "
                             "r200, r500, or r2500.")

        if inv_efunc:
            y_name = r"E(z)$^{-1}$M$_{\rm{hydro," + rn + "}}$"
            e_factor = self.cosmo.inv_efunc(self.redshifts)
        else:
            y_name = r"E(z)M$_{\rm{hydro," + rn + "}}$"
            e_factor = self.cosmo.efunc(self.redshifts)

        # Just make sure fit method is lower case
        fit_method = fit_method.lower()

        # Read out the richness values into variables just for convenience sake
        r_data = self.richness[:, 0]
        r_errs = self.richness[:, 1]

        # Read out the luminosity values, and multiply by the inverse e function for each cluster
        m_vals = self.hydrostatic_mass(outer_radius, temp_model_name, dens_model_name) * e_factor[..., None]
        m_data = m_vals[:, 0]
        m_err = m_vals[:, 1:]

        if fit_method == 'curve_fit':
            scale_rel = scaling_relation_curve_fit(power_law, m_data, m_err, r_data, r_errs, y_norm, x_norm,
                                                   start_pars=start_pars, y_name=y_name,
                                                   x_name=r"$\lambda$")
        elif fit_method == 'odr':
            scale_rel = scaling_relation_odr(power_law, m_data, m_err, r_data, r_errs, y_norm, x_norm,
                                             start_pars=start_pars, y_name=y_name, x_name=r"$\lambda$")
        elif fit_method == 'lira':
            scale_rel = scaling_relation_lira(m_data, m_err, r_data, r_errs, y_norm, x_norm,
                                              y_name=y_name, x_name=r"$\lambda$")
        elif fit_method == 'emcee':
            scaling_relation_emcee()
        else:
            raise ValueError('{e} is not a valid fitting method, please choose one of these: '
                             '{a}'.format(e=fit_method, a=' '.join(ALLOWED_FIT_METHODS)))

        return scale_rel

    def mass_Lx(self, outer_radius: str = 'r500', x_norm: Quantity = Quantity(1e+44, 'erg/s'),
                y_norm: Quantity = Quantity(5e+14, 'Msun'), fit_method: str = 'odr', start_pars: list = None,
                model: str = 'constant*tbabs*apec', lo_en: Quantity = Quantity(0.5, 'keV'),
                hi_en: Quantity = Quantity(2.0, 'keV'), lx_inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
                group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                temp_model_name: str = None, dens_model_name: str = None, inv_efunc: bool = False) -> ScalingRelation:
        """
        This generates a mass vs Lx scaling relation for this sample of Galaxy Clusters. If you have run fits
        to find core excised luminosity, and wish to use it in this scaling relation, then you can specify the inner
        radius of those spectra using lx_inner_radius.

        :param str outer_radius: The name of the radius (e.g. r500) to get values for.
        :param Quantity x_norm: Quantity to normalise the x data by.
        :param Quantity y_norm: Quantity to normalise the y data by.
        :param str fit_method: The name of the fit method to use to generate the scaling relation.
        :param list start_pars: The start parameters for the fit run.
        :param str model: The name of the model that the luminosities and temperatures were measured with.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param str/Quantity lx_inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the Lx. The same rules as tx_inner_radius apply, and this option
            is particularly useful if you have measured core-excised luminosity an wish to use it in a scaling relation.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param str temp_model_name: The name of the model used to fit the temperature profile used to generate the
            required hydrostatic mass profile, default is None.
        :param str dens_model_name: The name of the model used to fit the density profile used to generate the
            required hydrostatic mass profile, default is None.
        :param bool inv_efunc: Should the inverse E(z) function be applied to the y-axis, if False then the
            non-inverse will be applied.
        :return: The XGA ScalingRelation object generated for this sample.
        :rtype: ScalingRelation
        """
        if outer_radius in ['r200', 'r500', 'r2500']:
            rn = outer_radius[1:]
        else:
            raise ValueError("As this is a method for a whole population, please use a named radius such as "
                             "r200, r500, or r2500.")

        if lx_inner_radius.value != 0:
            lx_rn = "Core-Excised " + rn
        else:
            lx_rn = rn

        if inv_efunc:
            y_name = r"E(z)$^{-1}$M$_{\rm{hydro," + rn + "}}$"
            e_factor = self.cosmo.inv_efunc(self.redshifts)
        else:
            y_name = r"E(z)M$_{\rm{hydro," + rn + "}}$"
            e_factor = self.cosmo.efunc(self.redshifts)

        # Just make sure fit method is lower case
        fit_method = fit_method.lower()

        # Read out the luminosity values, and multiply by the inverse e function for each cluster
        m_vals = self.hydrostatic_mass(outer_radius, temp_model_name, dens_model_name) * e_factor[..., None]
        m_data = m_vals[:, 0]
        m_err = m_vals[:, 1:]

        # Read out the luminosity values, and multiply by the inverse e function for each cluster
        lx_vals = self.Lx(outer_radius, model, lx_inner_radius, lo_en, hi_en, group_spec, min_counts, min_sn,
                          over_sample)
        lx_data = lx_vals[:, 0]
        lx_err = lx_vals[:, 1:]

        x_name = r"L$_{\rm{x," + lx_rn + ',' + str(lo_en.value) + '-' + str(hi_en.value) + "}}$"
        if fit_method == 'curve_fit':
            scale_rel = scaling_relation_curve_fit(power_law, m_data, m_err, lx_data, lx_err, y_norm, x_norm,
                                                   start_pars=start_pars, y_name=y_name,
                                                   x_name=x_name)
        elif fit_method == 'odr':
            scale_rel = scaling_relation_odr(power_law, m_data, m_err, lx_data, lx_err, y_norm, x_norm,
                                             start_pars=start_pars, y_name=y_name, x_name=x_name)
        elif fit_method == 'lira':
            scale_rel = scaling_relation_lira(m_data, m_err, lx_data, lx_err, y_norm, x_norm,
                                              y_name=y_name, x_name=x_name)
        elif fit_method == 'emcee':
            scaling_relation_emcee()
        else:
            raise ValueError('{e} is not a valid fitting method, please choose one of these: '
                             '{a}'.format(e=fit_method, a=' '.join(ALLOWED_FIT_METHODS)))

        return scale_rel
