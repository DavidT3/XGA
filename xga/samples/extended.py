#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 08/12/2020, 13:21. Copyright (c) David J Turner

from warnings import warn

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity
from tqdm import tqdm

from .base import BaseSample
from ..exceptions import PeakConvergenceFailedError, ModelNotAssociatedError, ParameterNotAssociatedError
from ..imagetools.psf import rl_psf
from ..sources.extended import GalaxyCluster


# Names are required for the ClusterSample because they'll be used to access specific cluster objects
class ClusterSample(BaseSample):
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
        # TODO Make this logging rather than just printing
        print("Pre-generating necessary products")
        evselect_image(self, peak_lo_en, peak_hi_en)
        eexpmap(self, peak_lo_en, peak_hi_en)
        emosaic(self, "image", peak_lo_en, peak_hi_en)
        emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        # Now that we've made those images the BaseSource objects aren't required anymore, we're about
        #  to define GalaxyClusters
        del self._sources
        self._sources = {}

        dec_lb = tqdm(desc="Setting up Galaxy Clusters", total=len(self.names), disable=no_prog_bar)
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
                except PeakConvergenceFailedError:
                    warn("The peak finding algorithm has not converged for {}, using user "
                         "supplied coordinates".format(n))
                    self._sources[n] = GalaxyCluster(r, d, z, n, r2, r5, r25, lam, lam_err, wlm, wlm_err, cr, False,
                                                     peak_lo_en, peak_hi_en, back_inn_rad_factor, back_out_rad_factor,
                                                     cosmology, True, load_fits, clean_obs, clean_obs_reg,
                                                     clean_obs_threshold, False)

            dec_lb.update(1)
        dec_lb.close()

        # And again I ask XGA to generate the merged images and exposure maps, in case any sources have been
        #  cleaned and had data removed
        if clean_obs:
            emosaic(self, "image", peak_lo_en, peak_hi_en)
            emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        # TODO Reconsider if this is even necessary, the data that has been removed should by definition
        #  not really include the peak
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
                    except PeakConvergenceFailedError:
                        pass

        # I don't offer the user choices as to the configuration for PSF correction at the moment
        if psf_corr:
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

    def Tx(self, reg_type: str, model: str = 'tbabs*apec'):
        """
        A get method for temperatures measured for the constituent clusters of this sample. An error will be
        thrown if temperatures haven't been measured for the given region and model (default is the tbabs*apec model
        which single_temp_apec fits to cluster spectra). Any clusters for which temperature fits failed will return
        NaN temperatures.
        :param str reg_type: The type of region that the fitted spectra were generated from.
        :param str model: The name of the fitted model that you're requesting the results from (e.g. tbabs*apec).
        :return: An Nx3 array Quantity where N is the number of clusters. First column is the temperature, second
        column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN.
        :rtype: Quantity
        """
        temps = []
        for gcs in self._sources.values():
            try:
                # Fetch the temperature from a given cluster using the dedicated method
                gcs_temp = gcs.get_temperature(reg_type, model).value

                # If the measured temperature is 64keV I know that's a failure condition of the XSPEC fit,
                #  so its set to NaN
                if gcs_temp[0] == 64:
                    gcs_temp = np.array([np.NaN, np.NaN, np.NaN])
                    warn("A temperature of 64keV was measured for {s}, this is considered a failed fit by "
                         "XGA".format(s=gcs.name))
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

    def gas_mass(self, rad_name: str, dens_tech: str = 'inv_abel_model', conf_level: int = 90) -> Quantity:
        """
        A get method for gas masses measured for the constituent clusters of this sample.
        :param str rad_name: The name of the radius (e.g. r500) to calculate the gas mass within.
        :param str dens_tech: The technique used to generate the density profile, default is 'inv_abel_model',
        which is the superior of the two I have implemented as of 03/12/20.
        :param int conf_level: The desired confidence level of the uncertainties.
        :return: An Nx3 array Quantity where N is the number of clusters. First column is the gas mass, second
        column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN.
        :rtype: Quantity
        """
        gms = []

        # Iterate through all of our Galaxy Clusters
        for gcs in self._sources.values():
            dens_profs = gcs.get_products('combined_gas_density_profile')
            if len(dens_profs) == 0:
                # If no dens_prof has been run or something goes wrong then NaNs are added
                gms.append([np.NaN, np.NaN, np.NaN])
                warn("{s} doesn't have a density profile associated, please look at "
                     "sourcetools.density.".format(s=gcs.name))
            elif len(dens_profs) != 0:
                # This is because I store the profile products in a really dumb way which I'm going to need to
                #  correct - but for now this will do
                dens_prof = dens_profs[0][0]
                # Use the density profiles gas mass method to calculate the one we want
                gm = dens_prof.gas_mass(dens_tech, gcs.get_radius(rad_name, 'kpc'), conf_level)[0].value
                gms.append(gm)

            if len(dens_profs) > 1:
                warn("{s} has multiple density profiles associated with it, and until I upgrade XGA I can't"
                     " really tell them apart so I'm just taking the first one! I will fix this".format(s=gcs.name))

        gms = np.array(gms)

        # We're going to throw an error if all the gas masses are NaN, because obviously something is wrong
        check_gms = gms[~np.isnan(gms)]
        if len(check_gms) == 0:
            raise ValueError("All gas masses appear to be NaN.")

        return Quantity(gms, 'Msun')

    def gm_richness(self, fit_method: str, rad_name: str, dens_tech: str = 'inv_abel_model', conf_level: int = 90):
        pass

    def gm_Tx(self, fit_method: str):
        pass

    def Tx_richness(self, fit_method: str):
        pass

    def Lx_richness(self, fit_method: str):
        pass






