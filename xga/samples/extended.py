#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 12/11/2020, 15:40. Copyright (c) David J Turner

from warnings import warn

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity
from tqdm import tqdm

from .base import BaseSample
from ..exceptions import PeakConvergenceFailedError
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
    def r200_snrs(self) -> np.ndarray:
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
    def r500_snrs(self) -> np.ndarray:
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
    def r2500_snrs(self) -> np.ndarray:
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





