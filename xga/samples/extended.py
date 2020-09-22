#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 22/09/2020, 13:55. Copyright (c) David J Turner

from astropy.cosmology import Planck15
from astropy.units import Quantity
from numpy import ndarray
from tqdm import tqdm

from .base import BaseSample
from ..sources.extended import GalaxyCluster


# Names are required for the ClusterSample because they'll be used to access specific cluster objects
class ClusterSample(BaseSample):
    def __init__(self, ra: ndarray, dec: ndarray, redshift: ndarray, name: ndarray, r200: Quantity = None,
                 r500: Quantity = None, r2500: Quantity = None, richness: ndarray = None,
                 richness_err: ndarray = None, wl_mass: Quantity = None, wl_mass_err: Quantity = None,
                 custom_region_radius: Quantity = None, use_peak: bool = True,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5, cosmology=Planck15,
                 load_fits: bool = False, clean_obs: bool = True, clean_obs_reg: str = "r500",
                 clean_obs_threshold: float = 0.3, no_prog_bar: bool = True):

        # I don't like having this here, but it does avoid a circular import problem
        from xga.sas import evselect_image, eexpmap, emosaic

        # Using the super defines BaseSources and stores them in the self._sources dictionary
        super().__init__(ra, dec, redshift, name, cosmology, load_products=True, load_fits=False, no_prog_bar=True)

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

        dec_lb = tqdm(desc="Setting up Galaxy Clusters", total=len(ra), disable=no_prog_bar)
        for ind, r in enumerate(ra):
            # Just splitting out relevant values for this particular cluster so the object declaration isn't
            #  super ugly.
            d = dec[ind]
            z = redshift[ind]
            n = name[ind]

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
            self._sources[n] = GalaxyCluster(r, d, z, n, r2, r5, r25, lam, lam_err, wlm, wlm_err, cr, use_peak,
                                             peak_lo_en, peak_hi_en, back_inn_rad_factor, back_out_rad_factor,
                                             cosmology, True, load_fits, clean_obs, clean_obs_reg,
                                             clean_obs_threshold)
            dec_lb.update(1)
        dec_lb.close()





