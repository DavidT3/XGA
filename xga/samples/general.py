#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 24/05/2021, 13:34. Copyright (c) David J Turner

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity, Unit
from tqdm import tqdm

from .base import BaseSample
from ..exceptions import NoValidObservationsError
from ..sources.general import PointSource


class PointSample(BaseSample):
    """
    The sample class for general point sources, without the extra information required to analyse more specific
    X-ray point sources.
    """
    def __init__(self, ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray = None, name: np.ndarray = None,
                 point_radius: Quantity = Quantity(30, 'arcsec'), use_peak: bool = False,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5,
                 cosmology=Planck15, load_fits: bool = False, no_prog_bar: bool = False, psf_corr: bool = False):

        # Strongly enforce that its a quantity, this also means that it should be guaranteed that all radii have
        #  a single unit
        if not isinstance(point_radius, Quantity):
            raise TypeError("Please pass a quantity object for point_radius, rather than an array or list.")
        # People might pass a single value for point_radius, in which case we turn it into a non-scalar quantity
        elif point_radius.isscalar:
            point_radius = Quantity([point_radius.value]*len(ra), point_radius.unit)
        elif not point_radius.isscalar and len(point_radius) != len(ra):
            raise ValueError("If you pass a set of radii (rather than a single radius) to point_radius then there"
                             " must be one entry per object passed to this sample object.")

        # I don't like having this here, but it does avoid a circular import problem
        from xga.sas import evselect_image, eexpmap, emosaic

        # Using the super defines BaseSources and stores them in the self._sources dictionary
        super().__init__(ra, dec, redshift, name, cosmology, load_products=True, load_fits=False,
                         no_prog_bar=no_prog_bar)

        evselect_image(self, peak_lo_en, peak_hi_en)
        eexpmap(self, peak_lo_en, peak_hi_en)
        emosaic(self, "image", peak_lo_en, peak_hi_en)
        emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        del self._sources
        self._sources = {}

        # A list to store the inds of any declarations that failed due to NoValidObservations, so some
        #  attributes can be cleaned up later
        final_names = []
        self._point_radii = []
        with tqdm(desc="Setting up Point Sources", total=len(self._accepted_inds), disable=no_prog_bar) as dec_lb:
            for ind in range(len(self._accepted_inds)):
                r, d = ra[self._accepted_inds[ind]], dec[self._accepted_inds[ind]]
                if redshift is None:
                    z = None
                else:
                    z = redshift[self._accepted_inds[ind]]
                n = self._names[ind]
                pr = point_radius[self._accepted_inds[ind]]

                # Observation cleaning goes on automatically in PointSource, so if a NoValidObservations error is
                #  thrown I have to catch it and not add that source to this sample.
                try:
                    self._sources[n] = PointSource(r, d, z, n, pr, use_peak, peak_lo_en, peak_hi_en,
                                                   back_inn_rad_factor, back_out_rad_factor, cosmology, True,
                                                   load_fits, False)
                    self._point_radii.append(pr.value)
                    # I know this will write to this over and over, but it seems a bit silly to check whether this has
                    #  been set yet when all radii should be forced to be the same unit
                    self._pr_unit = pr.unit
                    final_names.append(n)
                except NoValidObservationsError:
                    self._failed_sources[n] = "CleanedNoMatch"

                dec_lb.update(1)
        self._names = final_names

        # I've cleaned the observations, and its possible some of the data has been thrown away,
        #  so I should regenerate the mosaic images/expmaps
        emosaic(self, "image", peak_lo_en, peak_hi_en)
        emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        # I don't offer the user choices as to the configuration for PSF correction at the moment
        if psf_corr:
            # Trying to see if this stops a circular import issue I've been having
            from ..imagetools.psf import rl_psf
            rl_psf(self, lo_en=peak_lo_en, hi_en=peak_hi_en)

    @property
    def point_radii(self) -> Quantity:
        """
        Property getter for the radii of the regions used for analysis of the point sources in this sample.

        :return: A non-scalar Quantity of the point source radii used for analysis of the point sources in
            this sample.
        :rtype: Quantity
        """
        return Quantity(self._point_radii, self._pr_unit)

    @property
    def point_radii_unit(self) -> Unit:
        """
        Property getter for the unit which the point radii values are stored in.

        :return: The unit that the point radii are stored in.
        :rtype: Unit
        """
        return self._pr_unit

    def _del_data(self, key: int):
        """
        Specific to the PointSample class, this deletes the extra data stored during the initialisation
        of this type of sample.

        :param int key: The index or name of the source to delete.
        """
        del self._point_radii[key]








