#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 02/02/2022, 11:37. Copyright (c) The Contributors

from warnings import warn

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity, Unit
from tqdm import tqdm

from .base import BaseSample
from ..exceptions import NoValidObservationsError, PeakConvergenceFailedError
from ..sources.general import PointSource, ExtendedSource


class ExtendedSample(BaseSample):
    """
    The sample class for exploring general extended sources without the extra information required to
    analyse more specific X-ray extended sources (like galaxy clusters).

    :param np.ndarray ra: The right-ascensions of the extended sources, in degrees.
    :param np.ndarray dec: The declinations of the extended sources, in degrees.
    :param np.ndarray redshift: The redshifts of the extended sources, optional. Default is None.
    :param np.ndarray name: The names of the extended sources, optional. If no names are supplied
        then they will be constructed from the supplied coordinates.
    :param Quantity custom_region_radius: Custom analysis region radius(ii) for these sources, optional. Either
        pass a scalar astropy quantity, or a non-scalar astropy quantity with length equal to the number of sources.
    :param bool use_peak: Whether peak positions should be found and used.
    :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak
        position from. Default is 0.5keV.
    :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak
        position from. Default is 2.0keV.
    :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
        radius for the background region. Default is 1.05.
    :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
        radius for the background region. Default is 1.5.
    :param cosmology: An astropy cosmology object for use throughout analysis of the source.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param bool no_prog_bar: Should a source declaration progress bar be shown during setup.
    :param bool psf_corr: Should images be PSF corrected with default settings during sample setup.
    :param str peak_find_method: Which peak finding method should be used (if use_peak is True). Default
        is hierarchical, simple may also be passed.
    """
    def __init__(self, ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray = None, name: np.ndarray = None,
                 custom_region_radius: Quantity = None, use_peak: bool = True,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5, cosmology=Planck15,
                 load_fits: bool = False, no_prog_bar: bool = False, psf_corr: bool = False,
                 peak_find_method: str = "hierarchical"):
        """
        The init method of the ExtendedSample class.
        """
        if custom_region_radius is not None and not isinstance(custom_region_radius, Quantity):
            raise TypeError("Please pass None or a quantity object for custom_region_radius, rather than an "
                            "array or list.")
        elif custom_region_radius is None:
            custom_region_radius = [None]*len(ra)
        # People might pass a single value for custom_region_radius, in which case we turn it into
        #  a non-scalar quantity
        elif custom_region_radius is not None and custom_region_radius.isscalar:
            custom_region_radius = Quantity([custom_region_radius.value]*len(ra), custom_region_radius.unit)
        elif custom_region_radius is not None and not custom_region_radius.isscalar \
                and len(custom_region_radius) != len(ra):
            raise ValueError("If you pass a set of radii (rather than a single radius) to custom_region_radius"
                             " then there must be one entry per object passed to this sample object.")

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
        self._custom_radii = []
        with tqdm(desc="Setting up Extended Sources", total=len(self._accepted_inds), disable=no_prog_bar) as dec_lb:
            for ind in range(len(self._accepted_inds)):
                r, d = ra[self._accepted_inds[ind]], dec[self._accepted_inds[ind]]
                if redshift is None:
                    z = None
                else:
                    z = redshift[self._accepted_inds[ind]]
                n = self._names[ind]
                cr = custom_region_radius[self._accepted_inds[ind]]

                try:
                    self._sources[n] = ExtendedSource(r, d, z, n, cr, use_peak, peak_lo_en, peak_hi_en,
                                                      back_inn_rad_factor, back_out_rad_factor, cosmology, True,
                                                      load_fits, peak_find_method)
                    if isinstance(cr, Quantity):
                        self._custom_radii.append(cr.value)
                        # I know this will write to this over and over, but it seems a bit silly to check
                        #  whether this has been set yet when all radii should be forced to be the same unit
                        self._cr_unit = cr.unit
                    else:
                        self._custom_radii.append(np.NaN)
                        self._cr_unit = Unit('')
                    final_names.append(n)
                except PeakConvergenceFailedError:
                    warn("The peak finding algorithm has not converged for {}, using user "
                         "supplied coordinates".format(n))
                    self._sources[n] = ExtendedSource(r, d, z, n, cr, False, peak_lo_en, peak_hi_en,
                                                      back_inn_rad_factor, back_out_rad_factor, cosmology, True,
                                                      load_fits, peak_find_method)
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
    def custom_radii(self) -> Quantity:
        """
        Property getter for the radii of the custom analysis regions that can be used for analysis of the
        extended sources in this sample. Users are not required to pass a custom analysis region so this
        may be NaN.

        :return: A non-scalar Quantity of the custom source radii passed in by the user.
        :rtype: Quantity
        """
        return Quantity(self._custom_radii, self._cr_unit)

    @property
    def custom_radii_unit(self) -> Unit:
        """
        Property getter for the unit which the custom analysis radii values are stored in.

        :return: The unit that the custom radii are stored in.
        :rtype: Unit
        """
        return self._cr_unit

    def _del_data(self, key: int):
        """
        Specific to the ExtendedSample class, this deletes the extra data stored during the initialisation
        of this type of sample.

        :param int key: The index or name of the source to delete.
        """
        del self._custom_radii[key]


class PointSample(BaseSample):
    """
    The sample class for general point sources, without the extra information required to analyse more specific
    X-ray point sources.

    :param np.ndarray ra: The right-ascensions of the point sources, in degrees.
    :param np.ndarray dec: The declinations of the point sources, in degrees.
    :param np.ndarray redshift: The redshifts of the point sources, optional. Default is None.
    :param np.ndarray name: The names of the point sources, optional. If no names are supplied
        then they will be constructed from the supplied coordinates.
    :param Quantity point_radius: The point source analysis region radius(ii) for this sample. Either
        pass a scalar astropy quantity, or a non-scalar astropy quantity with length equal to the number of sources.
    :param bool use_peak: Whether peak positions should be found and used. For PointSample the 'simple' peak
        finding method is the only one available.
    :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak
        position from. Default is 0.5keV.
    :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak
        position from. Default is 2.0keV.
    :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
        radius for the background region. Default is 1.05.
    :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
        radius for the background region. Default is 1.5.
    :param cosmology: An astropy cosmology object for use throughout analysis of the source.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param bool no_prog_bar: Should a source declaration progress bar be shown during setup.
    :param bool psf_corr: Should images be PSF corrected with default settings during sample setup.
    """
    def __init__(self, ra: np.ndarray, dec: np.ndarray, redshift: np.ndarray = None, name: np.ndarray = None,
                 point_radius: Quantity = Quantity(30, 'arcsec'), use_peak: bool = False,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5,
                 cosmology=Planck15, load_fits: bool = False, no_prog_bar: bool = False, psf_corr: bool = False):
        """
        The init method of the PointSample class.
        """

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
                except PeakConvergenceFailedError:
                    warn("The peak finding algorithm has not converged for {}, using user "
                         "supplied coordinates".format(n))
                    self._sources[n] = PointSource(r, d, z, n, pr, False, peak_lo_en, peak_hi_en,
                                                   back_inn_rad_factor, back_out_rad_factor, cosmology, True,
                                                   load_fits, False)
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








