#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/04/2023, 23:13. Copyright (c) The Contributors
from warnings import warn

import numpy as np
from astropy.cosmology import Cosmology
from astropy.units import Quantity, Unit, UnitConversionError
from tqdm import tqdm

from .base import BaseSample
from .. import DEFAULT_COSMO
from ..exceptions import NoValidObservationsError
from ..sources.point import Star


class StarSample(BaseSample):
    """
    An XGA class for the analysis of a large sample of local X-ray emitting stars.
    Takes information on stars to enable analysis.

    :param np.ndarray ra: The right-ascensions of the stars, in degrees.
    :param np.ndarray dec: The declinations of the stars, in degrees.
    :param np.ndarray distance: The distances to the stars in units convertible to parsecs, optional. Default is None.
    :param np.ndarray name: The names of the stars, optional. If no names are supplied
        then they will be constructed from the supplied coordinates.
    :param Quantity proper_motion: The proper motion of the stars, optional. This should be passed as a non-scalar
        astropy quantity; if magnitudes are passed it should be in the form Quantity([4, 5, 2, 7,...], 'arcsec/yr'),
        and if vectors are passed it should be in the form Quantity([[4, 2.3], [5, 4.3], [2, 2.5],
        [7.2, 6.9],...], 'arcsec/yr'), where the first is in RA and the second in Dec. Units should be convertible to
        arcseconds per year. Default is None
    :param Quantity point_radius: The point source analysis region radius(ii) for this sample. Either
        pass a scalar astropy quantity, or a non-scalar astropy quantity with length equal to the number of sources.
    :param Quantity match_radius: The radius within which point source regions are accepted as a match to the
        RAs and Dec passed by the user. The default value is 10 arcseconds.
    :param bool use_peak: Whether peak positions should be found and used. For StarSample the 'simple' peak
        finding method is the only one available.
    :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak
        position from. Default is 0.5keV.
    :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak
        position from. Default is 2.0keV.
    :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
        radius for the background region. Default is 1.05.
    :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
        radius for the background region. Default is 1.5.
    :param Cosmology cosmology: An astropy cosmology object for use throughout analysis of the source.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param bool no_prog_bar: Should a source declaration progress bar be shown during setup.
    :param bool psf_corr: Should images be PSF corrected with default settings during sample setup.
    """
    def __init__(self, ra: np.ndarray, dec: np.ndarray, distance: np.ndarray = None, name: np.ndarray = None,
                 proper_motion: Quantity = None, point_radius: Quantity = Quantity(30, 'arcsec'),
                 match_radius: Quantity = Quantity(10, 'arcsec'), use_peak: bool = False,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5,
                 cosmology: Cosmology = DEFAULT_COSMO, load_fits: bool = False, no_prog_bar: bool = False,
                 psf_corr: bool = False):
        """
         The init of the StarSample XGA class.
        """
        # Strongly enforce that its a quantity, this also means that it should be guaranteed that all radii have
        #  a single unit
        if not isinstance(point_radius, Quantity):
            raise TypeError("Please pass a quantity object for point_radius, rather than an array or list.")
        # People might pass a single value for point_radius, in which case we turn it into a non-scalar quantity
        elif point_radius.isscalar:
            point_radius = Quantity([point_radius.value] * len(ra), point_radius.unit)
        elif not point_radius.isscalar and len(point_radius) != len(ra):
            raise ValueError("If you pass a set of radii (rather than a single radius) to point_radius then there"
                             " must be one entry per object passed to this sample object.")

        # This does the checking for distance, making sure that if its a Quantity it fulfills our requirements
        if isinstance(distance, Quantity) and distance.isscalar:
            raise ValueError("Please pass a non-scalar quantity for distance.")
        elif isinstance(distance, Quantity) and len(distance) != len(ra):
            raise ValueError("Please pass a non-scalar quantity for distance, with one entry per source.")
        elif isinstance(distance, Quantity) and not distance.unit.is_equivalent('parsec'):
            raise UnitConversionError("Please pass distances that can be converted to parsecs.")

        # Here I also perform checks on the proper motion information passed (if any)
        if isinstance(proper_motion, Quantity) and proper_motion.isscalar:
            raise ValueError("Please pass a non-scalar quantity for proper_motion.")
        elif isinstance(proper_motion, Quantity) and proper_motion.ndim == 1 and len(proper_motion) != len(ra):
            raise ValueError("Please pass a non-scalar quantity for proper motion, and if passing proper"
                             " motion magnitudes please have one entry per source.")
        elif isinstance(proper_motion, Quantity) and proper_motion.ndim == 2 and proper_motion.shape != (len(ra), 2):
            raise ValueError("Please pass a non-scalar quantity for proper motion, and if passing proper"
                             " motion magnitudes please have one entry of two components per source.")

        # I don't like having this here, but it does avoid a circular import problem
        from xga.sas import evselect_image, eexpmap, emosaic

        # Using the super defines BaseSources and stores them in the self._sources dictionary
        super().__init__(ra, dec, None, name, cosmology, load_products=True, load_fits=False, no_prog_bar=no_prog_bar)
        evselect_image(self, peak_lo_en, peak_hi_en)
        eexpmap(self, peak_lo_en, peak_hi_en)
        emosaic(self, "image", peak_lo_en, peak_hi_en)
        emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        # Remove the BaseSources
        del self._sources
        self._sources = {}

        # A list to store the inds of any declarations that failed due to NoValidObservations, so some
        #  attributes can be cleaned up later
        final_names = []
        self._point_radii = []
        self._distances = []
        self._proper_motions = []
        # This records which sources had a failed peak finding attempt, for a warning at the end of the declaration
        failed_peak_find = []
        with tqdm(desc="Setting up Stars", total=len(self._accepted_inds), disable=no_prog_bar) as dec_lb:
            for ind in range(len(self._accepted_inds)):
                r, d = ra[self._accepted_inds[ind]], dec[self._accepted_inds[ind]]
                if distance is None:
                    di = None
                else:
                    di = distance[self._accepted_inds[ind]]

                if proper_motion is None:
                    pm = None
                else:
                    pm = proper_motion[self._accepted_inds[ind]]

                n = self._names[ind]
                pr = point_radius[self._accepted_inds[ind]]

                # Observation cleaning goes on automatically in PointSource, so if a NoValidObservations error is
                #  thrown I have to catch it and not add that source to this sample.
                try:
                    self._sources[n] = Star(r, d, di, n, pm, pr, match_radius, use_peak, peak_lo_en, peak_hi_en,
                                            back_inn_rad_factor, back_out_rad_factor, cosmology, True, load_fits,
                                            False, True)
                    self._point_radii.append(pr.value)
                    self._distances.append(di)
                    self._proper_motions.append(pm)
                    # I know this will write to this over and over, but it seems a bit silly to check whether this has
                    #  been set yet when all radii should be forced to be the same unit
                    self._pr_unit = pr.unit
                    final_names.append(n)
                except NoValidObservationsError:
                    self._failed_sources[n] = "CleanedNoMatch"

                dec_lb.update(1)
        self._names = final_names
        # Store the matching radius as an attribute, though its also in the sources
        self._match_radius = match_radius

        # I've cleaned the observations, and its possible some of the data has been thrown away,
        #  so I should regenerate the mosaic images/expmaps
        emosaic(self, "image", peak_lo_en, peak_hi_en)
        emosaic(self, "expmap", peak_lo_en, peak_hi_en)

        # I don't offer the user choices as to the configuration for PSF correction at the moment
        if psf_corr:
            # Trying to see if this stops a circular import issue I've been having
            from ..imagetools.psf import rl_psf
            rl_psf(self, lo_en=peak_lo_en, hi_en=peak_hi_en)

        # It is possible (especially if someone is using the Sample classes as a way to check whether things have
        #  XMM data) that no sources will have been declared by this point, in which case it should fail now
        if len(self._sources) == 0:
            raise NoValidObservationsError(
                "No Stars have been declared, none of the sample passed the cleaning steps.")

        # Put all the warnings for there being no XMM data in one - I think it's neater. Wait until after the check
        #  to make sure that are some sources because in that case this warning is redundant.
        no_data = [name for name in self._failed_sources if self._failed_sources[name] == 'NoMatch' or
                   self._failed_sources[name] == 'Failed ObsClean']
        # If there are names in that list, then we do the warning
        if len(no_data) != 0:
            warn("The following do not appear to have any XMM data, and will not be included in the "
                 "sample (can also check .failed_names); {n}".format(n=', '.join(no_data)), stacklevel=2)

        # We also do a combined warning for those clusters that had a failed peak finding attempt, if there are any
        if len(failed_peak_find) != 0:
            warn("Peak finding did not converge for the following; {n}, using user "
                 "supplied coordinates".format(n=', '.join(failed_peak_find)), stacklevel=2)

        # This shows a warning that tells the user how to see any suppressed warnings that occurred during source
        #  declarations, but only if there actually were any.
        self._check_source_warnings()

    @property
    def point_radii(self) -> Quantity:
        """
        Property getter for the radii of the regions used for analysis of the stars in this sample.

        :return: A non-scalar Quantity of the point source radii used for analysis of the stars in
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

    @property
    def match_radius(self) -> Quantity:
        """
        This tells you the matching radius used during the setup of this StarSample instance.

        :return: Matching radius defined at instantiation.
        :rtype: Quantity
        """
        return self._match_radius

    @property
    def distances(self) -> Quantity:
        """
        Property returning the distance to the star, as was passed in on creation of this source object.

        :return: The distance to the star.
        :rtype: Quantity
        """
        return Quantity(self._distances)

    @property
    def proper_motions(self) -> Quantity:
        """
        Property returning the proper motion (absolute value or vector) of the star.

        :return: A proper motion magnitude or vector.
        :rtype: Quantity
        """
        return Quantity(self._proper_motions)

    def _del_data(self, key: int):
        """
        Specific to the StarSample class, this deletes the extra data stored during the initialisation
        of this type of sample.

        :param int key: The index or name of the source to delete.
        """
        del self._point_radii[key]
        del self._distances[key]
        del self._proper_motions[key]
