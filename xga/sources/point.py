#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/04/2023, 23:13. Copyright (c) The Contributors

from typing import Tuple, Dict

import numpy as np
from astropy.cosmology import Cosmology
from astropy.units import Quantity, UnitConversionError

from .general import PointSource
from .. import DEFAULT_COSMO


class Star(PointSource):
    """
    An XGA class for the analysis of X-ray emission from stars within our galaxy. As such it does not accept a
    redshift argument, instead taking an optional distance measure. It will also accept either a proper motion
    magnitude, or a vector of proper motion in RA and Dec directions. Matching to region files also differs from the
    PointSource superclass, with point source regions within match_radius being designated as matches - this is
    because the local nature of stars can throw up problems with the strict matching of RA-Dec within region that
    PointSource uses.

    :param float ra: The right-ascension of the star, in degrees.
    :param float dec: The declination of the star, in degrees.
    :param Quantity distance: A proper distance to the star. Default is None.
    :param Quantity proper_motion: An astropy quantity describing the star's movement across the sky. This may
        have either one (for the magnitude of proper motion) or two (for an RA Dec proper motion vector)
        components. It must be in units that can be converted to arcseconds per year. Default is None.
    :param str name: The name of the star, optional. If no names are supplied then they will be constructed
        from the supplied coordinates.
    :param Quantity point_radius: The point source analysis region radius for this sample. An astropy quantity
        containing the radius should be passed; default is 30 arcsecond radius.
    :param Quantity match_radius: The radius within which point source regions are accepted as a match to the
        RA and Dec passed by the user. The default value is 10 arcseconds.
    :param bool use_peak: Whether peak position should be found and used. For Star the 'simple' peak
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
    :param bool load_products: Whether existing products should be loaded from disk.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param bool regen_merged: Should merged images/exposure maps be regenerated after cleaning. Default is
        True. This option is here so that sample objects can regenerate all merged products at once, which is
        more efficient as it can exploit parallelisation more fully - user probably doesn't need to touch this.
    :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
        to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
        is False. User should only set to True to remove warnings.
    """
    def __init__(self, ra: float, dec: float, distance: Quantity = None, name: str = None,
                 proper_motion: Quantity = None, point_radius: Quantity = Quantity(30, 'arcsec'),
                 match_radius: Quantity = Quantity(10, 'arcsec'), use_peak: bool = False,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5,
                 cosmology: Cosmology = DEFAULT_COSMO, load_products: bool = True, load_fits: bool = False,
                 regen_merged: bool = True, in_sample: bool = False):
        """
        An init of the XGA Star source class.

        :param float ra: The right-ascension of the star, in degrees.
        :param float dec: The declination of the star, in degrees.
        :param Quantity distance: A proper distance to the star. Default is None.
        :param Quantity proper_motion: An astropy quantity describing the star's movement across the sky. This may
            have either one (for the magnitude of proper motion) or two (for an RA Dec proper motion vector)
            components. It must be in units that can be converted to arcseconds per year. Default is None.
        :param str name: The name of the star, optional. If no names are supplied then they will be constructed
            from the supplied coordinates.
        :param Quantity point_radius: The point source analysis region radius for this sample. An astropy quantity
            containing the radius should be passed; default is 30 arcsecond radius.
        :param Quantity match_radius: The radius within which point source regions are accepted as a match to the
            RA and Dec passed by the user. The default value is 10 arcseconds.
        :param bool use_peak: Whether peak position should be found and used. For Star the 'simple' peak
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
        :param bool load_products: Whether existing products should be loaded from disk.
        :param bool load_fits: Whether existing fits should be loaded from disk.
        :param bool regen_merged: Should merged images/exposure maps be regenerated after cleaning. Default is
            True. This option is here so that sample objects can regenerate all merged products at once, which is
            more efficient as it can exploit parallelisation more fully - user probably doesn't need to touch this.
        :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
            to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
            is False. User should only set to True to remove warnings.
        """

        # This is before the super init call so that the changed _source_type_match method has a matching radius
        #  attribute to use
        # Check that the matching radius argument is acceptable, in terms of its units
        if isinstance(match_radius, Quantity) and not match_radius.unit.is_equivalent('arcsec'):
            raise UnitConversionError("The match_radius argument must be in units that can be converted to arcseconds.")
        elif not isinstance(match_radius, Quantity):
            raise TypeError("The match_radius must be an astropy quantity that can be converted to arcseconds.")
        else:
            match_radius = match_radius.to('arcsec')

        # Storing the match radius in an attribute
        self._match_radius = match_radius

        # Run the init of the PointSource superclass
        super().__init__(ra, dec, None, name, point_radius, use_peak, peak_lo_en, peak_hi_en, back_inn_rad_factor,
                         back_out_rad_factor, cosmology, load_products, load_fits, regen_merged, in_sample)

        # Checking that the distance argument (as redshift isn't really valid for objects within our galaxy) is
        #  in a unit that we understand and approve of
        if isinstance(distance, Quantity) and not distance.unit.is_equivalent('pc'):
            raise UnitConversionError("The distance argument cannot be converted to pc.")
        elif not isinstance(distance, Quantity) and distance is not None:
            raise TypeError("The distance argument must be an astropy quantity that can be converted to parsecs, "
                            "or None.")
        elif distance is not None:
            distance = distance.to('pc')

        # Checks the proper motion passed is acceptable
        self._check_proper_motion(proper_motion)
        # Then makes sure to convert it to the expected unit if its not None
        if proper_motion is not None:
            proper_motion = proper_motion.to("arcsec/yr")

        # Storing distance and proper motion in class attributes for use later
        self._distance = distance
        self._proper_motion = proper_motion

    def _source_type_match(self, source_type: str) -> Tuple[Dict, Dict, Dict]:
        """
        A function to override the _source_type_match method of the BaseSource class, containing a slightly more
        complex version of the point source matching criteria that the PointSource class uses. Here point source
        regions are considered a match if any part of them falls within the match_radius passed on instantiation
        of the Star class.

        :param str source_type: Should either be ext or pnt, describes what type of source I
            should be looking for in the region files.
        :return: A dictionary containing the matched region for each ObsID + a combined region, another
            dictionary containing any sources that matched to the coordinates and weren't chosen,
            and a final dictionary with sources that aren't the target, or in the 2nd dictionary.
        :rtype: Tuple[Dict, Dict, Dict]
        """

        # The original _source_type_match is run first, as it is still useful I just wish to refine the matches
        #  a bit afterwards by accepting other regions within a certain radius.
        results_dict, alt_match_dict, anti_results_dict = super()._source_type_match('pnt')

        # Then I run through the anti-results dictionary (no idea why I called it that originally but
        #  I'm running with it), here is where the extra checks occur. This for loop iterates through the
        #  different ObsIDs in the directory.
        for k, v in anti_results_dict.items():
            # We only want to check regions that are point sources, so in this instance we select red ones
            recheck = [r for r in v if r.visual['color'] == 'red']
            # Then we use the handy function I wrote ages ago to find if any of those regions lie within
            #  match_radius of the ra_dec of the source. This will return regions that have ANY PART of
            #  themselves within our search radius.
            within = self.regions_within_radii(Quantity(0, 'arcsec'), self._match_radius, self.ra_dec, recheck)

            # Make a copy of the current ObsID's non matched regions
            reg_copy = np.array(v).copy()

            # Find which of those regions are now considered to be extra matches, but as indexes
            ex_matches = np.argwhere(np.isin(v, within)).flatten()
            # And the inverse information, which should be kept as non-matches
            inv_ex_matches = np.argwhere(~np.isin(v, within)).flatten()

            # As for some reason I stored these as lists of regions, we have to convert back to a list, which is
            #  unfortunate but oh well got to be consistent. Then the regions that are kept as non-matches are made
            #  the new anti-results entry for that ObsID
            anti_results_dict[k] = list(reg_copy[inv_ex_matches])
            # This concatenates the new (if there are any) extra matches to the existing alternative matches list
            alt_match_dict[k] += list(reg_copy[ex_matches])

        return results_dict, alt_match_dict, anti_results_dict

    @property
    def match_radius(self) -> Quantity:
        """
        This tells you the matching radius used during the setup of this Star instance.

        :return: Matching radius defined at instantiation.
        :rtype: Quantity
        """
        return self._match_radius

    @property
    def distance(self) -> Quantity:
        """
        Property returning the distance to the star, as was passed in on creation of this source object.

        :return: The distance to the star.
        :rtype: Quantity
        """
        return self._distance

    @property
    def proper_motion(self) -> Quantity:
        """
        Property returning the proper motion (absolute value or vector) of the star.

        :return: A proper motion magnitude or vector.
        :rtype: Quantity
        """
        return self._proper_motion

    @proper_motion.setter
    def proper_motion(self, new_val: Quantity):
        # Runs the checks on proper motion, if it fails an exception is raised
        self._check_proper_motion(new_val)

        self._proper_motion = new_val

    @staticmethod
    def _check_proper_motion(prop_mot: Quantity):
        """
        Just checks that proper motion is passed in a way that the source will accept and understand.

        :param Quantity prop_mot: The proper motion quantity.
        """
        # Checking that the proper motion argument information is correctly formatted and in an appropriate
        #  unit. I think it tends to be measured in arcseconds / yr, and of course its a vector
        if isinstance(prop_mot, Quantity) and not prop_mot.unit.is_equivalent('arcsec/yr'):
            raise UnitConversionError("Proper motion value cannot be converted to arcsec/yr, please give proper"
                                      "motion in different units.")
        elif isinstance(prop_mot, Quantity) and not prop_mot.isscalar and len(prop_mot) > 2:
            raise ValueError("Proper motion may have one or two components (for absolute value and "
                             "vector respectively), no more.")
