#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/09/2020, 16:11. Copyright (c) David J Turner

import warnings
from typing import Tuple, List, Union, Dict

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitBase, deg, UnitConversionError
from numpy import ndarray

from .general import PointSource


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
    :param cosmology: An astropy cosmology object for use throughout analysis of the source.
    :param bool load_products: Whether existing products should be loaded from disk.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param bool regen_merged: Should merged images/exposure maps be regenerated after cleaning. Default is
        True. This option is here so that sample objects can regenerate all merged products at once, which is
        more efficient as it can exploit parallelisation more fully - user probably doesn't need to touch this.
    """
    def __init__(self, ra: float, dec: float, distance: Quantity = None, name: str = None,
                 proper_motion: Quantity = None, point_radius: Quantity = Quantity(30, 'arcsec'),
                 match_radius: Quantity = Quantity(10, 'arcsec'), use_peak: bool = False,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5, cosmology=Planck15,
                 load_products: bool = True, load_fits: bool = False, regen_merged: bool = True):

        # Run the init of the PointSource superclass
        super().__init__(ra, dec, None, name, point_radius, use_peak, peak_lo_en, peak_hi_en, back_inn_rad_factor,
                         back_out_rad_factor, cosmology, load_products, load_fits, regen_merged)

        # Checking that the distance argument (as redshift isn't really valid for objects within our galaxy) is
        #  in a unit that we understand and approve of
        if isinstance(distance, Quantity) and not distance.unit.is_equivalent('pc'):
            raise UnitConversionError("The distance argument cannot be converted to pc.")
        elif not isinstance(distance, Quantity) and distance is not None:
            raise TypeError("The distance argument must be an astropy quantity that can be converted to parsecs, "
                            "or None.")
        elif distance is not None:
            distance = distance.to('pc')

        # Check that the matching radius argument is acceptable, in terms of its units
        if isinstance(match_radius, Quantity) and not match_radius.unit.is_equivalent('arcsec'):
            raise UnitConversionError("The match_radius argument must be in units that can be converted to arcseconds.")
        elif not isinstance(match_radius, Quantity):
            raise TypeError("The match_radius must be an astropy quantity that can be converted to arcseconds.")
        else:
            match_radius = match_radius.to('arcsec')

        # Checks the proper motion passed is acceptable
        self._check_proper_motion(proper_motion)
        # Then makes sure to convert it to the expected unit if its not None
        if proper_motion is not None:
            proper_motion = proper_motion.to("arcsec/yr")

        # Storing the match radius in an attribute
        self._match_radius = match_radius

        # Storing distance and proper motion in class attributes for use later
        self._distance = distance
        self._proper_motion = proper_motion

    def _source_type_match(self, source_type: str) -> Tuple[Dict, Dict, Dict]:
        """
        A function to override the _source_type_match method of the BaseSource class, containing slightly
        more complex matching criteria for galaxy clusters. Galaxy clusters having their own version of this
        method was driven by issue #407, the problems I was having with low redshift clusters particularly.



        :param str source_type: Should either be ext or pnt, describes what type of source I
            should be looking for in the region files.
        :return: A dictionary containing the matched region for each ObsID + a combined region, another
            dictionary containing any sources that matched to the coordinates and weren't chosen,
            and a final dictionary with sources that aren't the target, or in the 2nd dictionary.
        :rtype: Tuple[Dict, Dict, Dict]
        """
        def dist_from_source(reg):
            """
            Calculates the euclidean distance between the centre of a supplied region, and the
            position of the source.

            :param reg: A region object.
            :return: Distance between region centre and source position.
            """
            ra = reg.center.ra.value
            dec = reg.center.dec.value
            return Quantity(np.sqrt(abs(ra - self._ra_dec[0]) ** 2 + abs(dec - self._ra_dec[1]) ** 2), 'deg')

        raise NotImplementedError("Not written this for Star yet")
        results_dict, alt_match_dict, anti_results_dict = super()._source_type_match('ext')

        # The 0.66 and 2.25 factors are intended to shift the r200 and r2500 values to approximately r500, and were
        #  decided on by dividing the Arnaud et al. 2005 R-T relations by one another and finding the mean factor
        if self._radii['r500'] is not None:
            check_rad = self.convert_radius(self._radii['r500'] * 0.15, 'deg')
        elif self._radii['r200'] is not None:
            check_rad = self.convert_radius(self._radii['r200'] * 0.66 * 0.15, 'deg')
        else:
            check_rad = self.convert_radius(self._radii['r2500'] * 2.25 * 0.15, 'deg')

        # Here we scrub the anti-results dictionary (I don't know why I called it that...) to make sure cool cores
        #  aren't accidentally removed, and that chunks of cluster emission aren't removed
        new_anti_results = {}
        for obs in self._obs:
            # This is where the cleaned interlopers will be stored
            new_anti_results[obs] = []
            # Cycling through the current interloper regions for the current ObsID
            for reg_obj in anti_results_dict[obs]:
                # Calculating the distance (in degrees) of the centre of the current interloper region from
                #  the user supplied coordinates of the cluster
                dist = dist_from_source(reg_obj)

                # If the current interloper source is a point source/a PSF sized extended source and is within the
                #  fraction of the chosen characteristic radius of the cluster then we assume it is a poorly handled
                #  cool core and allow it to stay in the analysis
                if reg_obj.visual["color"] == 'red' and dist < check_rad:
                    # We do print a warning though
                    warnings.warn("A point source has been detected in {o} and is very close to the user supplied "
                                  "coordinates of {s}. It will not be excluded from analysis due to the possibility "
                                  "of a mis-identified cool core".format(s=self.name, o=obs))
                elif reg_obj.visual["color"] == "magenta" and dist < check_rad:
                    warnings.warn("A PSF sized extended source has been detected in {o} and is very close to the "
                                  "user supplied coordinates of {s}. It will not be excluded from analysis due "
                                  "to the possibility of a mis-identified cool core".format(s=self.name, o=obs))
                else:
                    new_anti_results[obs].append(reg_obj)

            # Here we run through the 'chosen' region for each observation (so the region that we think is the
            #  cluster) and check if any of the current set of interloper regions intersects with it. If they do
            #  then they are allowed to stay in the analysis under assumption that they're actually part of the
            #  cluster
            for res_obs in results_dict:
                if results_dict[res_obs] is not None:
                    # Reads out the chosen region for res_obs
                    src_reg_obj = results_dict[res_obs]
                    # Stores its central coordinates in an astropy quantity
                    centre = Quantity([src_reg_obj.center.ra.value, src_reg_obj.center.dec.value], 'deg')

                    # At first I set the checking radius to the semimajor axis
                    rad = Quantity(src_reg_obj.width.to('deg').value/2, 'deg')
                    # And use my handy method to find which regions intersect with a circle with the semimajor length
                    #  as radius, centred on the centre of the current chosen region
                    within_width = self.regions_within_radii(Quantity(0, 'deg'), rad, centre, new_anti_results[obs])
                    # Make sure to only select extended (green) sources
                    within_width = [reg for reg in within_width if reg.visual['color'] == 'green']

                    # Then I repeat that process with the semiminor axis, and if a interloper intersects with both
                    #  then it would intersect with the ellipse of the current chosen region.
                    rad = Quantity(src_reg_obj.height.to('deg').value/2, 'deg')
                    within_height = self.regions_within_radii(Quantity(0, 'deg'), rad, centre, new_anti_results[obs])
                    within_height = [reg for reg in within_height if reg.visual['color'] == 'green']

                    # This finds which regions are present in both lists and makes sure if they are in both
                    #  then they are NOT removed from the analysis
                    intersect_regions = list(set(within_width) & set(within_height))
                    for inter_reg in intersect_regions:
                        inter_reg_ind = new_anti_results[obs].index(inter_reg)
                        new_anti_results[obs].pop(inter_reg_ind)

        return results_dict, alt_match_dict, new_anti_results

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
