#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 01/09/2020, 16:11. Copyright (c) David J Turner

import warnings
from typing import Tuple, List, Union

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitBase, deg, UnitConversionError
from numpy import ndarray

from .general import PointSource


class Star(PointSource):
    """
    An XGA class for the analysis of X-ray emission from stars within the milky way.

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
                 use_peak: bool = False, peak_lo_en: Quantity = Quantity(0.5, "keV"),
                 peak_hi_en: Quantity = Quantity(2.0, "keV"), back_inn_rad_factor: float = 1.05,
                 back_out_rad_factor: float = 1.5, cosmology=Planck15, load_products: bool = True,
                 load_fits: bool = False, regen_merged: bool = True):

        # Run the init of the PointSource superclass
        super().__init__(ra, dec, None, name, point_radius, use_peak, peak_lo_en, peak_hi_en, back_inn_rad_factor,
                         back_out_rad_factor, cosmology, load_products, load_fits, regen_merged)

        # Checking that the distance argument (as redshift isn't really valid for objects within our galaxy) is
        #  in a unit that we understand and approve of
        if isinstance(distance, Quantity) and not distance.unit.is_equivalent('pc'):
            raise UnitConversionError("The distance argument cannot be converted to pc.")

        # Checks the proper motion passed is acceptable
        self._check_proper_motion(proper_motion)

        # Storing distance and proper motion in class attributes for use later
        self._distance = distance
        self._proper_motion = proper_motion

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
