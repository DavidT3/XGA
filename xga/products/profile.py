#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 23/01/2021, 17:04. Copyright (c) David J Turner
from typing import Tuple, Union

import numpy as np
from astropy.units import Quantity, UnitConversionError
from scipy.integrate import trapz, cumtrapz

from ..products.base import BaseProfile1D
from ..products.phot import RateMap


class SurfaceBrightness1D(BaseProfile1D):
    def __init__(self, rt: RateMap, radii: Quantity, values: Quantity, centre: Quantity, pix_step: int,
                 min_snr: float, outer_rad: Quantity, radii_err: Quantity = None, values_err: Quantity = None,
                 background: Quantity = None, pixel_bins: np.ndarray = None, back_pixel_bin: np.ndarray = None,
                 ann_areas: Quantity = None):
        """
        A subclass of BaseProfile1D, designed to store and analyse surface brightness radial profiles
        of Galaxy Clusters. Allows for the viewing, fitting of the profile.

        :param RateMap rt: The RateMap from which this SB profile was generated.
        :param Quantity radii: The radii at which surface brightness has been measured.
        :param Quantity values: The surface brightnesses that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param int pix_step: The width of each annulus in pixels used to generate this profile.
        :param float min_snr: The minimum signal to noise imposed upon this profile.
        :param Quantity outer_rad: The outer radius of this profile.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        :param Quantity background: The background brightness value.
        :param np.ndarray pixel_bins: An optional argument that provides the pixel bins used to create the profile.
        :param np.ndarray back_pixel_bin: An optional argument that provides the pixel bin used for the background
            calculation of this profile.
        :param Quantity ann_areas: The area of the annuli.
        """
        super().__init__(radii, values, centre, rt.src_name, rt.obs_id, rt.instrument, radii_err, values_err)

        if type(background) != Quantity:
            raise TypeError("The background variables must be an astropy quantity.")

        # Saves the reference to the RateMap this profile was generated from
        self._ratemap = rt

        # Set the internal type attribute to brightness profile
        self._prof_type = "brightness"

        # Setting the energy bounds
        self._energy_bounds = rt.energy_bounds

        # Check that the background passed by the user is the same unit as values
        if background is not None and background.unit == values.unit:
            self._background = background
        elif background is not None and background.unit != values.unit:
            raise UnitConversionError("The background unit must be the same as the values unit.")
        # If no background is passed then the internal background attribute stays at 0 as it was set in
        #  BaseProfile1D

        # Useful quantities from generation of surface brightness profile
        self._pix_step = pix_step
        self._min_snr = min_snr
        self._outer_rad = outer_rad

        # This is an attribute that doesn't matter enough to be passed in, but can be set externally if it is relevant
        #  Describes whether minimum signal to noise re-binning was successful, we assume it is
        # There may be a process that doesn't generate this flag that creates this profile, so that is another reason
        #  it isn't passed in.
        self._succeeded = True

        # Storing the pixel bins used to create this particular profile, if passed, None if not.
        self._pix_bins = pixel_bins
        # Storing the pixel bin for the background region
        self._back_pix_bin = back_pixel_bin

        # Storing the annular areas for this particular profile, if passed, None if not.
        self._areas = ann_areas

    @property
    def pix_step(self) -> int:
        """
        Property that returns the integer pixel step size used to generate the annuli that
        make up this profile.

        :return: The pixel step used to generate the surface brightness profile.
        :rtype: int
        """
        return self._pix_step

    @property
    def min_snr(self) -> float:
        """
        Property that returns minimum signal to noise value that was imposed upon this profile
        during generation.

        :return: The minimum signal to noise value used to generate this profile.
        :rtype: float
        """
        return self._min_snr

    @property
    def outer_radius(self) -> Quantity:
        """
        Property that returns the outer radius used for the generation of this profile.

        :return: The outer radius used in the generation of the profile.
        :rtype: Quantity
        """
        return self._outer_rad

    @property
    def psf_corrected(self) -> bool:
        """
        Tells the user (and XGA), whether the RateMap this brightness profile was generated from has
        been PSF corrected or not.

        :return: Boolean flag, True means this object has been PSF corrected, False means it hasn't
        :rtype: bool
        """
        return self._ratemap.psf_corrected

    @property
    def psf_algorithm(self) -> Union[str, None]:
        """
        If the RateMap this brightness profile was generated from has been PSF corrected, this property gives
        the name of the algorithm used.

        :return: The name of the algorithm used to correct for PSF effects, or None if there was no PSF correction.
        :rtype: Union[str, None]
        """
        return self._ratemap.psf_algorithm

    @property
    def psf_bins(self) -> Union[int, None]:
        """
        If the RateMap this brightness profile was generated from has been PSF corrected, this property
        gives the number of bins that the X and Y axes were divided into to generate the PSFGrid.

        :return: The number of bins in X and Y for which PSFs were generated, or None if the object
        hasn't been PSF corrected.
        :rtype: Union[int, None]
        """
        return self._ratemap.psf_bins

    @property
    def psf_iterations(self) -> Union[int, None]:
        """
        If the RateMap this brightness profile was generated from has been PSF corrected, this property gives
        the number of iterations that the algorithm went through.

        :return: The number of iterations the PSF correction algorithm went through, or None if there has been
        no PSF correction.
        :rtype: Union[int, None]
        """
        return self._ratemap.psf_iterations

    @property
    def psf_model(self) -> Union[str, None]:
        """
        If the RateMap this brightness profile was generated from has been PSF corrected, this property gives the
        name of the PSF model used.

        :return: The name of the PSF model used to correct for PSF effects, or None if there has been no
        PSF correction.
        :rtype: Union[str, None]
        """
        return self._ratemap.psf_model

    @property
    def min_snr_succeeded(self) -> bool:
        """
        If True then the minimum signal to noise re-binning that can be applied to surface brightness profiles by
        some functions was successful, if False then it failed and the profile with no re-binning is stored here.

        :return: A boolean flag describing whether re-binning was successful or not.
        :rtype: bool
        """
        return self._succeeded

    @min_snr_succeeded.setter
    def min_snr_succeeded(self, new_val: bool):
        """
        A setter for the minimum signal to noise re-binning success flag. If True then the minimum signal to noise
        re-binning that can be applied to surface brightness profiles by some functions was successful, if False
        then it failed and the profile with no re-binning is stored here.

        :param bool new_val: The new value of the boolean flag describing whether re-binning was successful or not.
        """
        if not isinstance(new_val, bool):
            raise TypeError("min_snr_succeeded must be a boolean variable.")
        self._succeeded = new_val

    @property
    def pixel_bins(self) -> np.ndarray:
        """
        The annuli radii used to generate this profile, assuming they were passed on initialisation, otherwise None.

        :return: Numpy array containing the pixel bins used to measure this radial brightness profile.
        :rtype: np.ndarray
        """
        return self._pix_bins

    @property
    def back_pixel_bin(self) -> np.ndarray:
        """
        The annulus used to measure the background for this profile, assuming they were passed on
        initialisation, otherwise None.

        :return: Numpy array containing the pixel bin used to measure the background.
        :rtype: np.ndarray
        """
        return self._back_pix_bin

    @property
    def areas(self) -> Quantity:
        """
        Returns the areas of the annuli used to make this profile as an astropy Quantity.

        :return: Astropy non-scalar quantity containing the areas.
        :rtype: Quantity
        """
        return self._areas

    def check_match(self, rt: RateMap, centre: Quantity, pix_step: int, min_snr: float, outer_rad: Quantity) -> bool:
        """
        A method for external use to check whether this profile matches the requested configuration of surface
        brightness profile, put here just because I imagine it'll be used in quite a few places.

        :param RateMap rt: The RateMap to compare to this profile.
        :param Quantity centre: The central coordinate to compare to this profile.
        :param int pix_step: The width of each annulus in pixels to compare to this profile.
        :param float min_snr: The minimum signal to noise to compare to this profile.
        :param Quantity outer_rad: The outer radius to compare to this profile.
        :return: Whether this profile matches the passed parameters or not.
        :rtype: bool
        """
        # Matching the passed RateMap to the internal RateMap is very powerful, as by definition it checks
        #  all of the PSF related attributes. Don't need to directly compare the radii values either because
        #  they are a combination of the other parameters here.
        if rt == self._ratemap and np.all(centre == self._centre) and pix_step == self._pix_step \
                and min_snr == self._min_snr and outer_rad == self._outer_rad:
            match = True
        else:
            match = False
        return match


class GasMass1D(BaseProfile1D):
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None):
        """
        A subclass of BaseProfile1D, designed to store and analyse gas mass radial profiles of Galaxy
        Clusters.

        :param Quantity radii: The radii at which gas mass has been measured.
        :param Quantity values: The gas mass that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err)
        self._prof_type = "gas_mass"

        # As this will more often than not be generated from GasDensity1D, we have to allow an
        #  external realisation to be added
        self._allowed_real_types = ["gas_dens_prof"]


class GasDensity1D(BaseProfile1D):
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None):
        """
        A subclass of BaseProfile1D, designed to store and analyse gas density radial profiles of Galaxy
        Clusters. Allows for the viewing, fitting of the profile, as well as measurement of gas masses,
        and generation of gas mass radial profiles.

        :param Quantity radii: The radii at which gas density has been measured.
        :param Quantity values: The gas densities that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err)

        # Actually imposing limits on what units are allowed for the radii and values for this - just
        #  to make things like the gas mass integration easier and more reliable. Also this is for mass
        #  density, not number density.
        if not radii.unit.is_equivalent("Mpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("solMass / Mpc3"):
            raise UnitConversionError("Values unit cannot be converted to solMass / Mpc3")

        # These are the allowed realisation types (in addition to whatever density models there are
        self._allowed_real_types = ["inv_abel_model", "inv_abel_data"]

        # Setting the type
        self._prof_type = "gas_density"

        # Setting up a dictionary to store gas mass results in.
        self._gas_masses = {}

    def gas_mass(self, real_type: str, outer_rad: Quantity, conf_level: int = 90) -> Tuple[Quantity, Quantity]:
        """
        A method to calculate and return the gas mass (with uncertainties).

        :param str real_type: The realisation type to measure the mass from.
        :param Quantity outer_rad: The radius to measure the gas mass out to.
        :param int conf_level: The confidence level for the gas mass uncertainties.
        :return: A Quantity containing three values (mass, -err, +err), and another Quantity containing
            the entire mass distribution from the whole realisation.
        :rtype: Tuple[Quantity, Quantity]
        """
        if real_type not in self._realisations:
            raise ValueError("{r} is not an acceptable realisation type, this profile object currently has "
                             "realisations stored for".format(r=real_type,
                                                              a=", ".join(list(self._realisations.keys()))))
        if not outer_rad.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("The supplied outer radius cannot be converted to the radius unit"
                                      " of this profile ({u})".format(u=self.radii_unit.to_string()))
        else:
            outer_rad = outer_rad.to(self.radii_unit)

        run_int = True
        # Setting up storage structure if this particular configuration hasn't been run already
        # It goes realisation type - radius - confidence level
        if real_type not in self._gas_masses:
            self._gas_masses[real_type] = {}
            self._gas_masses[real_type][str(outer_rad.value)] = {}
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)] = {"result": None,
                                                                                  "distribution": None}
        elif str(outer_rad.value) not in self._gas_masses[real_type]:
            self._gas_masses[real_type][str(outer_rad.value)] = {}
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)] = {"result": None,
                                                                                  "distribution": None}
        elif str(conf_level) not in self._gas_masses[real_type][str(outer_rad.value)]:
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)] = {"result": None,
                                                                                  "distribution": None}
        else:
            run_int = False

        if real_type not in self._good_model_fits and run_int:
            real_info = self._realisations[real_type]

            allowed_ind = np.where(real_info["mod_radii"] <= outer_rad)[0]
            trunc_rad = real_info["mod_radii"][allowed_ind].to("Mpc")
            trunc_real = real_info["mod_real"].to("solMass / Mpc3")[allowed_ind, :] * trunc_rad[..., None]**2

            gas_masses = Quantity(4*np.pi*trapz(trunc_real.value.T, trunc_rad.value), "solMass")

            upper = 50 + (conf_level / 2)
            lower = 50 - (conf_level / 2)

            gas_mass_mean = np.mean(gas_masses)
            gas_mass_lower = gas_mass_mean - np.percentile(gas_masses, lower)
            gas_mass_upper = np.percentile(gas_masses, upper) - gas_mass_mean
            storage = Quantity(np.array([gas_mass_mean.value, gas_mass_lower.value, gas_mass_upper.value]),
                               gas_mass_mean.unit)
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]["result"] = storage
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]["distribution"] = gas_masses

        elif real_type in self._good_model_fits and run_int:
            raise NotImplementedError("Cannot integrate models yet")

        results: Quantity = self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]['result']
        dist: Quantity = self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]["distribution"]
        return results, dist

    def gas_mass_profile(self, real_type: str, outer_rad: Quantity, conf_level: int = 90) -> GasMass1D:
        """
        A method to calculate and return a gas mass profile.

        :param str real_type: The realisation type to measure the mass profile from.
        :param Quantity outer_rad: The radius to measure the gas mass profile out to.
        :param int conf_level: The confidence level for the gas mass profile uncertainties.
        :return:
        :rtype:
        """
        # Run this for the checks it performs
        mass_res = self.gas_mass(real_type, outer_rad, conf_level)

        real_info = self._realisations[real_type]
        allowed_ind = np.where(real_info["mod_radii"] <= outer_rad)[0]
        trunc_rad = real_info["mod_radii"][allowed_ind].to("Mpc")
        trunc_real = real_info["mod_real"].to("solMass / Mpc3")[allowed_ind, :] * trunc_rad[..., None] ** 2
        gas_mass_real = Quantity(4 * np.pi * cumtrapz(trunc_real.value.T, trunc_rad.value), "solMass").T

        gas_mass_prof = np.mean(gas_mass_real, axis=1)
        # TODO Implement upper and lower bounds when BaseProfile1D supports non-gaussian errors
        gm_prof = GasMass1D(trunc_rad[1:], gas_mass_prof, self.centre, self.src_name, self.obs_id, self.instrument)
        gm_prof.add_realisation("gas_dens_prof", trunc_rad[1:], gas_mass_real, conf_level)

        return gm_prof


class ProjectedGasTemperature1D(BaseProfile1D):
    """
    A profile product meant to hold a radial profile of projected X-ray temperature, as measured from a set
    of annular spectra by XSPEC. These are typically only defined by XGA methods.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a 1D projected temperature profile.

        :param Quantity radii: The radii at which the projected gas temperatures have been measured, this should
            be in a proper radius unit, such as kpc.
        :param Quantity values: The projected gas temperatures that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        """
        #
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err)

        # Actually imposing limits on what units are allowed for the radii and values for this - just
        #  to make things like the gas mass integration easier and more reliable. Also this is for mass
        #  density, not number density.
        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("keV"):
            raise UnitConversionError("Values unit cannot be converted to keV")

        # Setting the type
        self._prof_type = "1d_proj_temperature"


class ProjectedGasMetallicity1D(BaseProfile1D):
    """
    A profile product meant to hold a radial profile of projected X-ray metallicities/abundances, as measured
    from a set of annular spectra by XSPEC. These are typically only defined by XGA methods.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a 1D projected metallicity/abundance profile.

        :param Quantity radii: The radii at which the projected gas metallicity have been measured, this should
            be in a proper radius unit, such as kpc.
        :param Quantity values: The projected gas metallicity that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        """
        #
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err)

        # Actually imposing limits on what units are allowed for the radii and values for this - just
        #  to make things like the gas mass integration easier and more reliable. Also this is for mass
        #  density, not number density.
        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent(""):
            raise UnitConversionError("Values unit cannot be converted to dimensionless")

        # Setting the type
        self._prof_type = "1d_proj_metallicity"









