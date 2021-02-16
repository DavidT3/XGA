#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 16/02/2021, 11:31. Copyright (c) David J Turner
from typing import Tuple, Union
from warnings import warn

import numpy as np
from astropy.constants import k_B, G
from astropy.units import Quantity, UnitConversionError, temperature_energy, K
from scipy.integrate import trapz, cumtrapz, quad
from scipy.misc import derivative

from .. import NHC, HY_MASS, ABUND_TABLES
from ..exceptions import ModelNotAssociatedError, XGAInvalidModelError, XGAFitError
from ..models import PROF_TYPE_MODELS
from ..products.base import BaseProfile1D
from ..products.phot import RateMap
from ..sourcetools.deproj import shell_ann_vol_intersect
from ..sourcetools.misc import ang_to_rad


class SurfaceBrightness1D(BaseProfile1D):
    """
    This class provides an interface to radially symmetric X-ray surface brightness profiles of extended objects.
    """
    def __init__(self, rt: RateMap, radii: Quantity, values: Quantity, centre: Quantity, pix_step: int,
                 min_snr: float, outer_rad: Quantity, radii_err: Quantity = None, values_err: Quantity = None,
                 background: Quantity = None, pixel_bins: np.ndarray = None, back_pixel_bin: np.ndarray = None,
                 ann_areas: Quantity = None, deg_radii: Quantity = None):
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
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        super().__init__(radii, values, centre, rt.src_name, rt.obs_id, rt.instrument, radii_err, values_err,
                         deg_radii=deg_radii)

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

        # This is the type of compromise I make when I am utterly exhausted, I am just going to require this be in
        #  degrees
        if not outer_rad.unit.is_equivalent('deg'):
            raise UnitConversionError("outer_rad must be convertible to degrees.")
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

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Surface Brightness"

        en_key = "bound_{l}-{h}_".format(l=rt.energy_bounds[0].to('keV').value, h=rt.energy_bounds[1].to('keV').value)
        if rt.psf_corrected:
            psf_key = "_" + rt.psf_model + "_" + str(rt.psf_bins) + "_" + rt.psf_algorithm + str(rt.psf_iterations)
        else:
            psf_key = ""

        ro = outer_rad.to('deg').value
        self._storage_key = en_key + psf_key + self._storage_key + "_st{ps}_minsn{ms}_ro{ro}".format(ps=int(pix_step),
                                                                                                     ms=min_snr, ro=ro)

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


# TODO WRITE A CUSTOM STORAGE KEY
class GasMass1D(BaseProfile1D):
    """
    This class provides an interface to a cumulative gas mass profile of a Galaxy Cluster.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, deg_radii: Quantity = None):
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
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, deg_radii=deg_radii)
        self._prof_type = "gas_mass"

        # As this will more often than not be generated from GasDensity3D, we have to allow an
        #  external realisation to be added
        self._allowed_real_types = ["gas_dens_prof"]

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Cumulative Gas Mass"


# TODO WRITE A CUSTOM STORAGE KEY
class GasDensity3D(BaseProfile1D):
    """
    This class provides an interface to a gas density profile of a galaxy cluster.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
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
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable. It is
            possible for a Gas Density profile to be generated from spectral or photometric information.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        # Actually imposing limits on what units are allowed for the radii and values for this - just
        #  to make things like the gas mass integration easier and more reliable. Also this is for mass
        #  density, not number density.
        if not radii.unit.is_equivalent("Mpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("solMass / Mpc^3"):
            raise UnitConversionError("Values unit cannot be converted to solMass / Mpc3")

        # Its easier for me if I just make sure they are in this particular unit, especially when it comes to
        #  fitting and integrating models
        values = values.to("solMass / Mpc^3")
        if values_err is not None:
            values_err = values_err.to("solMass / Mpc^3")

        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        # These are the allowed realisation types (in addition to whatever density models there are
        self._allowed_real_types = ["inv_abel_model", "inv_abel_data"]

        # Setting the type
        self._prof_type = "gas_density"

        # Setting up a dictionary to store gas mass results in.
        self._gas_masses = {}

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Gas Density"

    def gas_mass(self, real_type: str, outer_rad: Quantity, conf_level: int = 90, num_real: int = 300) \
            -> Tuple[Quantity, Quantity]:
        """
        A method to calculate and return the gas mass (with uncertainties).

        :param str real_type: The realisation type to measure the mass from.
        :param Quantity outer_rad: The radius to measure the gas mass out to.
        :param int conf_level: The confidence level for the gas mass uncertainties.
        :param int num_real: Number of realisations of model to be generated, if a model fit is being integrated
            for the gas mass value.
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

        if not outer_rad.isscalar:
            raise ValueError("Only a scalar quantity may be passed for outer_rad.")

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

        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)
        if real_type not in self._good_model_fits and run_int:
            real_info = self._realisations[real_type]

            allowed_ind = np.where(real_info["mod_radii"] <= outer_rad)[0]
            trunc_rad = real_info["mod_radii"][allowed_ind].to("Mpc")
            trunc_real = real_info["mod_real"].to("solMass / Mpc3")[allowed_ind, :] * trunc_rad[..., None]**2

            gas_masses = Quantity(4*np.pi*trapz(trunc_real.value.T, trunc_rad.value), "solMass")
            nan_not_present = np.array(list(set(np.where(~np.isnan(gas_masses))[0])))
            gas_masses = gas_masses[nan_not_present]

            gas_mass_mean = np.mean(gas_masses)
            gas_mass_lower = gas_mass_mean - np.percentile(gas_masses, lower)
            gas_mass_upper = np.percentile(gas_masses, upper) - gas_mass_mean

            # TODO SHOULD IT BE THE MEAN VALUE OR THE VALUE WITH THE FINAL FIT PARAMETERS?
            storage = Quantity(np.array([gas_mass_mean.value, gas_mass_lower.value, gas_mass_upper.value]),
                               gas_mass_mean.unit)
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]["result"] = storage
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]["distribution"] = gas_masses

        elif real_type in self._good_model_fits and run_int:
            dens_fit = self._good_model_fits[real_type]
            dens_fit_pars = dens_fit['par']
            # TODO GENERATING REALISATIONS HERE ASSUMES GAUSSIAN ERRORS AGAIN
            dens_one_sig_err = dens_fit['par_err_1sig']

            dens_model_par = np.repeat(dens_fit_pars[..., None], num_real, axis=1).T
            dens_model_par_err = np.repeat(dens_one_sig_err[..., None], num_real, axis=1).T
            dens_par_dists = np.random.normal(dens_model_par, dens_model_par_err)

            vol_rat = (Quantity(1, outer_rad.unit) / Quantity(1, 'Mpc')).to('').value ** 3

            gm_val = 4*np.pi*quad(lambda r: dens_fit['model_func'](r, *tuple(dens_fit_pars))*vol_rat*np.power(r, 2), 0,
                                  outer_rad.value)[0]
            gas_masses = []
            for i in range(0, num_real):
                gd_func = lambda r: dens_fit['model_func'](r, *dens_par_dists[i, :])*vol_rat*np.power(r, 2)
                gm_real = 4*np.pi*quad(gd_func, 0, outer_rad.value)[0]
                gas_masses.append(gm_real)

            # gm_val = Quantity(gm_val, "Msun")
            gas_masses = Quantity(gas_masses, "Msun")

            # Making sure we don't include any gas masses that might be NaN by selecting only realisations where
            #  no NaN is present
            nan_not_present = np.array(list(set(np.where(~np.isnan(gas_masses))[0])))
            gas_masses = gas_masses[nan_not_present]

            gas_mass_mean = np.mean(gas_masses)
            gas_mass_lower = gas_mass_mean - np.percentile(gas_masses, lower)
            gas_mass_upper = np.percentile(gas_masses, upper) - gas_mass_mean

            # TODO SHOULD IT BE THE MEAN VALUE OR THE VALUE WITH THE FINAL FIT PARAMETERS?
            storage = Quantity(np.array([gas_mass_mean.value, gas_mass_lower.value, gas_mass_upper.value]),
                               gas_mass_mean.unit)
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]["result"] = storage
            self._gas_masses[real_type][str(outer_rad.value)][str(conf_level)]["distribution"] = gas_masses

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
                 radii_err: Quantity = None, values_err: Quantity = None, upper_limit: Quantity = Quantity(70, 'keV'),
                 associated_set_id: int = None, set_storage_key: str = None, deg_radii: Quantity = None):
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
        :param Quantity upper_limit: An upper limit on what the temperature values are allowed to be, meant to
            combat XSPEC's habit of putting failed temperature fits to 64keV. It can only be set on init, and
            any points over that value will have their value and uncertainty set to NaN. Default is 70keV, which
            WILL NOT exclude any values generated by XSPEC as the maximum temperature XSPEC measures is 64keV.
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("keV"):
            raise UnitConversionError("Values unit cannot be converted to keV")

        # Making a copy of the original data, just so it can be accessed if desired after the upper limit is applied
        self._og_values = self.values.copy()
        self._og_values_err = self.values_err.copy()

        # This just checks that the upper limit is in a legal unit
        if not upper_limit.unit.is_equivalent(self.values_unit):
            raise UnitConversionError("The upper limit unit {uu} cannot be converted to the temperature unit "
                                      "{tu}".format(uu=upper_limit.unit.to_string(), tu=self.values.unit.to_string()))
        else:
            upper_limit = upper_limit.to(self.values_unit)

        # Applying the upper limit passed by the user
        self._values[self._values > upper_limit] = np.nan
        self._values_err[self._values > upper_limit] = np.nan

        # Putting the upper limit into an attribute
        self._upper_lim = upper_limit

        # Setting the type
        self._prof_type = "1d_proj_temperature"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Projected Temperature"

    @property
    def original_values(self) -> Quantity:
        """
        A way to access the original temperature values of this profile, in case the upper limit
        has removed some points.

        :return: An astropy quantity containing the un-edited temperature profile values.
        :rtype: Quantity
        """
        return self._og_values

    @property
    def original_values_err(self) -> Quantity:
        """
        A way to access the original temperature value errors of this profile, in case the upper limit
        has removed some points.

        :return: An astropy quantity containing the un-edited temperature profile value errors.
        :rtype: Quantity
        """
        return self._og_values_err

    @property
    def upper_limit(self) -> Quantity:
        """
        Property which returns the temperature upper limit passed on init, and which has been used to cut the data.

        :return: An astropy quantity containing the upper limit value.
        :rtype: Quantity
        """
        return self._upper_lim


class APECNormalisation1D(BaseProfile1D):
    """
    A profile product meant to hold a radial profile of XSPEC normalisation, as measured from a set of annular spectra
    by XSPEC. These are typically only defined by XGA methods. This is a useful profile because it allows to not
    only infer 3D profiles of temperature and metallicity, but can also allow us to infer the 3D density profile.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a 1D XSPEC normalisation profile.

        :param Quantity radii: The radii at which the XSPEC normalisations have been measured, this should
            be in a proper radius unit, such as kpc.
        :param Quantity values: The XSPEC normalisations that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("cm^-5"):
            raise UnitConversionError("Values unit cannot be converted to keV")

        # Setting the type
        self._prof_type = "1d_apec_norm"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "APEC Normalisation"

    def _gen_profile_setup(self, redshift: float, cosmo: Quantity, abund_table: str = 'angr') \
            -> Tuple[Quantity, Quantity, float]:
        """
        There are many common steps in the gas_density_profile and emission_measure_profile methods, so I decided to
        put some of the common setup steps in this internal function

        :param float redshift: The redshift of the source that this profile was generated from.
        :param cosmo: The chosen cosmology.
        :param str abund_table: The abundance table to used for the conversion from n_e x n_H to n_e^2 during density
            calculation. Default is the famous Anders & Grevesse table.
        :return:
        :rtype: Tuple[Quantity, Quantity, float]
        """
        # We need radii errors so that BaseProfile init can calculate the annular radii. The only possible time
        #  this would be triggered is if a user defines their own normalisation profile.
        if self.radii_err is None:
            raise ValueError("There are no radii uncertainties available for this APEC normalisation profile, they"
                             " are required to generate a profile.")

        # This just checks that the input abundance table is legal
        if abund_table in NHC and abund_table in ABUND_TABLES:
            hy_to_elec = NHC[abund_table]
        elif abund_table in ABUND_TABLES and abund_table not in NHC:
            avail_nhc = ", ".join(list(NHC.keys()))
            raise ValueError(
                "{a} is a valid choice of XSPEC abundance table, but XGA doesn't have an electron to hydrogen "
                "ratio for that table yet, this is the developers fault so please remind him if you see this "
                "error. Please select from one of these in the meantime; {av}".format(a=abund_table, av=avail_nhc))
        elif abund_table not in ABUND_TABLES:
            avail_abund = ", ".join(ABUND_TABLES)
            raise ValueError("{a} is not a valid abundance table choice, please use one of the "
                             "following; {av}".format(a=abund_table, av=avail_abund))

        # Converts the radii to cm so that the volume intersections are in the right units.
        if self.annulus_bounds.unit.is_equivalent('kpc'):
            cur_rads = self.annulus_bounds.to('cm')
        elif self.annulus_bounds.unit.is_equivalent('deg'):
            cur_rads = ang_to_rad(self.annulus_bounds.to('deg'), redshift, cosmo).to('cm')
        else:
            raise UnitConversionError("Somehow you have an unrecognised distance unit for the radii of this profile")

        # Calculate the angular diameter distance to the source (in cm), just need the redshift and the cosmology
        #  which has chosen for analysis
        ang_dist = cosmo.angular_diameter_distance(redshift).to("cm")

        return cur_rads, ang_dist, hy_to_elec

    def gas_density_profile(self, redshift: float, cosmo: Quantity, abund_table: str = 'angr', num_real: int = 100,
                            sigma: int = 2) -> GasDensity3D:
        """
        A method to calculate the gas density profile from the APEC normalisation profile, which in turn was
        measured from XSPEC fits of an AnnularSpectra.

        :param float redshift: The redshift of the source that this profile was generated from.
        :param cosmo: The chosen cosmology.
        :param str abund_table: The abundance table to used for the conversion from n_e x n_H to n_e^2 during density
            calculation. Default is the famous Anders & Grevesse table.
        :param int num_real: The number of data realisations which should be generated to infer density errors.
        :param int sigma: What sigma of error should the density profile be created with, the default is 2σ.
        :return: The gas density profile which has been calculated from the APEC normalisation profile.
        :rtype: GasDensity3D
        """
        # There are commonalities between this method and others in this class, so I shifted some steps into an
        #  internal method which we will call now
        cur_rads, ang_dist, hy_to_elec = self._gen_profile_setup(redshift, cosmo, abund_table)

        # This uses a handy function I defined a while back to calculate the volume intersections between the annuli
        #  and spherical shells
        vol_intersects = shell_ann_vol_intersect(cur_rads, cur_rads)

        # This is essentially the constants bit of the XSPEC APEC normalisation
        # Angular diameter distance is calculated using the cosmology which was associated with the cluster
        #  at definition
        conv_factor = (4 * np.pi * (ang_dist * (1 + redshift)) ** 2) / (hy_to_elec * 10 ** -14)
        gas_dens = np.sqrt(np.linalg.inv(vol_intersects.T) @ self.values * conv_factor) * HY_MASS

        norm_real = self.generate_data_realisations(num_real)
        gas_dens_reals = Quantity(np.zeros(norm_real.shape), gas_dens.unit)
        # Using a loop here is ugly and relatively slow, but it should be okay
        for i in range(0, num_real):
            gas_dens_reals[i, :] = np.sqrt(np.linalg.inv(vol_intersects.T) @ norm_real[i, :] * conv_factor) * HY_MASS

        # Convert the profile and the realisations to the correct unit
        gas_dens = gas_dens.to("Msun/Mpc^3")
        gas_dens_reals = gas_dens_reals.to("Msun/Mpc^3")

        # Calculates the standard deviation of each data point, this is how we estimate the density errors
        dens_sigma = np.std(gas_dens_reals, axis=0)*sigma

        # Set up the actual profile object and return it
        dens_prof = GasDensity3D(self.radii, gas_dens, self.centre, self.src_name, self.obs_id, self.instrument,
                                 self.radii_err, dens_sigma, self.set_ident, self.associated_set_storage_key,
                                 self.deg_radii)
        return dens_prof

    def emission_measure_profile(self, redshift: float, cosmo: Quantity, abund_table: str = 'angr',
                                 num_real: int = 100, sigma: int = 2):
        """
        A method to calculate the emission measure profile from the APEC normalisation profile, which in turn was
        measured from XSPEC fits of an AnnularSpectra.

        :param float redshift: The redshift of the source that this profile was generated from.
        :param cosmo: The chosen cosmology.
        :param str abund_table: The abundance table to used for the conversion from n_e x n_H to n_e^2 during density
            calculation. Default is the famous Anders & Grevesse table.
        :param int num_real: The number of data realisations which should be generated to infer emission measure errors.
        :param int sigma: What sigma of error should the density profile be created with, the default is 2σ.
        :return:
        :rtype:
        """
        cur_rads, ang_dist, hy_to_elec = self._gen_profile_setup(redshift, cosmo, abund_table)

        # This is essentially the constants bit of the XSPEC APEC normalisation
        # Angular diameter distance is calculated using the cosmology which was associated with the cluster
        #  at definition
        conv_factor = (4 * np.pi * (ang_dist * (1 + redshift)) ** 2) / (hy_to_elec * 10 ** -14)
        em_meas = self.values * conv_factor

        norm_real = self.generate_data_realisations(num_real)
        em_meas_reals = norm_real * conv_factor

        # Calculates the standard deviation of each data point, this is how we estimate the density errors
        em_meas_sigma = np.std(em_meas_reals, axis=0)*sigma

        # Set up the actual profile object and return it
        em_meas_prof = EmissionMeasure1D(self.radii, em_meas, self.centre, self.src_name, self.obs_id, self.instrument,
                                         self.radii_err, em_meas_sigma, self.set_ident, self.associated_set_storage_key,
                                         self.deg_radii)
        return em_meas_prof


class EmissionMeasure1D(BaseProfile1D):
    """
    A profile product meant to hold a radial profile of X-ray emission measure.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a radial emission measure profile.

        :param Quantity radii: The radii at which the emission measures have been measured, this should
            be in a proper radius unit, such as kpc.
        :param Quantity values: The emission measures that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        #
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)
        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("cm^-3"):
            raise UnitConversionError("Values unit cannot be converted to cm^-3")

        # Setting the type
        self._prof_type = "1d_emission_measure"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Emission Measure"


class ProjectedGasMetallicity1D(BaseProfile1D):
    """
    A profile product meant to hold a radial profile of projected X-ray metallicities/abundances, as measured
    from a set of annular spectra by XSPEC. These are typically only defined by XGA methods.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
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
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        #
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        # Actually imposing limits on what units are allowed for the radii and values for this - just
        #  to make things like the gas mass integration easier and more reliable. Also this is for mass
        #  density, not number density.
        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent(""):
            raise UnitConversionError("Values unit cannot be converted to dimensionless")

        # Setting the type
        self._prof_type = "1d_proj_metallicity"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Projected Metallicity"


class GasTemperature3D(BaseProfile1D):
    """
    A profile product meant to hold a 3D radial profile of X-ray temperature, as measured by some form of
    de-projection applied to a projected temperature profile
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None,  associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a radial 3D temperature profile.

        :param Quantity radii: The radii at which the gas temperatures have been measured, this should
            be in a proper radius unit, such as kpc.
        :param Quantity values: The gas temperatures that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("keV"):
            raise UnitConversionError("Values unit cannot be converted to keV")

        # Setting the type
        self._prof_type = "gas_temperature"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "3D Temperature"


class BaryonFraction(BaseProfile1D):
    """
    A profile product which will hold a profile showing how the baryon fraction of a galaxy cluster changes
    with radius. These profiles are typically generated from a HydrostaticMass profile product instance.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None,  associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a radial baryon fraction profile.

        :param Quantity radii: The radii at which the baryon fracion have been measured, this should
            be in a proper radius unit, such as kpc.
        :param Quantity values: The baryon fracions that have been measured.
        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent(""):
            raise UnitConversionError("Values unit cannot be converted to dimensionless")

        # Setting the type
        self._prof_type = "baryon_fraction"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Baryon Fraction"


# TODO WRITE CUSTOM STORAGE KEY HERE AS WELL
class HydrostaticMass(BaseProfile1D):
    """
    A profile product which uses input GasTemperature3D and GasDensity3D profiles to generate a hydrostatic
    mass profile, which in turn can be used to measure the hydrostatic mass at a particular radius. In contrast
    to other profile objects, this one calculates the y values itself.
    """
    def __init__(self, temperature_profile: GasTemperature3D, temperature_model: str, density_profile: GasDensity3D,
                 density_model: str, radii: Quantity, radii_err: Quantity, deg_radii: Quantity):

        # We check whether the temperature profile passed is actually the type of profile we need
        if type(temperature_profile) != GasTemperature3D:
            raise TypeError("Only a GasTemperature3D instance may be passed for temperature_profile, check "
                            "you haven't accidentally passed a ProjectedGasTemperature1D.")
        # Now we check the model choice passed for the temperature profile
        try:
            self._temp_fit = temperature_profile.get_model_fit(temperature_model)
        except XGAInvalidModelError:
            allowed = ", ".format(list(PROF_TYPE_MODELS['gas_temperature'].keys()))
            raise ValueError("{p} is not an allowed temperature_model value, please use one of the "
                             "following: {a}".format(a=allowed, p=temperature_model))
        except ModelNotAssociatedError:
            raise ValueError("{p} is an allowed temperature_model value, but hasn't yet been fitted to the "
                             "profile".format(p=temperature_model))
        except XGAFitError:
            raise ValueError("{p} is an allowed temperature_model value, but the fit you performed was not "
                             "successful".format(p=temperature_model))

        # We repeat this process with the density profile and model
        if type(density_profile) != GasDensity3D:
            raise TypeError("Only a GasDensity3D instance may be passed for density_profile, check you haven't "
                            "accidentally passed a GasDensity3D.")

        try:
            self._dens_fit = density_profile.get_model_fit(density_model)
        except XGAInvalidModelError:
            allowed = ", ".format(list(PROF_TYPE_MODELS['gas_density'].keys()))
            raise ValueError("{p} is not an allowed density_model value, please use one of the "
                             "following: {a}".format(a=allowed, p=density_model))
        except ModelNotAssociatedError:
            raise ValueError("{p} is an allowed density_model value, but hasn't yet been fitted to the "
                             "profile".format(p=density_model))
        except XGAFitError:
            raise ValueError("{p} is an allowed density_model value, but the fit you performed was not "
                             "successful".format(p=density_model))

        # We also need to check that someone hasn't done something dumb like pass profiles from two different
        #  clusters, so we'll compare source names.
        if temperature_profile.src_name != density_profile.src_name:
            raise ValueError("You have passed temperature and density profiles from two different "
                             "sources, any resulting hydrostatic mass measurements would not be valid, so this is not "
                             "allowed.")
        # And check they were generated with the same central coordinate, otherwise they may not be valid. I
        #  considered only raising a warning, but I need a consistent central coordinate to pass to the super init
        elif np.any(temperature_profile.centre != density_profile.centre):
            raise ValueError("The temperature and density profiles do not have the same central coordinate.")
        # Same reasoning with the ObsID and instrument
        elif temperature_profile.obs_id != density_profile.obs_id:
            raise ValueError("The temperature and density profiles do not have the same associated ObsID.")
        elif temperature_profile.instrument != density_profile.instrument:
            raise ValueError("The temperature and density profiles do not have the same associated instrument.")

        # We see if either of the profiles have an associated spectrum
        if temperature_profile.set_ident is None and density_profile.set_ident is None:
            set_id = None
            set_store = None
        elif temperature_profile.set_ident is None and density_profile.set_ident is not None:
            set_id = density_profile.set_ident
            set_store = density_profile.associated_set_storage_key
        elif temperature_profile.set_ident is not None and density_profile.set_ident is None:
            set_id = temperature_profile.set_ident
            set_store = temperature_profile.associated_set_storage_key
        elif temperature_profile.set_ident is not None and density_profile.set_ident is not None:
            if temperature_profile.set_ident != density_profile.set_ident:
                warn("The temperature and density profile you passed where generated from different sets of annular"
                     " spectra, the mass profiles associated set ident will be set to None.")
                set_id = None
                set_store = None
            else:
                set_id = temperature_profile.set_ident
                set_store = temperature_profile.associated_set_storage_key

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")
        else:
            radii = radii.to('kpc')
            radii_err = radii_err.to('kpc')

        # We won't REQUIRE that the profiles have data point generated at the same radii, as we're gonna
        #  measure masses from the models, but I do need to check that the passed radii are within the radii of the
        #  and warn the user if they aren't
        if (radii > temperature_profile.annulus_bounds[-1]).any() \
                or (radii[-1] > density_profile.annulus_bounds[-1]).any():
            warn("Some radii passed to the HydrostaticMass init are outside the data range covered by the temperature "
                 "or density profiles, as such you will be extrapolating based on the model fits.")

        self._temp_prof = temperature_profile
        self._dens_prof = density_profile
        self._temp_model = temperature_model
        self._dens_model = density_model

        mass, mass_dist = self.mass(radii, conf_level=68)
        mass_vals = mass[0, :]
        mass_errs = np.mean(mass[1:, :], axis=0)

        super().__init__(radii, mass_vals, self._temp_prof.centre, self._temp_prof.src_name, self._temp_prof.obs_id,
                         self._temp_prof.instrument, radii_err, mass_errs, set_id, set_store, deg_radii)

        # Setting the type
        self._prof_type = "hydrostatic_mass"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = r"M$_{\rm{hydro}}$"

        # Setting up a dictionary to store hydro mass results in.
        self._masses = {}

        # This dictionary is for measurements of the baryon fraction
        self._baryon_fraction = {}

    def add_realisation(self):
        """
        This profile does not support adding realisations from an external source.
        """
        pass

    def mass(self, radius: Quantity, conf_level: int = 90, num_real: int = 300) -> Union[Quantity, Quantity]:
        """
        A method which will measure a hydrostatic mass and hydrostatic mass uncertainty within the given
        radius/radii. No corrections are applied to the values calculated by this method, it is just the vanilla
        hydrostatic mass.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            mass within.
        :param int conf_level: The confidence level for the mass uncertainties.
        :param int num_real: The number of model realisations which should be generated for error propagation.
        :return: An astropy quantity containing the mass/masses, lower and upper uncertainties, and another containing
            the mass realisation distribution.
        :rtype: Union[Quantity, Quantity]
        """
        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        if (radius.max() > self._temp_prof.annulus_bounds[-1]).any() \
                or (radius.max() > self._dens_prof.annulus_bounds[-1]).any():
            warn("The radius at which you have requested the mass is greater than the outermost radius of the "
                 "temperature or density profile used to generate this mass profile, prediction may not be valid.")

        if radius.isscalar and str(radius.value) in self._masses \
                and str(conf_level) in self._masses[str(radius.value)]:
            already_run = True
            mass_res = self._masses[str(radius.value)][str(conf_level)]["result"]
            real_masses = self._masses[str(radius.value)][str(conf_level)]["distribution"]
        else:
            already_run = False

        if not already_run:
            # Reading out the fit parameters of the chosen temperature model, just for convenience
            temp_fit_pars = self._temp_fit['par']
            # TODO GENERATING REALISATIONS HERE ASSUMES GAUSSIAN ERRORS AGAIN
            temp_one_sig_err = self._temp_fit['par_err_1sig']

            temp_model_par = np.repeat(temp_fit_pars[..., None], num_real, axis=1).T
            temp_model_par_err = np.repeat(temp_one_sig_err[..., None], num_real, axis=1).T

            # This generates model_real random samples from the passed model parameters, assuming they are Gaussian
            temp_par_dists = np.random.normal(temp_model_par, temp_model_par_err)

            # Setting up the units for the derivative of the temperature profile
            der_temp_unit = self._temp_prof.values_unit / radius.unit

            # Actually calculating the derivative of the temperature profile
            # TODO DO THE DERIVATIVES OF THE MODELS SO THIS COULD BE DONE ANALYTICALLY WHERE POSSIBLE
            der_temp = Quantity(derivative(lambda r: self._temp_fit['model_func'](r, *temp_fit_pars), radius.value),
                                der_temp_unit)
            # The realisation derivatives
            der_real_temps = Quantity(derivative(lambda r: self._temp_fit['model_func'](r, *temp_par_dists.T),
                                                 radius.value[..., None]), der_temp_unit).T
            temp = Quantity(self._temp_fit['model_func'](radius.value, *temp_fit_pars), self._temp_prof.values_unit)
            # Realisation temperatures
            real_temps = Quantity(self._temp_fit['model_func'](radius.value[..., None], *temp_par_dists.T),
                                  self._temp_prof.values_unit).T

            # As of the time of writing, its not currently possible to get this far with a temperature profile
            #  in Kelvin, only keV, but I may allow it later and I would like to be ready for that possibility
            if not temp.unit == K:
                # Convert the temperature to Kelvin using the temperature energy equivalency, the value that
                #  goes into the hydrostatic mass equation must be in Kelvin
                temp = temp.to('K', equivalencies=temperature_energy())
                real_temps = real_temps.to('K', equivalencies=temperature_energy())
                # I can't use the equivalency when there is another unit in there for some reason, so I just do it
                #  manually by dividing by the Boltzmann constant
                der_temp = (der_temp / k_B).to(K / radius.unit)
                der_real_temps = (der_real_temps / k_B).to(K / radius.unit)

            # Now setting up the unit for the density profile derivative
            der_dens_unit = self._dens_prof.values_unit / radius.unit

            # This process is all essentially the same as the temperature derivatives
            dens_fit_pars = self._dens_fit['par']
            # TODO GENERATING REALISATIONS HERE ASSUMES GAUSSIAN ERRORS AGAIN
            dens_one_sig_err = self._dens_fit['par_err_1sig']

            dens_model_par = np.repeat(dens_fit_pars[..., None], num_real, axis=1).T
            dens_model_par_err = np.repeat(dens_one_sig_err[..., None], num_real, axis=1).T
            dens_par_dists = np.random.normal(dens_model_par, dens_model_par_err)

            der_dens = Quantity(derivative(lambda r: self._dens_fit['model_func'](r, *dens_fit_pars), radius.value),
                                der_dens_unit)
            der_real_dens = Quantity(derivative(lambda r: self._dens_fit['model_func'](r, *dens_par_dists.T),
                                                radius.value[..., None]), der_dens_unit).T
            dens = Quantity(self._dens_fit['model_func'](radius.value, *dens_fit_pars), self._dens_prof.values_unit)
            real_dens = Quantity(self._dens_fit['model_func'](radius.value[..., None], *dens_par_dists.T),
                                 self._dens_prof.values_unit).T

            # Please note that this is just the vanilla hydrostatic mass equation, but not written in the standard form.
            # Here there are no logs in the derivatives, and I've also written it in such a way that mass densities are
            #  used rather than number densities
            mass = ((-1 * k_B * np.power(radius, 2)) / (dens * HY_MASS * G)) * ((dens * der_temp) + (temp * der_dens))
            # Just converts the mass/masses to the unit we normally use for them
            mass = mass.to('Msun')

            real_masses = ((-1 * k_B * np.power(radius, 2)) / (real_dens * HY_MASS * G)) * \
                          ((real_dens * der_real_temps) + (real_temps * der_real_dens))
            real_masses = real_masses.to('Msun')

            # Making sure we don't include any profiles with NaN in by selecting only realisations where
            #  no NaN is present
            nan_not_present = np.array(list(set(np.where(~np.isnan(real_masses))[0])))
            if real_masses.ndim == 2:
                real_masses = real_masses[nan_not_present, :]
            elif real_masses.ndim == 1:
                real_masses = real_masses[nan_not_present]
            else:
                raise ValueError("You have a 3D mass realisation array and I have no idea how...")

            mass_mean = np.mean(real_masses, axis=0)
            mass_lower = mass_mean - np.percentile(real_masses, lower, axis=0)
            mass_upper = np.percentile(real_masses, upper, axis=0) - mass_mean

            # mass_err = np.nanstd(real_masses, axis=0) * sigma

            mass_res = Quantity(np.array([mass_mean.value, mass_lower.value, mass_upper.value]), mass.unit)

            if mass_res.ndim == 1 and str(radius.value) not in self._masses:
                self._masses[str(radius.value)] = {str(conf_level): {"result": mass_res, "distribution": real_masses}}
            elif mass_res.ndim == 1 and str(radius.value) in self._masses and \
                    str(conf_level) not in self._masses[str(radius.value)]:
                self._masses[str(radius.value)][str(conf_level)] = {"result": mass_res, "distribution": real_masses}

        return mass_res, real_masses

    def baryon_fraction(self, radius: Quantity, conf_level: int = 68, num_real: int = 300) -> Tuple[Quantity, Quantity]:
        """
        A method to use the hydrostatic mass information of this profile, and the gas density information of the
        input gas density profile, to calculate a baryon fraction within the given radius.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            baryon fraction within.
        :param int conf_level: The confidence level for the uncertainties.
        :param int num_real: The number of model realisations which should be generated for error propagation.
        :return: An astropy quantity containing the baryon fraction, -ve error, and +ve error, and another quantity
            containing the baryon fraction distribution.
        :rtype: Tuple[Quantity, Quantity]
        """
        if not radius.isscalar:
            raise ValueError("Unfortunately this method can only calculate the baryon fraction within one "
                             "radius, multiple radii are not supported.")

        if str(radius.value) in self._baryon_fraction and str(conf_level) in self._baryon_fraction[str(radius.value)]:
            already_run = True
            bar_frac_res = self._baryon_fraction[str(radius.value)][str(conf_level)]["result"]
            bar_frac_dist = self._baryon_fraction[str(radius.value)][str(conf_level)]["distribution"]
        else:
            already_run = False

        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        if not already_run:
            hy_mass, hy_mass_dist = self.mass(radius, conf_level, num_real)

            gas_mass, gas_mass_dist = self._dens_prof.gas_mass(self._dens_model, radius, conf_level)
            if len(hy_mass_dist) < len(gas_mass_dist):
                bar_frac_dist = gas_mass_dist[:len(hy_mass_dist)] / hy_mass_dist
            elif len(hy_mass_dist) > len(gas_mass_dist):
                bar_frac_dist = gas_mass_dist / hy_mass_dist[:len(gas_mass_dist)]
            else:
                bar_frac_dist = gas_mass_dist / hy_mass_dist

            bar_frac_mean = np.mean(bar_frac_dist, axis=0)
            bar_frac_lower = bar_frac_mean - np.percentile(bar_frac_dist, lower, axis=0)
            bar_frac_upper = np.percentile(bar_frac_dist, upper, axis=0) - bar_frac_mean
            # TODO Reconcile this with issue #403, should I be returning the gm/hym, or the mean of the dist
            bar_frac_res = Quantity([(gas_mass[0]/hy_mass[0]).value, bar_frac_lower.value, bar_frac_upper.value], '')

            if str(radius.value) not in self._baryon_fraction:
                self._baryon_fraction[str(radius.value)] = {str(conf_level): {"result": bar_frac_res,
                                                                              "distribution": bar_frac_dist}}
            elif str(radius.value) in self._baryon_fraction and str(conf_level) \
                    not in self._baryon_fraction[str(radius.value)]:
                self._baryon_fraction[str(radius.value)][str(conf_level)] = {"result": bar_frac_res,
                                                                             "distribution": bar_frac_dist}

        return bar_frac_res, bar_frac_dist

    def baryon_fraction_profile(self, conf_level: int = 68, num_real: int = 300) -> BaryonFraction:
        """
        A method which uses the baryon_fraction method to construct a baryon fraction profile at the radii of
        this HydrostaticMass profile.

        :param int conf_level: The confidence level for the uncertainties.
        :param int num_real: The number of model realisations which should be generated for error propagation.
        :return: An XGA BaryonFraction object.
        :rtype: BaryonFraction
        """
        frac = []
        frac_err = []
        # Step through the radii of this profile
        for rad in self.radii:
            # Grabs the baryon fraction for the current radius
            b_frac = self.baryon_fraction(rad, conf_level, num_real)[0]

            # Only need the actual result, not the distribution
            frac.append(b_frac[0])
            # Calculates a mean uncertainty
            frac_err.append(b_frac[1:].mean())

        # Makes them unit-less quantities, as baryon fraction is mass/mass
        frac = Quantity(frac, '')
        frac_err = Quantity(frac_err, '')

        return BaryonFraction(self.radii, frac, self.centre, self.src_name, self.obs_id, self.instrument,
                              self.radii_err, frac_err, self.set_ident, self.associated_set_storage_key, self.deg_radii)


class Generic1D(BaseProfile1D):
    """
    A 1D profile product meant to hold profiles which have been dynamically generated by XSPEC profile fitting
    of models that I didn't build into XGA. It can also be used to make arbitrary profiles using external data.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 y_axis_label: str, prof_type: str, radii_err: Quantity = None, values_err: Quantity = None,
                 associated_set_id: int = None, set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of this subclass of BaseProfile1D, used by a dynamic XSPEC fitting process, or directly by a user,
        to set up an XGA profile with custom data.

        :param Quantity centre: The central coordinate the profile was generated from.
        :param str source_name: The name of the source this profile is associated with.
        :param str obs_id: The observation which this profile was generated from.
        :param str inst: The instrument which this profile was generated from.
        :param str y_axis_label: The label to apply to the y-axis of any plots generated from this profile.
        :param str prof_type: This is a string description of the profile, used to store it in an XGA source (with
            _profile appended). For instance the prof_type of a ProjectedGasTemperature1D instance is
            1d_proj_temperature, and it would be stored under 1d_proj_temperature_profile.
        :param Quantity radii_err: Uncertainties on the radii.
        :param Quantity values_err: Uncertainties on the values.
        :param int associated_set_id: The set ID of the AnnularSpectra that generated this - if applicable.
        :param str set_storage_key: Must be present if associated_set_id is, this is the storage key which the
            associated AnnularSpectra generates to place itself in XGA's store structure.
        :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
            units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
            values converted to degrees, and allows this object to construct a predictable storage key.
        """

        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)
        self._prof_type = prof_type
        self._y_axis_name = y_axis_label








