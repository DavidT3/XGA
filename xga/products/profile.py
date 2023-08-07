#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 15/06/2023, 14:28. Copyright (c) The Contributors

from copy import copy
from typing import Tuple, Union, List
from warnings import warn

import numpy as np
from astropy.constants import k_B, G, m_p
from astropy.cosmology import Cosmology
from astropy.units import Quantity, UnitConversionError, Unit
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .. import NHC, ABUND_TABLES, MEAN_MOL_WEIGHT
from ..exceptions import ModelNotAssociatedError, XGAInvalidModelError, XGAFitError
from ..models import PROF_TYPE_MODELS, BaseModel1D
from ..products.base import BaseProfile1D
from ..products.phot import RateMap
from ..sourcetools.deproj import shell_ann_vol_intersect
from ..sourcetools.misc import ang_to_rad


class SurfaceBrightness1D(BaseProfile1D):
    """
    This class provides an interface to radially symmetric X-ray surface brightness profiles of extended objects.

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
    :param bool min_snr_succeeded: A boolean flag describing whether re-binning was successful or not.
    """
    def __init__(self, rt: RateMap, radii: Quantity, values: Quantity, centre: Quantity, pix_step: int,
                 min_snr: float, outer_rad: Quantity, radii_err: Quantity = None, values_err: Quantity = None,
                 background: Quantity = None, pixel_bins: np.ndarray = None, back_pixel_bin: np.ndarray = None,
                 ann_areas: Quantity = None, deg_radii: Quantity = None, min_snr_succeeded: bool = True):
        """
        A subclass of BaseProfile1D, designed to store and analyse surface brightness radial profiles
        of Galaxy Clusters. Allows for the viewing, fitting of the profile.
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

        # Describes whether minimum signal to noise re-binning was successful, we assume it is
        self._succeeded = min_snr_succeeded

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
            psf_key = rt.psf_model + "_" + str(rt.psf_bins) + "_" + rt.psf_algorithm + str(rt.psf_iterations) + "_"
        else:
            psf_key = "_"

        ro = outer_rad.to('deg').value
        self._storage_key = en_key + psf_key + "st{ps}_minsn{ms}_ro{ro}_".format(ps=int(pix_step), ms=min_snr, ro=ro) \
                            + self._storage_key

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

        # This method means that a change has happened to the model, so it should be re-saved
        self.save()

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
    """
    This class provides an interface to a cumulative gas mass profile of a Galaxy Cluster.

    :param Quantity radii: The radii at which gas mass has been measured.
    :param Quantity values: The gas mass that have been measured.
    :param Quantity centre: The central coordinate the profile was generated from.
    :param str source_name: The name of the source this profile is associated with.
    :param str obs_id: The observation which this profile was generated from.
    :param str inst: The instrument which this profile was generated from.
    :param str dens_method: A keyword describing the method used to generate the density profile that was
        used to measure this gas mass profile.
    :param SurfaceBrightness1D/APECNormalisation1D associated_prof: The profile that the gas density profile
        was measured from.
    :param Quantity radii_err: Uncertainties on the radii.
    :param Quantity values_err: Uncertainties on the values.
    :param Quantity deg_radii: A slightly unfortunate variable that is required only if radii is not in
        units of degrees, or if no set_storage_key is passed. It should be a quantity containing the radii
        values converted to degrees, and allows this object to construct a predictable storage key.
    """
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 dens_method: str, associated_prof, radii_err: Quantity = None, values_err: Quantity = None,
                 deg_radii: Quantity = None):
        """
        A subclass of BaseProfile1D, designed to store and analyse gas mass radial profiles of Galaxy
        Clusters.
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, deg_radii=deg_radii)
        self._prof_type = "gas_mass"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Cumulative Gas Mass"

        # The profile from which the densities here were inferred
        self._gen_prof = associated_prof

        if isinstance(associated_prof, SurfaceBrightness1D):
            br_key = copy(self._gen_prof.storage_key)
            en_key = "bound_{l}-{u}_".format(l=associated_prof.energy_bounds[0].value,
                                             u=associated_prof.energy_bounds[1].value)
            extra_info = "_" + br_key.split(en_key)[-1].split("_ra")[0] + "_"
        else:
            extra_info = "_"

        # The density class has an extra bit of information in the storage key, the method used to generate it
        self._storage_key = "me" + dens_method + extra_info + self._storage_key

        self._gen_method = dens_method

    @property
    def density_method(self) -> str:
        """
        Gives the user the method used to generate the density profile used to make this gas mass profile.

        :return: The string describing the method
        :rtype: str
        """
        return self._gen_method

    @property
    def generation_profile(self) -> BaseProfile1D:
        """
        Provides the profile from which the density profile used to make this gas mass profile was measured. Either
        a surface brightness profile if measured using SB methods, or an APEC normalisation profile if inferred
        from annular spectra.

        :return: The profile from which the density profile that made this profile was measured.
        :rtype: Union[SurfaceBrightness1D, APECNormalisation1D]
        """
        return self._gen_prof


class GasDensity3D(BaseProfile1D):
    """
    This class provides an interface to a gas density profile of a galaxy cluster.

    :param Quantity radii: The radii at which gas density has been measured.
    :param Quantity values: The gas densities that have been measured.
    :param Quantity centre: The central coordinate the profile was generated from.
    :param str source_name: The name of the source this profile is associated with.
    :param str obs_id: The observation which this profile was generated from.
    :param str inst: The instrument which this profile was generated from.
    :param str dens_method: A keyword describing the method used to generate this density profile.
    :param SurfaceBrightness1D/APECNormalisation1D associated_prof: The profile that this gas density profile
        was measured from.
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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 dens_method: str, associated_prof, radii_err: Quantity = None, values_err: Quantity = None,
                 associated_set_id: int = None, set_storage_key: str = None, deg_radii: Quantity = None):
        """
        A subclass of BaseProfile1D, designed to store and analyse gas density radial profiles of Galaxy
        Clusters. Allows for the viewing, fitting of the profile, as well as measurement of gas masses,
        and generation of gas mass radial profiles. Values of density should either be in a unit of mass/volume,
        or a particle number density unit of 1/cm^3.
        """
        # Actually imposing limits on what units are allowed for the radii and values for this - just
        #  to make things like the gas mass integration easier and more reliable. Also this is for mass
        #  density, not number density.
        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")
        else:
            radii = radii.to('kpc')

        # Densities are allowed to be either a mass or number density
        if not values.unit.is_equivalent("solMass / Mpc^3") and not values.unit.is_equivalent("1/cm^3"):
            raise UnitConversionError("Values unit cannot be converted to either solMass / Mpc3 or 1/cm^3")
        elif values.unit.is_equivalent("solMass / Mpc^3"):
            values = values.to('solMass / Mpc^3')
            # As two different types of gas density are allowed I need to store which one we're dealing with
            self._sub_type = "mass_dens"
            chosen_unit = Unit('solMass / Mpc^3')
        elif values.unit.is_equivalent("1/cm^3"):
            values = values.to('1/cm^3')
            self._sub_type = "num_dens"
            chosen_unit = Unit("1/cm^3")

        if values_err is not None:
            values_err = values_err.to(chosen_unit)

        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        # Setting the type
        self._prof_type = "gas_density"

        # Setting up a dictionary to store gas mass results in.
        self._gas_masses = {}

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Gas Density"

        # Stores the density generation method
        self._gen_method = dens_method

        # The profile from which the densities here were inferred
        self._gen_prof = associated_prof

        if isinstance(associated_prof, SurfaceBrightness1D):
            br_key = copy(self._gen_prof.storage_key)
            en_key = "bound_{l}-{u}_".format(l=associated_prof.energy_bounds[0].value,
                                             u=associated_prof.energy_bounds[1].value)
            extra_info = "_" + br_key.split(en_key)[-1].split("_ra")[0] + "_"
        else:
            extra_info = "_"

        # The density class has an extra bit of information in the storage key, the method used to generate it
        self._storage_key = "me" + dens_method + extra_info + self._storage_key

    def gas_mass(self, model: str, outer_rad: Quantity, inner_rad: Quantity = None, conf_level: float = 68.2,
                 fit_method: str = 'mcmc', radius_err: Quantity = None) -> Tuple[Quantity, Quantity]:
        """
        A method to calculate and return the gas mass (with uncertainties). This method uses the model to generate
        a gas mass distribution (using the fit parameter distributions from the fit performed using the model), then
        measures the median mass, along with lower and upper uncertainties.

        Passing uncertainties on the outer (and inner) radii for the gas mass calculation is supported, with such
        uncertainties assumed to be representing a Gaussian distribution. Radii distributions will be drawn from a
        Gaussian, though any radii that are negative will be set to zero, so it could be a truncated Gaussian.

        :param str model: The name of the model from which to derive the gas mass.
        :param Quantity outer_rad: The radius to measure the gas mass out to. Only one radius may be passed at a time.
        :param Quantity inner_rad: The inner radius within which to measure the gas mass, this enables measuring
            core-excised gas masses. Default is None, which equates to zero. If passing separate uncertainties for
            inner and outer radii using `radius_err', the inner radius error must be the second entry.
        :param float conf_level: The confidence level to use to calculate the mass errors
        :param str fit_method: The method that was used to fit the model, default is 'mcmc'.
        :param Quantity radius_err: A standard deviation on radius, which will be taken into account during the
            calculation of gas mass. If both an inner and outer radius have been passed, then you may pass either
            a single standard deviation value for both, or a Quantity with two entries. THE FIRST being the outer
            radius error, THE SECOND being inner radius error.
        :return: A Quantity containing three values (mass, -err, +err), and another Quantity containing
            the entire mass distribution from the whole realisation.
        :rtype: Tuple[Quantity, Quantity]
        """
        # First of all we have to find the model that has been fit to this gas density profile.
        if model not in PROF_TYPE_MODELS[self._prof_type]:
            raise XGAInvalidModelError("{m} is not a valid model for a gas density profile".format(m=model))
        elif model not in self.good_model_fits:
            raise ModelNotAssociatedError("{m} is valid model type, but no fit has been performed".format(m=model))
        else:
            model_obj = self.get_model_fit(model, fit_method)

        if not model_obj.success:
            raise ValueError("The fit to that model was not considered a success by the fit method, cannot proceed.")

        if not outer_rad.isscalar:
            raise ValueError("Gas masses can only be calculated within one radii at a time, please pass a scalar "
                             "value for outer_rad.")
        elif inner_rad is not None and not inner_rad.isscalar:
            raise ValueError("Gas masses can only be calculated within one radii at a time, please pass a scalar "
                             "value for inner_rad.")

        # This checks to see if inner radius is None (probably how it will be used most of the time), and if
        #  it is then creates a Quantity with the same units as outer_radius
        if inner_rad is None:
            inner_rad = Quantity(0, outer_rad.unit)
        elif inner_rad is not None and not inner_rad.unit.is_equivalent(outer_rad):
            raise UnitConversionError("If an inner_radius Quantity is supplied, then it must be in the same units"
                                      " as the outer_radius Quantity.")

        # Checking the input radius units
        if not outer_rad.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("The supplied outer radius cannot be converted to the radius unit"
                                      " of this profile ({u})".format(u=self.radii_unit.to_string()))
        else:
            # This is for consistency, to make sure the same units as the profile radii are used for calculation
            #  and for storage keys
            outer_rad = outer_rad.to(self.radii_unit)
            inner_rad = inner_rad.to(self.radii_unit)

        # When only an outer radius has been passed (i.e. inner radius is zero), then we can only allow one
        #  radius error to be passed
        if radius_err is not None and inner_rad == 0 and not radius_err.isscalar:
            raise ValueError('You may only pass a two-element radius error quantity if you have also set inner_radius '
                             'to a non-zero value.')
        # We know that there is no circumstance where more than two radius errors should be passed
        elif radius_err is not None and not radius_err.isscalar and len(radius_err) > 2:
            raise ValueError("The 'radius_error' argument may have a maximum of two entries, a single value for both"
                             "outer and inner radii, or separate entries for outer and inner radii.")
        # Now we check to see whether the radius error unit is compatible with the radius units we're already
        #  working with
        elif radius_err is not None and not radius_err.unit.is_equivalent(outer_rad.unit):
            raise UnitConversionError("The radius_err quantity must be in units that are equivalent to units "
                                      "of {}.".format(outer_rad.unit.to_string()))

        # Now we make absolutely sure that the radius error(s) are in the correct units
        if radius_err is not None:
            radius_err = radius_err.to(self.radii_unit)

        # Doing an extra check to warn the user if the radius they supplied is outside the radii
        #  covered by the data
        if outer_rad >= self.radii[-1]:
            warn("The outer radius you supplied is greater than or equal to the outer radius covered by the data, so"
                 " you are effectively extrapolating using the model.", stacklevel=2)

        # The next step is setting up radius distributions, if the radius error is not None. The outer_rad
        #  and inner_rad (if applicable) variables will be overwritten with a distribution, which will be picked up
        #  on by the volume integral part of the model function.
        rng = np.random.default_rng()
        if radius_err is None:
            # This is the simplest case, where there is no error at all - here the storage keys are just string
            #  versions of the inner and outer radii
            out_stor_key = str(outer_rad)
            inn_stor_key = str(inner_rad)
        elif radius_err is not None and inner_rad == 0:
            # The keys are defined first because 'outer_rad' is about to be turned into a radius distribution rather
            #  than a single value and we need the original values for string representations. Here the outer radius
            #  is uncertain and the size of the standard deviation becomes part of the storage key
            out_stor_key = str(outer_rad.value) + '_' + str(radius_err.value) + " " + str(outer_rad.unit)
            inn_stor_key = str(inner_rad)
            # The length of one of the parameter distributions in the model is used to tell us how many samples to
            #  draw from our radius distribution, as we need it to be the same length for the volume integral.
            outer_rad = Quantity(rng.normal(outer_rad.value, radius_err.value, len(model_obj.par_dists[0])),
                                 radius_err.unit)
        elif radius_err is not None and radius_err.isscalar:
            # The keys are defined first because the radii variables are about to be turned into radius
            #  distributions rather than single values and we need the original values for string representations.
            #  Here the radii are uncertain (with the same st dev) and the size of the standard deviation becomes
            #  part of the storage key
            out_stor_key = str(outer_rad.value) + '_' + str(radius_err.value) + " " + str(outer_rad.unit)
            inn_stor_key = str(inner_rad.value) + '_' + str(radius_err.value) + " " + str(outer_rad.unit)
            outer_rad = Quantity(rng.normal(outer_rad.value, radius_err.value, len(model_obj.par_dists[0])),
                                 radius_err.unit)
            inner_rad = Quantity(rng.normal(inner_rad.value, radius_err.value, len(model_obj.par_dists[0])),
                                 radius_err.unit)
        elif radius_err is not None and len(radius_err) == 2:
            # The keys are defined first because the radii variables are about to be turned into radius
            #  distributions rather than single values and we need the original values for string representations.
            #  Here the radii are uncertain (with different st devs) and the size of the standard deviations become
            #  part of the storage keys
            out_stor_key = str(outer_rad.value) + '_' + str(radius_err[0].value) + " " + str(outer_rad.unit)
            inn_stor_key = str(inner_rad.value) + '_' + str(radius_err[1].value) + " " + str(outer_rad.unit)
            outer_rad = Quantity(rng.normal(outer_rad.value, radius_err.value[0], len(model_obj.par_dists[0])),
                                 radius_err.unit)
            inner_rad = Quantity(rng.normal(inner_rad.value, radius_err.value[1], len(model_obj.par_dists[0])),
                                 radius_err.unit)
        else:
            raise ValueError("Somehow you have passed a radius error with more than two entries and "
                             "it hasn't been caught - contact the developer.")

        # If we're using a radius distribution(s), then this part checks to ensure that none of the values are
        #  negative because that doesn't make any sense! In such cases the offending radii are set to zero, so really
        #  the radii could be a truncated Gaussian distribution.
        if not outer_rad.isscalar:
            outer_rad[outer_rad < 0] = 0
        if not inner_rad.isscalar:
            inner_rad[inner_rad < 0] = 0

        # Just preparing the way, setting up the storage dictionary - top level identifies the model
        if str(model_obj) not in self._gas_masses:
            self._gas_masses[str(model_obj)] = {}
        # The next layer is the outer radius key, then finally the result will be stored using the inner radius key
        if out_stor_key not in self._gas_masses[str(model_obj)]:
            self._gas_masses[str(model_obj)][out_stor_key] = {}

        # This runs the volume integral on the density profile, using the built-in integral method in the model.
        if inn_stor_key not in self._gas_masses[str(model_obj)][out_stor_key] and \
                out_stor_key != str(Quantity(0, outer_rad.unit)):
            mass_dist = model_obj.volume_integral(outer_rad, inner_rad, use_par_dist=True)
            # Converts to an actual mass rather than a total number of particles
            if self._sub_type == 'num_dens':
                mass_dist *= (MEAN_MOL_WEIGHT*m_p)
            # Converts to solar masses and stores inside the current profile for future reference
            mass_dist = mass_dist.to('Msun')
            self._gas_masses[str(model_obj)][out_stor_key][inn_stor_key] = mass_dist

        # Obviously the mass contained within a zero radius bin is zero, but the integral can fall over sometimes when
        #  this is requested so I put in this special case
        elif inn_stor_key not in self._gas_masses[str(model_obj)][out_stor_key] and \
                (outer_rad.isscalar and outer_rad == 0):
            mass_dist = Quantity(np.zeros(len(model_obj.par_dists[0])), 'Msun')
            self._gas_masses[str(model_obj)][out_stor_key][inn_stor_key] = mass_dist

        else:
            mass_dist = self._gas_masses[str(model_obj)][out_stor_key][inn_stor_key]

        med_mass = np.percentile(mass_dist, 50).value
        upp_mass = np.percentile(mass_dist, 50 + (conf_level/2)).value
        low_mass = np.percentile(mass_dist, 50 - (conf_level/2)).value
        gas_mass = Quantity([med_mass, med_mass-low_mass, upp_mass-med_mass], mass_dist.unit)

        if np.any(gas_mass[0] < 0):
            raise ValueError("A gas mass of less than zero has been measured, which is not physical.")

        # This method means that a change has happened to the model, so it should be re-saved
        self.save()
        return gas_mass, mass_dist

    @property
    def density_method(self) -> str:
        """
        Gives the user the method used to generate this density profile.

        :return: The string describing the method
        :rtype: str
        """
        return self._gen_method

    @property
    def generation_profile(self) -> BaseProfile1D:
        """
        Provides the profile from which this density profile was measured. Either a surface brightness profile
        if measured using SB methods, or an APEC normalisation profile if inferred from annular spectra.

        :return: The profile from which the densities were measured.
        :rtype: Union[SurfaceBrightness1D, APECNormalisation1D]
        """
        return self._gen_prof

    def view_gas_mass_dist(self, model: str, outer_rad: Quantity, conf_level: float = 68.2, figsize=(8, 8),
                           bins: Union[str, int] = 'auto', colour: str = "lightslategrey", fit_method: str = 'mcmc'):
        """
        A method which will generate a histogram of the gas mass distribution that resulted from the gas mass
        calculation at the supplied radius. If the mass for the passed radius has already been measured it, and the
        mass distribution, will be retrieved from the storage of this product rather than re-calculated.

        :param str model: The name of the model from which to derive the gas mass.
        :param Quantity outer_rad: The radius within which to calculate the gas mass.
        :param float conf_level: The confidence level for the mass uncertainties, this doesn't affect the
            distribution, only the vertical lines indicating the measured value of gas mass.
        :param str colour: The desired colour of the histogram.
        :param tuple figsize: The desired size of the histogram figure.
        :param int/str bins: The argument to be passed to plt.hist, either a number of bins or a binning
            algorithm name.
        :param str fit_method: The method that was used to fit the model, default is 'mcmc'.
        """
        if not outer_rad.isscalar:
            raise ValueError("Unfortunately this method can only display a distribution for one radius, so "
                             "arrays of radii are not supported.")

        gas_mass, gas_mass_dist = self.gas_mass(model, outer_rad, conf_level=conf_level, fit_method=fit_method)

        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        ax.yaxis.set_ticklabels([])

        plt.hist(gas_mass_dist.value, bins=bins, color=colour, alpha=0.7, density=False)
        plt.xlabel("Gas Mass [M$_{\odot}$]")
        plt.title("Gas Mass Distribution at {}".format(outer_rad.to_string()))

        mass_label = gas_mass.to("10^13Msun")
        vals_label = str(mass_label[0].round(2).value) + "^{+" + str(mass_label[2].round(2).value) + "}" + \
                     "_{-" + str(mass_label[1].round(2).value) + "}"
        res_label = r"$\rm{M_{gas}} = " + vals_label + "10^{13}M_{\odot}$"

        plt.axvline(gas_mass[0].value, color='red', label=res_label)
        plt.axvline(gas_mass[0].value-gas_mass[1].value, color='red', linestyle='dashed')
        plt.axvline(gas_mass[0].value+gas_mass[2].value, color='red', linestyle='dashed')
        plt.legend(loc='best', prop={'size': 12})
        plt.tight_layout()
        plt.show()

    def gas_mass_profile(self, model: str, radii: Quantity = None, deg_radii: Quantity = None,
                         fit_method: str = 'mcmc') -> GasMass1D:
        """
        A method to calculate and return a gas mass profile.

        :param str model: The name of the model from which to derive the gas mass.
        :param Quantity radii: The radii at which to measure gas masses. The default is None, in which
            case the radii at which this density profile has data points will be used.
        :param Quantity deg_radii: The equivelant radii to `radii` but in degrees, required for defining
            a profile. The default is None, but if custom radii are passed then this variable must be passed too.
        :param str fit_method: The method that was used to fit the model, default is 'mcmc'.
        :return: A cumulative gas mass distribution.
        :rtype: GasMass1D
        """
        if radii is None and self.radii[0] == 0:
            radii = self.radii[1:]
            deg_radii = self.deg_radii[1:]
        elif radii is None:
            radii = self.radii
            deg_radii = self.deg_radii
        elif radii is not None and not radii.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("The custom radii passed to this method cannot be converted to "
                                      "{}".format(self.radii_unit.to_string()))

        if radii is not None and deg_radii is None:
            raise ValueError('If a custom set of radii is passed then their equivalents in degrees must '
                             'also be passed')

        mass_vals = []
        mass_errs = []
        for rad in radii:
            gas_mass = self.gas_mass(model, rad, fit_method=fit_method)[0]
            mass_vals.append(gas_mass.value[0])
            mass_errs.append(gas_mass[1:].max().value)

        mass_vals = Quantity(mass_vals, 'Msun')
        mass_errs = Quantity(mass_errs, 'Msun')
        gm_prof = GasMass1D(radii, mass_vals, self.centre, self.src_name, self.obs_id, self.instrument,
                            self._gen_method, self._gen_prof, values_err=mass_errs, deg_radii=deg_radii)

        return gm_prof


class ProjectedGasTemperature1D(BaseProfile1D):
    """
    A profile product meant to hold a radial profile of projected X-ray temperature, as measured from a set
    of annular spectra by XSPEC. These are typically only defined by XGA methods.

    :param Quantity radii: The radii at which the projected gas temperatures have been measured, this should
        be in a proper radius unit, such as kpc.
    :param Quantity values: The projected gas temperatures that have been measured.
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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a 1D projected temperature profile. This profile
        will be considered unusable if a temperature value of greater than 30keV is present in the profile, or if a
        negative error value is detected (XSPEC can produce those).
        """
        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("keV"):
            raise UnitConversionError("Values unit cannot be converted to keV")

        # Setting the type
        self._prof_type = "1d_proj_temperature"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = "Projected Temperature"

        # This sets the profile to unusable if there is a problem with the data
        if self._values_err is not None and np.any((self._values+self._values_err) > Quantity(30, 'keV')):
            self._usable = False
        elif self._values_err is None and np.any(self._values > Quantity(30, 'keV')):
            self._usable = False

        # And this does the same but if there is a problem with the uncertainties
        if self._values_err is not None and np.any(self._values_err < Quantity(0, 'keV')):
            self._usable = False


class APECNormalisation1D(BaseProfile1D):
    """
    A profile product meant to hold a radial profile of XSPEC normalisation, as measured from a set of annular spectra
    by XSPEC. These are typically only defined by XGA methods. This is a useful profile because it allows to not
    only infer 3D profiles of temperature and metallicity, but can also allow us to infer the 3D density profile.

    :param Quantity radii: The radii at which the APEC normalisations have been measured, this should
        be in a proper radius unit, such as kpc.
    :param Quantity values: The APEC normalisations that have been measured.
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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a 1D APEC normalisation profile.
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

    def _gen_profile_setup(self, redshift: float, cosmo: Cosmology, abund_table: str = 'angr') \
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
            e_to_p_ratio = NHC[abund_table]
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

        return cur_rads, ang_dist, e_to_p_ratio

    def gas_density_profile(self, redshift: float, cosmo: Quantity, abund_table: str = 'angr', num_real: int = 10000,
                            sigma: int = 1, num_dens: bool = True) -> GasDensity3D:
        """
        A method to calculate the gas density profile from the APEC normalisation profile, which in turn was
        measured from XSPEC fits of an AnnularSpectra. This method supports the generation of both number density
        and mass density profiles through the use of the num_dens keyword.

        :param float redshift: The redshift of the source that this profile was generated from.
        :param cosmo: The chosen cosmology.
        :param str abund_table: The abundance table to used for the conversion from n_e x n_H to n_e^2 during density
            calculation. Default is the famous Anders & Grevesse table.
        :param int num_real: The number of data realisations which should be generated to infer density errors.
        :param int sigma: What sigma of error should the density profile be created with, the default is 1σ.
        :param bool num_dens: If True then a number density profile will be generated, otherwise a mass density profile
        will be generated.
        :return: The gas density profile which has been calculated from the APEC normalisation profile.
        :rtype: GasDensity3D
        """
        # There are commonalities between this method and others in this class, so I shifted some steps into an
        #  internal method which we will call now
        cur_rads, ang_dist, e_to_p_ratio = self._gen_profile_setup(redshift, cosmo, abund_table)

        # This uses a handy function I defined a while back to calculate the volume intersections between the annuli
        #  and spherical shells
        vol_intersects = shell_ann_vol_intersect(cur_rads, cur_rads)

        # This is essentially the constants bit of the XSPEC APEC normalisation
        # Angular diameter distance is calculated using the cosmology which was associated with the cluster
        #  at definition
        conv_factor = (4 * np.pi * (ang_dist * (1 + redshift)) ** 2) / (e_to_p_ratio * 10 ** -14)
        num_gas_scale = (1+e_to_p_ratio)
        conv_mass = MEAN_MOL_WEIGHT*m_p

        # Generating random normalisation profile realisations from DATA
        norm_real = self.generate_data_realisations(num_real, truncate_zero=True)

        if num_dens:
            gas_dens_reals = Quantity(np.zeros(norm_real.shape), "cm^-3")
        else:
            gas_dens_reals = Quantity(np.zeros(norm_real.shape), "kg cm^-3")

        # Using a loop here is ugly and relatively slow, but it should be okay
        for i in range(0, num_real):
            if num_dens:
                gas_dens_reals[i, :] = np.sqrt(np.linalg.inv(vol_intersects.T) @
                                               norm_real[i, :] * conv_factor) * num_gas_scale
            else:
                gas_dens_reals[i, :] = np.sqrt(np.linalg.inv(vol_intersects.T) @
                                               norm_real[i, :] * conv_factor) * num_gas_scale * conv_mass

        if not num_dens:
            # Convert the realisations to the correct unit
            gas_dens_reals = gas_dens_reals.to("Msun/Mpc^3")

        med_dens = np.nanpercentile(gas_dens_reals, 50, axis=0)
        # Calculates the standard deviation of each data point, this is how we estimate the density errors
        dens_sigma = np.nanstd(gas_dens_reals, axis=0)*sigma

        # Set up the actual profile object and return it
        dens_prof = GasDensity3D(self.radii, med_dens, self.centre, self.src_name, self.obs_id, self.instrument,
                                 'spec', self, self.radii_err, dens_sigma, self.set_ident,
                                 self.associated_set_storage_key, self.deg_radii)
        return dens_prof

    def emission_measure_profile(self, redshift: float, cosmo: Cosmology, abund_table: str = 'angr',
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
        conv_factor = (4 * np.pi * (ang_dist * (1 + redshift)) ** 2) / (10 ** -14)
        em_meas = self.values * conv_factor

        norm_real = self.generate_data_realisations(num_real, truncate_zero=True)
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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a radial emission measure profile.
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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a 1D projected metallicity/abundance profile.
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
    de-projection applied to a projected temperature profile.

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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None,  associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a radial 3D temperature profile.
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


# TODO WRITE CUSTOM STORAGE KEY HERE AS WELL
class BaryonFraction(BaseProfile1D):
    """
    A profile product which will hold a profile showing how the baryon fraction of a galaxy cluster changes
    with radius. These profiles are typically generated from a HydrostaticMass profile product instance.

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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None,  associated_set_id: int = None,
                 set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of a subclass of BaseProfile1D which will hold a radial baryon fraction profile.
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


class HydrostaticMass(BaseProfile1D):
    """
    A profile product which uses input GasTemperature3D and GasDensity3D profiles to generate a hydrostatic
    mass profile, which in turn can be used to measure the hydrostatic mass at a particular radius. In contrast
    to other profile objects, this one calculates the y values itself, as such any radii may be passed.

    :param GasTemperature3D temperature_profile: The XGA 3D temperature profile to take temperature
        information from.
    :param str/BaseModel1D temperature_model: The model to fit to the temperature profile, either a name or an
        instance of an XGA temperature model class.
    :param GasDensity3D density_profile: The XGA 3D density profile to take density information from.
    :param str/BaseModel1D density_model: The model to fit to the density profile, either a name or an
        instance of an XGA density model class.
    :param Quantity radii: The radii at which to measure the hydrostatic mass for the declaration of the profile.
    :param Quantity radii_err: The uncertainties on the radii.
    :param Quantity deg_radii: The radii values, but in units of degrees. This is required to set up a storage key
        for the profile to be filed in an XGA source.
    :param str fit_method: The name of the fit method to use for the fitting of the profiles, default is 'mcmc'.
    :param int num_walkers: If the fit method is 'mcmc' then this will set the number of walkers for the emcee
        sampler to set up.
    :param list/int num_steps: If the fit method is 'mcmc' this will set the number of steps for each sampler to
        take. If a single number is passed then that number of steps is used for both profiles, otherwise if a list
        is passed the first entry is used for the temperature fit, and the second for the density fit.
    :param int num_samples: The number of random samples to be drawn from the posteriors of the fit results.
    :param bool show_warn: Should warnings thrown during the fitting processes be shown.
    :param bool progress: Should fit progress bars be shown.
    """
    def __init__(self, temperature_profile: GasTemperature3D, temperature_model: Union[str, BaseModel1D],
                 density_profile: GasDensity3D, density_model: Union[str, BaseModel1D], radii: Quantity,
                 radii_err: Quantity, deg_radii: Quantity, fit_method: str = "mcmc", num_walkers: int = 20,
                 num_steps: [int, List[int]] = 20000, num_samples: int = 10000, show_warn: bool = True,
                 progress: bool = True):
        """
        The init method for the HydrostaticMass class, uses temperature and density profiles, along with models, to
        set up the hydrostatic mass profile.
        """
        # We check whether the temperature profile passed is actually the type of profile we need
        if type(temperature_profile) != GasTemperature3D:
            raise TypeError("Only a GasTemperature3D instance may be passed for temperature_profile, check "
                            "you haven't accidentally passed a ProjectedGasTemperature1D.")

        # We repeat this process with the density profile and model
        if type(density_profile) != GasDensity3D:
            raise TypeError("Only a GasDensity3D instance may be passed for density_profile, check you haven't "
                            "accidentally passed a GasDensity1D.")

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
            warn("The temperature and density profiles do not have the same associated ObsID.", stacklevel=2)
        elif temperature_profile.instrument != density_profile.instrument:
            warn("The temperature and density profiles do not have the same associated instrument.", stacklevel=2)

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
                warn("The temperature and density profile you passed were generated from different sets of annular"
                     " spectra, the mass profiles associated set ident will be set to None.", stacklevel=2)
                set_id = None
                set_store = None
            else:
                set_id = temperature_profile.set_ident
                set_store = temperature_profile.associated_set_storage_key

        self._temp_prof = temperature_profile
        self._dens_prof = density_profile

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")
        else:
            radii = radii.to('kpc')
            radii_err = radii_err.to('kpc')
        # This will be overwritten by the super() init call, but it allows rad_check to work
        self._radii = radii

        # We won't REQUIRE that the profiles have data point generated at the same radii, as we're gonna
        #  measure masses from the models, but I do need to check that the passed radii are within the radii of the
        #  and warn the user if they aren't
        self.rad_check(radii)

        if isinstance(num_steps, int):
            temp_steps = num_steps
            dens_steps = num_steps
        elif isinstance(num_steps, list) and len(num_steps) == 2:
            temp_steps = num_steps[0]
            dens_steps = num_steps[1]
        else:
            raise ValueError("If a list is passed for num_steps then it must have two entries, the first for the "
                             "temperature profile fit and the second for the density profile fit")

        # Make sure the model fits have been run, and retrieve the model objects
        temperature_model = temperature_profile.fit(temperature_model, fit_method, num_samples, temp_steps, num_walkers,
                                                    progress, show_warn)
        density_model = density_profile.fit(density_model, fit_method, num_samples, dens_steps, num_walkers, progress,
                                            show_warn)

        # Have to check whether the fits were actually successful, as the fit method will return a model instance
        #  either way
        if not temperature_model.success:
            raise XGAFitError("The fit to the temperature was unsuccessful, cannot define hydrostatic mass profile.")
        if not density_model.success:
            raise XGAFitError("The fit to the density was unsuccessful, cannot define hydrostatic mass profile.")

        self._temp_model = temperature_model
        self._dens_model = density_model

        mass, mass_dist = self.mass(radii, conf_level=68)
        mass_vals = mass[0, :]
        mass_errs = np.mean(mass[1:, :], axis=0)

        super().__init__(radii, mass_vals, self._temp_prof.centre, self._temp_prof.src_name, self._temp_prof.obs_id,
                         self._temp_prof.instrument, radii_err, mass_errs, set_id, set_store, deg_radii)

        # Need a custom storage key for this mass profile, incorporating all the information we have about what
        #  went into it, density profile, temperature profile, radii, density and temperature models.
        dens_part = "dprof_{}".format(self._dens_prof.storage_key)
        temp_part = "tprof_{}".format(self._temp_prof.storage_key)
        cur_part = self.storage_key
        new_part = "tm{t}_dm{d}".format(t=self._temp_model.name, d=self._dens_model.name)
        whole_new = "{n}_{c}_{t}_{d}".format(n=new_part, c=cur_part, t=temp_part, d=dens_part)
        self._storage_key = whole_new

        # Setting the type
        self._prof_type = "hydrostatic_mass"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = r"M$_{\rm{hydro}}$"

        # Setting up a dictionary to store hydro mass results in.
        self._masses = {}

    def mass(self, radius: Quantity, conf_level: float = 68.2,
             radius_err: Quantity = None) -> Union[Quantity, Quantity]:
        """
        A method which will measure a hydrostatic mass and hydrostatic mass uncertainty within the given
        radius/radii. No corrections are applied to the values calculated by this method, it is just the vanilla
        hydrostatic mass.

        If the models for temperature and density have analytical solutions to their derivative wrt to radius then
        those will be used to calculate the gradients at radius, but if not then a numerical method will be used for
        which dx will be set to radius/1e+6.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            mass within.
        :param float conf_level: The confidence level for the mass uncertainties, the default is 68.2% (~1σ).
        :param Quantity radius_err: A standard deviation on radius, which will be taken into account during the
            calculation of hydrostatic mass.
        :return: An astropy quantity containing the mass/masses, lower and upper uncertainties, and another containing
            the mass realisation distribution.
        :rtype: Union[Quantity, Quantity]
        """
        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        # Prints a warning of the mass is outside the range of the data
        self.rad_check(radius)

        # We need check that, if the user has passed uncertainty information on radii, it is how we expect it to be.
        #  First off, are there the right number of entries?
        if not radius.isscalar and radius_err is not None and (radius_err.isscalar or len(radius) != len(radius_err)):
            raise ValueError("If a set of radii are passed, and radius uncertainty information is provided, the "
                             "'radius_err' argument must contain the same number of entries as the 'radius' argument.")
        # Same deal here, if only one radius is passed, only one error may be passed
        elif radius.isscalar and radius_err is not None and not radius_err.isscalar:
            raise ValueError("When a radius uncertainty ('radius_err') is passed for a single radius value, "
                             "'radius_err' must be scalar.")
        # Now we check that the units of the radius and radius error quantities are compatible
        elif radius_err is not None and not radius_err.unit.is_equivalent(radius.unit):
            raise UnitConversionError("The radius_err quantity must be in units that are equivalent to units "
                                      "of {}.".format(radius.unit.to_string()))

        # Now we make absolutely sure that the radius error(s) are in the correct units
        if radius_err is not None:
            radius_err = radius_err.to(self.radii_unit)

        # Here we construct the storage key for the radius passed, and the uncertainty if there is one
        if radius.isscalar and radius_err is None:
            stor_key = str(radius.value) + " " + str(radius.unit)
        elif radius.isscalar and radius_err is not None:
            stor_key = str(radius.value) + '_' + str(radius_err.value) + " " + str(radius.unit)
        # In this case, as the radius is not scalar, the masses won't be stored so we don't need a storage key
        else:
            stor_key = None

        # Check to see whether the calculation has to be run again
        if radius.isscalar and stor_key in self._masses:
            already_run = True
            mass_dist = self._masses[stor_key]
        else:
            already_run = False

        # If the models don't have analytical solutions to their derivative then the derivative method will need
        #  a dx to assume, so I will set one equal to radius/1e+6 (or the max radius if non-scalar), should be
        #  small enough.
        if radius.isscalar:
            dx = radius/1e+6
        else:
            dx = radius.max()/1e+6

        # Declaring this allows us to randomly draw from Gaussians, if the user has given us radius error information
        rng = np.random.default_rng()
        # In this case a single radius value, and a radius uncertainty has been passed
        if radius.isscalar and radius_err is not None:
            # We just want one a single distribution of radius here, but make sure that it is in a (1, N) shaped array
            #  as some downstream tasks in model classes, such as get_realisations and derivative, want radius
            #  DISTRIBUTIONS to be 2dim arrays, and multiple radius VALUES (e.g. [1, 2, 3, 4]) to be 1dim arrays
            calc_rad = Quantity(rng.normal(radius.value, radius_err.value, (1, len(self._dens_model.par_dists[0]))),
                                radius_err.unit)
        # In this case multiple radius values have been passed, each with an uncertainty
        elif not radius.isscalar and radius_err is not None:
            # So here we're setting up M radius distributions, where M is the number of input radii. So this radius
            #  array ends up being shape (M, N), where M is the number of radii, and M is the number of samples in
            #  the model posterior distributions
            calc_rad = Quantity(rng.normal(radius.value, radius_err.value,
                                           (len(self._dens_model.par_dists[0]), len(radius))), radius_err.unit).T

        # This is the simplest case, just a radius (or a set of radii) with no uncertainty information has been passed
        else:
            calc_rad = radius

        # If we need to do the calculation (i.e. no existing result is available, and the model fits worked) then
        #  we had best get to it!
        if not already_run and self._dens_model.success and self._temp_model.success:
            # This grabs gas density values from the density model, need to check whether the model is in units
            #  of mass or number density
            if self._dens_model.y_unit.is_equivalent('1/cm^3'):
                dens = self._dens_model.get_realisations(calc_rad)
                dens_der = self._dens_model.derivative(calc_rad, dx, True)
            else:
                dens = self._dens_model.get_realisations(calc_rad) / (MEAN_MOL_WEIGHT*m_p)
                dens_der = self._dens_model.derivative(calc_rad, dx, True) / (MEAN_MOL_WEIGHT*m_p)

            # We do the same for the temperature vals, again need to check the units
            if self._temp_model.y_unit.is_equivalent("keV"):
                temp = (self._temp_model.get_realisations(calc_rad)/k_B).to('K')
                temp_der = self._temp_model.derivative(calc_rad, dx, True)/k_B
                temp_der = temp_der.to(Unit('K')/self._temp_model.x_unit)
            else:
                temp = self._temp_model.get_realisations(calc_rad).to('K')
                temp_der = self._temp_model.derivative(calc_rad, dx, True).to('K')

            # Please note that this is just the vanilla hydrostatic mass equation, but not written in the "standard
            #  form". Here there are no logs in the derivatives, because it's easier to take advantage of astropy's
            #  quantities that way.
            mass_dist = ((-1 * k_B * np.power(radius[..., None], 2)) / (dens * (MEAN_MOL_WEIGHT*m_p) * G)) * \
                        ((dens * temp_der) + (temp * dens_der))

            # Just converts the mass/masses to the unit we normally use for them
            mass_dist = mass_dist.to('Msun').T

            if radius.isscalar:
                self._masses[stor_key] = mass_dist

        elif not self._temp_model.success or not self._dens_model.success:
            raise XGAFitError("One or both of the fits to the temperature model and density profiles were "
                              "not successful")

        mass_med = np.percentile(mass_dist, 50, axis=0)
        mass_lower = mass_med - np.percentile(mass_dist, lower, axis=0)
        mass_upper = np.percentile(mass_dist, upper, axis=0) - mass_med

        mass_res = Quantity(np.array([mass_med.value, mass_lower.value, mass_upper.value]), mass_dist.unit)

        # We check to see if any of the upper limits (i.e. measured value plus +ve error) are below zero, and if so
        #  then we throw an exception up
        if np.any((mass_res[0] + mass_res[1]) < 0):
            raise ValueError("A mass upper limit (i.e. measured value plus +ve error) of less than zero has been "
                             "measured, which is not physical.")

        return mass_res, mass_dist

    def annular_mass(self, outer_radius: Quantity, inner_radius: Quantity, conf_level: float = 68.2):
        """
        Calculate the hydrostatic mass contained within a specific 3D annulus, bounded by the outer and inner radius
        supplied to this method. Annular mass is calculated by measuring the mass within the inner and outer
        radii, and then subtracting the inner from the outer. Also supports calculating multiple annular masses
        when inner_radius and outer_radius are non-scalar.

        WARNING - THIS METHOD INVOLVES SUBTRACTING TWO MASS DISTRIBUTIONS, WHICH CAN'T NECESSARILY BE APPROXIMATED
        AS GAUSSIAN DISTRIBUTIONS, AS SUCH RESULTS FROM THIS METHOD SHOULD BE TREATED WITH SOME SUSPICION.

        :param Quantity outer_radius: Astropy containing outer radius (or radii) for the annulus (annuli) within
            which you wish to measure the mass. If calculating multiple annular masses, the length of outer_radius
            must be the same as inner_radius.
        :param Quantity inner_radius: Astropy containing inner radius (or radii) for the annulus (annuli) within
            which you wish to measure the mass. If calculating multiple annular masses, the length of inner_radius
            must be the same as outer_radius.
        :param float conf_level: The confidence level for the mass uncertainties, the default is 68.2% (~1σ).
        :return: An astropy quantity containing a mass distribution(s). Quantity will become two-dimensional
            when multiple sets of inner and outer radii are passed by the user.
        :rtype: Quantity
        """
        # Perform some checks to make sure that the user has passed inner and outer radii quantities that are valid
        #  and won't break any of the calculations that will be happening in this method
        if outer_radius.isscalar != inner_radius.isscalar:
            raise ValueError("The outer_radius and inner_radius Quantities must both be scalar, or both "
                             "be non-scalar.")
        elif (not inner_radius.isscalar and inner_radius.ndim != 1) or \
                (not outer_radius.isscalar and outer_radius.ndim != 1):
            raise ValueError('Non-scalar radius Quantities must have only one dimension')
        elif not outer_radius.isscalar and not inner_radius.isscalar and outer_radius.shape != inner_radius.shape:
            raise ValueError('The outer_radius and inner_radius Quantities must be the same shape.')

        # This just measures the masses within two radii, the outer and the inner supplied by the user. The mass()
        #  method will automatically deal with the input of multiple entries for each radius
        outer_mass, outer_mass_dist = self.mass(outer_radius, conf_level)
        inner_mass, inner_mass_dist = self.mass(inner_radius, conf_level)

        # This PROBABLY NOT AT ALL valid because they're just posterior distributions of mass
        return outer_mass_dist - inner_mass_dist

    def view_mass_dist(self, radius: Quantity, conf_level: float = 68.2, figsize=(8, 8), bins: Union[str, int] = 'auto',
                       colour: str = "lightslategrey"):
        """
        A method which will generate a histogram of the mass distribution that resulted from the mass calculation
        at the supplied radius. If the mass for the passed radius has already been measured it, and the mass
        distribution, will be retrieved from the storage of this product rather than re-calculated.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            mass within.
        :param float conf_level: The confidence level for the mass uncertainties, the default is 68.2% (~1σ).
        :param int/str bins: The argument to be passed to plt.hist, either a number of bins or a binning
            algorithm name.
        :param str colour: The desired colour of the histogram.
        :param tuple figsize: The desired size of the histogram figure.
        """
        if not radius.isscalar:
            raise ValueError("Unfortunately this method can only display a distribution for one radius, so "
                             "arrays of radii are not supported.")

        # Grabbing out the mass distribution, as well as the single result that describes the mass distribution.
        hy_mass, hy_dist = self.mass(radius, conf_level)
        # Setting up the figure
        plt.figure(figsize=figsize)
        ax = plt.gca()
        # Includes nicer ticks
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        # And removing the yaxis tick labels as its just a number of values per bin
        ax.yaxis.set_ticklabels([])

        # Plot the histogram and set up labels
        plt.hist(hy_dist.value, bins=bins, color=colour, alpha=0.7, density=False)
        plt.xlabel(self._y_axis_name + " M$_{\odot}$")
        plt.title("Mass Distribution at {}".format(radius.to_string()))

        lab_hy_mass = hy_mass.to("10^14Msun")
        vals_label = str(lab_hy_mass[0].round(2).value) + "^{+" + str(lab_hy_mass[2].round(2).value) + "}" + \
                     "_{-" + str(lab_hy_mass[1].round(2).value) + "}"
        res_label = r"$\rm{M_{hydro}} = " + vals_label + "10^{14}M_{\odot}$"

        # And this just plots the 'result' on the distribution as a series of vertical lines
        plt.axvline(hy_mass[0].value, color='red', label=res_label)
        plt.axvline(hy_mass[0].value-hy_mass[1].value, color='red', linestyle='dashed')
        plt.axvline(hy_mass[0].value+hy_mass[2].value, color='red', linestyle='dashed')
        plt.legend(loc='best', prop={'size': 12})
        plt.tight_layout()
        plt.show()

    def baryon_fraction(self, radius: Quantity, conf_level: float = 68.2) -> Tuple[Quantity, Quantity]:
        """
        A method to use the hydrostatic mass information of this profile, and the gas density information of the
        input gas density profile, to calculate a baryon fraction within the given radius.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            baryon fraction within.
        :param float conf_level: The confidence level for the mass uncertainties, the default is 68.2% (~1σ).
        :return: An astropy quantity containing the baryon fraction, -ve error, and +ve error, and another quantity
            containing the baryon fraction distribution.
        :rtype: Tuple[Quantity, Quantity]
        """
        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        if not radius.isscalar:
            raise ValueError("Unfortunately this method can only calculate the baryon fraction within one "
                             "radius, multiple radii are not supported.")

        # Grab out the hydrostatic mass distribution, and the gas mass distribution
        hy_mass, hy_mass_dist = self.mass(radius, conf_level)
        gas_mass, gas_mass_dist = self._dens_prof.gas_mass(self._dens_model.name, radius, conf_level=conf_level,
                                                           fit_method=self._dens_model.fit_method)

        # If the distributions don't have the same number of entries (though as far I can recall they always should),
        #  then we just make sure we have two equal length distributions to divide
        if len(hy_mass_dist) < len(gas_mass_dist):
            bar_frac_dist = gas_mass_dist[:len(hy_mass_dist)] / hy_mass_dist
        elif len(hy_mass_dist) > len(gas_mass_dist):
            bar_frac_dist = gas_mass_dist / hy_mass_dist[:len(gas_mass_dist)]
        else:
            bar_frac_dist = gas_mass_dist / hy_mass_dist

        bfrac_med = np.percentile(bar_frac_dist, 50, axis=0)
        bfrac_lower = bfrac_med - np.percentile(bar_frac_dist, lower, axis=0)
        bfrac_upper = np.percentile(bar_frac_dist, upper, axis=0) - bfrac_med
        bar_frac_res = Quantity([bfrac_med.value, bfrac_lower.value, bfrac_upper.value])

        return bar_frac_res, bar_frac_dist

    def view_baryon_fraction_dist(self, radius: Quantity, conf_level: float = 68.2, figsize=(8, 8),
                                  bins: Union[str, int] = 'auto', colour: str = "lightslategrey"):
        """
        A method which will generate a histogram of the baryon fraction distribution that resulted from the mass
        calculation at the supplied radius. If the baryon fraction for the passed radius has already been
        measured it, and the baryon fraction distribution, will be retrieved from the storage of this product
        rather than re-calculated.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            baryon fraction within.
        :param float conf_level: The confidence level for the mass uncertainties, the default is 68.2% (~1σ).
        :param int/str bins: The argument to be passed to plt.hist, either a number of bins or a binning
            algorithm name.
        :param tuple figsize: The desired size of the histogram figure.
        :param str colour: The desired colour of the histogram.
        """
        if not radius.isscalar:
            raise ValueError("Unfortunately this method can only display a distribution for one radius, so "
                             "arrays of radii are not supported.")

        bar_frac, bar_frac_dist = self.baryon_fraction(radius, conf_level)
        plt.figure(figsize=figsize)
        ax = plt.gca()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        ax.yaxis.set_ticklabels([])

        plt.hist(bar_frac_dist.value, bins=bins, color=colour, alpha=0.7)
        plt.xlabel("Baryon Fraction")
        plt.title("Baryon Fraction Distribution at {}".format(radius.to_string()))

        vals_label = str(bar_frac[0].round(2).value) + "^{+" + str(bar_frac[2].round(2).value) + "}" + \
                        "_{-" + str(bar_frac[1].round(2).value) + "}"
        res_label = r"$\rm{f_{gas}} = " + vals_label + "$"

        plt.axvline(bar_frac[0].value, color='red', label=res_label)
        plt.axvline(bar_frac[0].value-bar_frac[1].value, color='red', linestyle='dashed')
        plt.axvline(bar_frac[0].value+bar_frac[2].value, color='red', linestyle='dashed')
        plt.legend(loc='best', prop={'size': 12})
        plt.xlim(0)
        plt.tight_layout()
        plt.show()

    def baryon_fraction_profile(self) -> BaryonFraction:
        """
        A method which uses the baryon_fraction method to construct a baryon fraction profile at the radii of
        this HydrostaticMass profile. The uncertainties on the baryon fraction are calculated at the 1σ level.

        :return: An XGA BaryonFraction object.
        :rtype: BaryonFraction
        """
        frac = []
        frac_err = []
        # Step through the radii of this profile
        for rad in self.radii:
            # Grabs the baryon fraction for the current radius
            b_frac = self.baryon_fraction(rad)[0]

            # Only need the actual result, not the distribution
            frac.append(b_frac[0])
            # Calculates a mean uncertainty
            frac_err.append(b_frac[1:].mean())

        # Makes them unit-less quantities, as baryon fraction is mass/mass
        frac = Quantity(frac, '')
        frac_err = Quantity(frac_err, '')

        return BaryonFraction(self.radii, frac, self.centre, self.src_name, self.obs_id, self.instrument,
                              self.radii_err, frac_err, self.set_ident, self.associated_set_storage_key,
                              self.deg_radii)

    def overdensity_radius(self, delta: int, redshift: float, cosmo, init_lo_rad: Quantity = Quantity(100, 'kpc'),
                           init_hi_rad: Quantity = Quantity(3500, 'kpc'), init_step: Quantity = Quantity(100, 'kpc'),
                           out_unit: Union[Unit, str] = Unit('kpc')) -> Quantity:
        """
        This method uses the mass profile to find the radius that corresponds to the user-supplied
        overdensity - common choices for cluster analysis are Δ=2500, 500, and 200. Overdensity radii are
        defined as the radius at which the density is Δ times the critical density of the Universe at the
        cluster redshift.

        This method takes a numerical approach to the location of the requested radius. Though we have calculated
        analytical hydrostatic mass models for common choices of temperature and density profile models, there are
        no analytical solutions for R.

        When an overdensity radius is being calculated, we initially measure masses for a range of radii between
        init_lo_rad - init_hi_rad in steps of init_step. From this we find the two radii that bracket the radius where
        average density - Delta*critical density = 0. Between those two radii we perform the same test with another
        range of radii (in steps of 1 kpc this time), finding the radius that corresponds to the minimum
        density difference value.

        :param int delta: The overdensity factor for which a radius is to be calculated.
        :param float redshift: The redshift of the cluster.
        :param cosmo: The cosmology in which to calculate the overdensity. Should be an astropy cosmology instance.
        :param Quantity init_lo_rad: The lower radius bound for the first radii array generated to find the wide
            brackets around the requested overdensity radius. Default value is 100 kpc.
        :param Quantity init_hi_rad: The upper radius bound for the first radii array generated to find the wide
            brackets around the requested overdensity radius. Default value is 3500 kpc.
        :param Quantity init_step: The step size for the first radii array generated to find the wide brackets
            around the requested overdensity radius. Default value is 100 kpc, recommend that you don't set it
            smaller than 10 kpc.
        :param Unit/str out_unit: The unit that this method should output the radius with.
        :return: The calculated overdensity radius.
        :rtype: Quantity
        """
        def turning_point(brackets: Quantity, step_size: Quantity) -> Quantity:
            """
            This is the meat of the overdensity_radius method. It goes looking for radii that bracket the
            requested overdensity radius. This works by calculating an array of masses, calculating densities
            from them and the radius array, then calculating the difference between Delta*critical density at
            source redshift. Where the difference array flips from being positive to negative is where the
            bracketing radii are.

            :param Quantity brackets: The brackets within which to generate our array of radii.
            :param Quantity step_size: The step size for the array of radii
            :return: The bracketing radii for the requested overdensity for this search.
            :rtype: Quantity
            """
            # Just makes sure that the step size is definitely in the same unit as the bracket
            #  variable, as I take the value of step_size later
            step_size = step_size.to(brackets.unit)

            # This sets up a range of radii within which to calculate masses, which in turn are used to find the
            #  closest value to the Delta*critical density we're looking for
            rads = Quantity(np.arange(*brackets.value, step_size.value), 'kpc')
            # The masses contained within the test radii, the transpose is just there because the array output
            #  by that function is weirdly ordered - there is an issue open that will remind to eventually change that
            rad_masses = self.mass(rads)[0].T
            # Calculating the density from those masses - uses the radii that the masses were measured within
            rad_dens = rad_masses[:, 0] / (4 * np.pi * (rads ** 3) / 3)
            # Finds the difference between the density array calculated above and the requested
            #  overdensity (i.e. Delta * the critical density of the Universe at the source redshift).
            rad_dens_diffs = rad_dens - (delta * z_crit_dens)

            if np.all(rad_dens_diffs.value > 0) or np.all(rad_dens_diffs.value < 0):
                raise ValueError("The passed lower ({l}) and upper ({u}) radii don't appear to bracket the "
                                 "requested overdensity (Delta={d}) radius.".format(l=brackets[0], u=brackets[1],
                                                                                    d=delta))

            # This finds the index of the radius where the turnover between the density difference being
            #  positive and negative happens. The radius of that index, and the index before it, bracket
            #  the requested overdensity.
            turnover = np.where(rad_dens_diffs.value < 0, rad_dens_diffs.value, -np.inf).argmax()
            brackets = rads[[turnover - 1, turnover]]

            return brackets

        # First perform some sanity checks to make sure that the user hasn't passed anything silly
        # Check that the overdensity is a positive, non-zero (because that wouldn't make sense) integer.
        if not type(delta) == int or delta <= 0:
            raise ValueError("The overdensity must be a positive, non-zero, integer.")

        # The user is allowed to pass either a unit instance or a string, we make sure the out_unit is consistently
        #  a unit instance for the benefit of the rest of this method.
        if isinstance(out_unit, str):
            out_unit = Unit(out_unit)
        elif not isinstance(out_unit, Unit):
            raise ValueError("The out_unit argument must be either an astropy Unit instance, or a string "
                             "representing an astropy unit.")

        # We know that if we have arrived here then the out_unit variable is a Unit instance, so we just check
        #  that it's a distance unit that makes sense. I haven't allowed degrees, arcmins etc. because it would
        #  entail a little extra work, and I don't care enough right now.
        if not out_unit.is_equivalent('kpc'):
            raise UnitConversionError("The out_unit argument must be supplied with a unit that is convertible "
                                      "to kpc. Angular units such as deg are not currently supported.")

        # Obviously redshift can't be negative, and I won't allow zero redshift because it doesn't
        #  make sense for clusters and completely changes how distance calculations are done.
        if redshift <= 0:
            raise ValueError("Redshift cannot be less than or equal to zero.")

        # This is the critical density of the Universe at the cluster redshift - this is what we compare the
        #  cluster density too to figure out the requested overdensity radius.
        z_crit_dens = cosmo.critical_density(redshift)

        wide_bracket = turning_point(Quantity([init_lo_rad, init_hi_rad]), init_step)
        if init_step != Quantity(1, 'kpc'):
            # In this case I buffer the wide bracket (subtract 5 kpc from the lower bracket and add 5 kpc to the upper
            #  bracket) - this is a fix to help avoid errors when the turning point is equal to the upper or lower
            #  bracket
            buffered_wide_bracket = wide_bracket + Quantity([-5, 5], 'kpc')
            tight_bracket = turning_point(buffered_wide_bracket, Quantity(1, 'kpc'))
        else:
            tight_bracket = wide_bracket

        return ((tight_bracket[0]+tight_bracket[1])/2).to(out_unit)

    def _diag_view_prep(self, src) -> Tuple[int, RateMap, SurfaceBrightness1D]:
        """
        This internal function just serves to grab the relevant photometric products (if available) and check to
        see how many plots will be in the diagnostic view. The maximum is five; mass profile, temperature profile,
        density profile, surface brightness profile, and ratemap.

        :param GalaxyCluster src: The source object for which this hydrostatic mass profile was created
        :return: The number of plots, a RateMap (if src was pass, otherwise None), and a SB profile (if the
            density profile was created with the SB method, otherwise None).
        :rtype: Tuple[int, RateMap, SurfaceBrightness1D]
        """

        # This checks to make sure that the source is a galaxy cluster, I do it this way (with strings) to avoid
        #  annoying circular import errors. The source MUST be a galaxy cluster because you can only calculate
        #  hydrostatic mass profiles for galaxy clusters.
        if src is not None and type(src).__name__ != 'GalaxyCluster':
            raise TypeError("The src argument must be a GalaxyCluster object.")

        # This just checks to make sure that the name of the passed source is the same as the stored source name
        #  of this profile. Maybe in the future this won't be necessary because a reference to the source
        #  will be stored IN the profile.
        if src is not None and src.name != self.src_name:
            raise ValueError("The passed source has a different name to the source that was used to generate"
                             " this HydrostaticMass profile.")

        # If the hydrostatic mass profile was created using combined data then I grab a combined image
        if self.obs_id == 'combined' and src is not None:
            rt = src.get_combined_ratemaps(src.peak_lo_en, src.peak_hi_en)
        # Otherwise we grab the specific relevant image
        elif self.obs_id != 'combined' and src is not None:
            rt = src.get_ratemaps(self.obs_id, self.instrument, src.peak_lo_en, src.peak_hi_en)
        # If there is no source passed, then we don't get a ratemap
        else:
            rt = None

        # Checks to see whether the generation profile of the density profile is a surface brightness
        #  profile. The other option is that it's an apec normalisation profile if generated from the spectra method
        if type(self.density_profile.generation_profile) == SurfaceBrightness1D:
            sb = self.density_profile.generation_profile
        # Otherwise there is no SB profile
        else:
            sb = None

        # Maximum number of plots is five, this just figures out how many there are going to be based on what the
        #  ratemap and surface  brightness profile values are
        num_plots = 5 - sum([rt is None, sb is None])

        return num_plots, rt, sb

    def _gen_diag_view(self, fig: Figure, src, num_plots: int, rt: RateMap, sb: SurfaceBrightness1D):
        """
        This populates the diagnostic plot figure, grabbing axes from various classes of profile product.

        :param Figure fig: The figure instance being populated.
        :param GalaxyCluster src: The galaxy cluster source that this hydrostatic mass profile was created for.
        :param int num_plots: The number of plots in this diagnostic view.
        :param RateMap rt: A RateMap to add to this diagnostic view.
        :param SurfaceBrightness1D sb: A surface brightness profile to add to this diagnostic view.
        :return: The axes array of this diagnostic view.
        :rtype: np.ndarray([Axes])
        """
        from ..imagetools.misc import physical_rad_to_pix

        # The preparation method has already figured out how many plots there will be, so we create those subplots
        ax_arr = fig.subplots(nrows=1, ncols=num_plots)

        # If a RateMap has been passed then we need to get the view, calculate some things, and then add it to our
        #  diagnostic plot
        if rt is not None:
            # As the RateMap is the first plot, and is not guaranteed to be present, I use the offset parameter
            #  later in this function to shift the other plots across by 1 if it is present.
            offset = 1
            # If the source was setup to use a peak coordinate, then we want to include that in the ratemap display
            if src.use_peak:
                ch = Quantity([src.peak, src.ra_dec])
                # I also grab the annulus boundaries from the temperature profile used to create this
                #  HydrostaticMass profile, then convert to pixels. That does depend on there being a source, but
                #  we know that we wouldn't have a RateMap at this point if the user hadn't passed a source
                pix_rads = physical_rad_to_pix(rt, self.temperature_profile.annulus_bounds, src.peak, src.redshift,
                                               src.cosmo)

            else:
                # No peak means we just use the original user-passed RA-Dec
                ch = src.ra_dec
                pix_rads = physical_rad_to_pix(rt, self.temperature_profile.annulus_bounds, src.ra_dec, src.redshift,
                                               src.cosmo)

            # This gets the nicely setup view from the RateMap object and adds it to our array of matplotlib axes
            ax_arr[0] = rt.get_view(ax_arr[0], ch, radial_bins_pix=pix_rads.value)
        else:
            # In this case there is no RateMap to add, so I don't need to shift the other plots across
            offset = 0

        # These simply plot the mass, temperature, and density profiles with legends turned off, residuals turned
        #  off, and no title
        ax_arr[0+offset] = self.get_view(fig, ax_arr[0+offset], show_legend=False, custom_title='',
                                         show_residual_ax=False)[0]
        ax_arr[1+offset] = self.temperature_profile.get_view(fig, ax_arr[1+offset], show_legend=False, custom_title='',
                                                             show_residual_ax=False)[0]
        ax_arr[2+offset] = self.density_profile.get_view(fig, ax_arr[2+offset], show_legend=False, custom_title='',
                                                         show_residual_ax=False)[0]
        # Then if there is a surface brightness profile thats added too
        if sb is not None:
            ax_arr[3+offset] = sb.get_view(fig, ax_arr[3+offset], show_legend=False, custom_title='',
                                           show_residual_ax=False)[0]

        return ax_arr

    def diagnostic_view(self, src=None, figsize: tuple = None):
        """
        This method produces a figure with the most important products that went into the creation of this
        HydrostaticMass profile, for the purposes of quickly checking that everything looks sensible. The
        maximum number of plots included is five; mass profile, temperature profile, density profile,
        surface brightness profile, and ratemap. The RateMap will only be included if the source that this profile
        was generated from is passed.

        :param GalaxyCluster src: The GalaxyCluster source that this HydrostaticMass profile was generated from.
        :param tuple figsize: A tuple that sets the size of the diagnostic plot, default is None in which case
            it is set automatically.
        """

        # Run the preparatory method to get the number of plots, RateMap, and SB profile - also performs
        #  some common sense checks if a source has been passed.
        num_plots, rt, sb = self._diag_view_prep(src)

        # Calculate a sensible figsize if the user didn't pass one
        if figsize is None:
            figsize = (7.2*num_plots, 7)

        # Set up the figure
        fig = plt.figure(figsize=figsize)
        # Set up and populate the axes with plots
        ax_arr = self._gen_diag_view(fig, src, num_plots, rt, sb)

        # And show the figure
        plt.tight_layout()
        plt.show()

        plt.close('all')

    def save_diagnostic_view(self, save_path: str, src=None, figsize: tuple = None):
        """
        This method saves a figure (without displaying) with the most important products that went into the creation
        of this HydrostaticMass profile, for the purposes of quickly checking that everything looks sensible. The
        maximum number of plots included is five; mass profile, temperature profile, density profile, surface
        brightness profile, and ratemap. The RateMap will only be included if the source that this profile
        was generated from is passed.

        :param str save_path: The path and filename where the diagnostic figure should be saved.
        :param GalaxyCluster src: The GalaxyCluster source that this HydrostaticMass profile was generated from.
        :param tuple figsize: A tuple that sets the size of the diagnostic plot, default is None in which case
            it is set automatically.
        """
        # Run the preparatory method to get the number of plots, RateMap, and SB profile - also performs
        #  some common sense checks if a source has been passed.
        num_plots, rt, sb = self._diag_view_prep(src)

        # Calculate a sensible figsize if the user didn't pass one
        if figsize is None:
            figsize = (7.2*num_plots, 7)

        # Set up the figure
        fig = plt.figure(figsize=figsize)
        # Set up and populate the axes with plots
        ax_arr = self._gen_diag_view(fig, src, num_plots, rt, sb)

        # And show the figure
        plt.tight_layout()
        plt.savefig(save_path)

        plt.close('all')

    @property
    def temperature_profile(self) -> GasTemperature3D:
        """
        A method to provide access to the 3D temperature profile used to generate this hydrostatic mass profile.

        :return: The input temperature profile.
        :rtype: GasTemperature3D
        """
        return self._temp_prof

    @property
    def density_profile(self) -> GasDensity3D:
        """
        A method to provide access to the 3D density profile used to generate this hydrostatic mass profile.

        :return: The input density profile.
        :rtype: GasDensity3D
        """
        return self._dens_prof

    @property
    def temperature_model(self) -> BaseModel1D:
        """
        A method to provide access to the model that was fit to the temperature profile.

        :return: The fit temperature model.
        :rtype: BaseModel1D
        """
        return self._temp_model

    @property
    def density_model(self) -> BaseModel1D:
        """
        A method to provide access to the model that was fit to the density profile.

        :return: The fit density profile.
        :rtype: BaseModel1D
        """
        return self._dens_model

    def rad_check(self, rad: Quantity):
        """
        Very simple method that prints a warning if the radius is outside the range of data covered by the
        density or temperature profiles.

        :param Quantity rad: The radius to check.
        """
        if not rad.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("You can only check radii in units convertible to the radius units of "
                                      "the profile ({})".format(self.radii_unit.to_string()))

        if (self._temp_prof.annulus_bounds is not None and (rad > self._temp_prof.annulus_bounds[-1]).any()) \
                or (self._dens_prof.annulus_bounds is not None and (rad > self._dens_prof.annulus_bounds[-1]).any()):
            warn("Some radii are outside the data range covered by the temperature or density profiles, as such "
                 "you will be extrapolating based on the model fits.", stacklevel=2)


class SpecificEntropy(BaseProfile1D):
    """
    A profile product which uses input GasTemperature3D and GasDensity3D profiles to generate a specific entropy
    profile. Functionally this is extremely similar to the HydrostaticMass profile class, as it calculates the y
    values itself, rather than them being part of the declaration.

    :param GasTemperature3D temperature_profile: The XGA 3D temperature profile to take temperature
        information from.
    :param str/BaseModel1D temperature_model: The model to fit to the temperature profile, either a name or an
        instance of an XGA temperature model class.
    :param GasDensity3D density_profile: The XGA 3D density profile to take density information from.
    :param str/BaseModel1D density_model: The model to fit to the density profile, either a name or an
        instance of an XGA density model class.
    :param Quantity radii: The radii at which to measure the entropy for the declaration of the profile.
    :param Quantity radii_err: The uncertainties on the radii.
    :param Quantity deg_radii: The radii values, but in units of degrees. This is required to set up a storage key
        for the profile to be filed in an XGA source.
    :param str fit_method: The name of the fit method to use for the fitting of the profiles, default is 'mcmc'.
    :param int num_walkers: If the fit method is 'mcmc' then this will set the number of walkers for the emcee
        sampler to set up.
    :param list/int num_steps: If the fit method is 'mcmc' this will set the number of steps for each sampler to
        take. If a single number is passed then that number of steps is used for both profiles, otherwise if a list
        is passed the first entry is used for the temperature fit, and the second for the density fit.
    :param int num_samples: The number of random samples to be drawn from the posteriors of the fit results.
    :param bool show_warn: Should warnings thrown during the fitting processes be shown.
    :param bool progress: Should fit progress bars be shown.
    """
    def __init__(self, temperature_profile: GasTemperature3D, temperature_model: Union[str, BaseModel1D],
                 density_profile: GasDensity3D, density_model: Union[str, BaseModel1D], radii: Quantity,
                 radii_err: Quantity, deg_radii: Quantity, fit_method: str = "mcmc", num_walkers: int = 20,
                 num_steps: [int, List[int]] = 20000, num_samples: int = 10000, show_warn: bool = True,
                 progress: bool = True):
        """
        The init method for the SpecificEntropy profile class, uses temperature and density profiles, along with
        models, to set up the entropy profile.

        A profile product which uses input GasTemperature3D and GasDensity3D profiles to generate a specific
        entropy profile. Functionally this is extremely similar to the HydrostaticMass profile class, as it
        calculates the y values itself, rather than them being part of the declaration.

        :param GasTemperature3D temperature_profile: The XGA 3D temperature profile to take temperature
            information from.
        :param str/BaseModel1D temperature_model: The model to fit to the temperature profile, either a name or an
            instance of an XGA temperature model class.
        :param GasDensity3D density_profile: The XGA 3D density profile to take density information from.
        :param str/BaseModel1D density_model: The model to fit to the density profile, either a name or an
            instance of an XGA density model class.
        :param Quantity radii: The radii at which to measure the entropy for the declaration of the profile.
        :param Quantity radii_err: The uncertainties on the radii.
        :param Quantity deg_radii: The radii values, but in units of degrees. This is required to set up a
            storage key for the profile to be filed in an XGA source.
        :param str fit_method: The name of the fit method to use for the fitting of the profiles, default is 'mcmc'.
        :param int num_walkers: If the fit method is 'mcmc' then this will set the number of walkers for the emcee
            sampler to set up.
        :param list/int num_steps: If the fit method is 'mcmc' this will set the number of steps for each sampler
            to take. If a single number is passed then that number of steps is used for both profiles, otherwise
            if a list is passed the first entry is used for the temperature fit, and the second for the
            density fit.
        :param int num_samples: The number of random samples to be drawn from the posteriors of the fit results.
        :param bool show_warn: Should warnings thrown during the fitting processes be shown.
        :param bool progress: Should fit progress bars be shown.
        """
        # This init is unfortunately almost identical to HydrostaticMass, there is a lot of duplicated code.

        # We check whether the temperature profile passed is actually the type of profile we need
        if type(temperature_profile) != GasTemperature3D:
            raise TypeError("Only a GasTemperature3D instance may be passed for temperature_profile, check "
                            "you haven't accidentally passed a ProjectedGasTemperature1D.")

        # We repeat this process with the density profile and model
        if type(density_profile) != GasDensity3D:
            raise TypeError("Only a GasDensity3D instance may be passed for density_profile, check you haven't "
                            "accidentally passed a GasDensity1D.")

        # We also need to check that someone hasn't done something dumb like pass profiles from two different
        #  clusters, so we'll compare source names.
        if temperature_profile.src_name != density_profile.src_name:
            raise ValueError("You have passed temperature and density profiles from two different "
                             "sources, any resulting entropy measurements would not be valid, so this is not "
                             "allowed.")
        # And check they were generated with the same central coordinate, otherwise they may not be valid. I
        #  considered only raising a warning, but I need a consistent central coordinate to pass to the super init
        elif np.any(temperature_profile.centre != density_profile.centre):
            raise ValueError("The temperature and density profiles do not have the same central coordinate.")
        # Same reasoning with the ObsID and instrument
        elif temperature_profile.obs_id != density_profile.obs_id:
            warn("The temperature and density profiles do not have the same associated ObsID.", stacklevel=2)
        elif temperature_profile.instrument != density_profile.instrument:
            warn("The temperature and density profiles do not have the same associated instrument.", stacklevel=2)

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
                warn("The temperature and density profile you passed were generated from different sets of annular"
                     " spectra, the entropy profile's associated set ident will be set to None.", stacklevel=2)
                set_id = None
                set_store = None
            else:
                set_id = temperature_profile.set_ident
                set_store = temperature_profile.associated_set_storage_key

        self._temp_prof = temperature_profile
        self._dens_prof = density_profile

        if not radii.unit.is_equivalent("kpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")
        else:
            radii = radii.to('kpc')
            radii_err = radii_err.to('kpc')
        # This will be overwritten by the super() init call, but it allows rad_check to work
        self._radii = radii

        # We won't REQUIRE that the profiles have data point generated at the same radii, as we're gonna
        #  measure entropy from the models, but I do need to check that the passed radii are within the radii of the
        #  and warn the user if they aren't
        self.rad_check(radii)

        if isinstance(num_steps, int):
            temp_steps = num_steps
            dens_steps = num_steps
        elif isinstance(num_steps, list) and len(num_steps) == 2:
            temp_steps = num_steps[0]
            dens_steps = num_steps[1]
        else:
            raise ValueError("If a list is passed for num_steps then it must have two entries, the first for the "
                             "temperature profile fit and the second for the density profile fit")

        # Make sure the model fits have been run, and retrieve the model objects
        temperature_model = temperature_profile.fit(temperature_model, fit_method, num_samples, temp_steps, num_walkers,
                                                    progress, show_warn)
        density_model = density_profile.fit(density_model, fit_method, num_samples, dens_steps, num_walkers, progress,
                                            show_warn)

        # Have to check whether the fits were actually successful, as the fit method will return a model instance
        #  either way
        if not temperature_model.success:
            raise XGAFitError("The fit to the temperature was unsuccessful, cannot define entropy profile.")
        if not density_model.success:
            raise XGAFitError("The fit to the density was unsuccessful, cannot define entropy profile.")

        self._temp_model = temperature_model
        self._dens_model = density_model

        ent, ent_dist = self.entropy(radii, conf_level=68)
        ent_vals = ent[0, :]
        ent_errs = np.mean(ent[1:, :], axis=0)

        super().__init__(radii, ent_vals, self._temp_prof.centre, self._temp_prof.src_name, self._temp_prof.obs_id,
                         self._temp_prof.instrument, radii_err, ent_errs, set_id, set_store, deg_radii)

        # Need a custom storage key for this entropy profile, incorporating all the information we have about what
        #  went into it, density profile, temperature profile, radii, density and temperature models - identical to
        #  the form used by HydrostaticMass profiles.
        dens_part = "dprof_{}".format(self._dens_prof.storage_key)
        temp_part = "tprof_{}".format(self._temp_prof.storage_key)
        cur_part = self.storage_key
        new_part = "tm{t}_dm{d}".format(t=self._temp_model.name, d=self._dens_model.name)
        whole_new = "{n}_{c}_{t}_{d}".format(n=new_part, c=cur_part, t=temp_part, d=dens_part)
        self._storage_key = whole_new

        # Setting the type
        self._prof_type = "specific_entropy"

        # This is what the y-axis is labelled as during plotting
        self._y_axis_name = r"K$_{\rm{X}}$"

        # Setting up a dictionary to store entropy results in.
        self._entropies = {}

    def entropy(self, radius: Quantity, conf_level: float = 68.2) -> Union[Quantity, Quantity]:
        """
        A method which will measure a specific entropy and specific entropy uncertainty within the given
        radius/radii.

        If the models for temperature and density have analytical solutions to their derivative wrt to radius then
        those will be used to calculate the gradients at radius, but if not then a numerical method will be used for
        which dx will be set to radius/1e+6.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            mass within.
        :param float conf_level: The confidence level for the entropy uncertainties, the default is 68.2% (~1σ).
        :return: An astropy quantity containing the entropy/entropies, lower and upper uncertainties, and another
            containing the mass realisation distribution.
        :rtype: Union[Quantity, Quantity]
        """
        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        # Prints a warning if the radius at which to calculate the entropy is outside the range of the data
        self.rad_check(radius)

        if radius.isscalar and radius in self._entropies:
            already_run = True
            ent_dist = self._entropies[radius]
        else:
            already_run = False

        if not already_run and self._dens_model.success and self._temp_model.success:
            # This grabs gas density values from the density model, need to check whether the model is in units
            #  of mass or number density
            if self._dens_model.y_unit.is_equivalent('1/cm^3'):
                dens = self._dens_model.get_realisations(radius)
            else:
                dens = self._dens_model.get_realisations(radius) / (MEAN_MOL_WEIGHT*m_p)

            # We do the same for the temperature vals, again need to check the units
            if self._temp_model.y_unit.is_equivalent("keV"):
                temp = self._temp_model.get_realisations(radius)
            else:
                temp = (self._temp_model.get_realisations(radius)*k_B).to('keV')

            ent_dist = (temp / dens**(2/3)).T

            if radius.isscalar:
                self._entropies[radius] = ent_dist

        elif not self._temp_model.success or not self._dens_model.success:
            raise XGAFitError("One or both of the fits to the temperature model and density profiles were "
                              "not successful")

        ent_med = np.percentile(ent_dist, 50, axis=0)
        ent_lower = ent_med - np.percentile(ent_dist, lower, axis=0)
        ent_upper = np.percentile(ent_dist, upper, axis=0) - ent_med

        ent_res = Quantity(np.array([ent_med.value, ent_lower.value, ent_upper.value]), ent_dist.unit)

        if np.any(ent_res[0] < 0):
            raise ValueError("A specific entropy of less than zero has been measured, which is not physical.")

        return ent_res, ent_dist

    def view_entropy_dist(self, radius: Quantity, conf_level: float = 68.2, figsize=(8, 8),
                          bins: Union[str, int] = 'auto', colour: str = "lightslategrey"):
        """
        A method which will generate a histogram of the entropy distribution that resulted from the entropy calculation
        at the supplied radius. If the entropy for the passed radius has already been measured it, and the entropy
        distribution, will be retrieved from the storage of this product rather than re-calculated.

        :param Quantity radius: An astropy quantity containing the radius/radii that you wish to calculate the
            entropy at.
        :param float conf_level: The confidence level for the entropy uncertainties, the default is 68.2% (~1σ).
        :param int/str bins: The argument to be passed to plt.hist, either a number of bins or a binning
            algorithm name.
        :param str colour: The desired colour of the histogram.
        :param tuple figsize: The desired size of the histogram figure.
        """
        if not radius.isscalar:
            raise ValueError("Unfortunately this method can only display a distribution for one radius, so "
                             "arrays of radii are not supported.")

        # Grabbing out the mass distribution, as well as the single result that describes the entropy distribution.
        ent, ent_dist = self.entropy(radius, conf_level)
        # Setting up the figure
        plt.figure(figsize=figsize)
        ax = plt.gca()
        # Includes nicer ticks
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        # And removing the yaxis tick labels as it's just a number of values per bin
        ax.yaxis.set_ticklabels([])

        # Plot the histogram and set up labels
        plt.hist(ent_dist.value, bins=bins, color=colour, alpha=0.7, density=False)
        plt.xlabel(self._y_axis_name + '[' + self.values_unit.to_string('latex') + ']')
        plt.title("Entropy Distribution at {}".format(radius.to_string()))

        vals_label = '$' + str(ent[0].round(2).value) + "^{+" + str(ent[2].round(2).value) + "}" + \
                     "_{-" + str(ent[1].round(2).value) + "}$"
        res_label = r"$K_{\rm{X}}$ = " + vals_label + '[' + self.values_unit.to_string('latex') + ']'

        # And this just plots the 'result' on the distribution as a series of vertical lines
        plt.axvline(ent[0].value, color='red', label=res_label)
        plt.axvline(ent[0].value-ent[1].value, color='red', linestyle='dashed')
        plt.axvline(ent[0].value+ent[2].value, color='red', linestyle='dashed')
        plt.legend(loc='best', prop={'size': 12})
        plt.tight_layout()
        plt.show()

    @property
    def temperature_profile(self) -> GasTemperature3D:
        """
        A method to provide access to the 3D temperature profile used to generate this entropy profile.

        :return: The input temperature profile.
        :rtype: GasTemperature3D
        """
        return self._temp_prof

    @property
    def density_profile(self) -> GasDensity3D:
        """
        A method to provide access to the 3D density profile used to generate this entropy profile.

        :return: The input density profile.
        :rtype: GasDensity3D
        """
        return self._dens_prof

    @property
    def temperature_model(self) -> BaseModel1D:
        """
        A method to provide access to the model that was fit to the temperature profile.

        :return: The fit temperature model.
        :rtype: BaseModel1D
        """
        return self._temp_model

    @property
    def density_model(self) -> BaseModel1D:
        """
        A method to provide access to the model that was fit to the density profile.

        :return: The fit density profile.
        :rtype: BaseModel1D
        """
        return self._dens_model

    def rad_check(self, rad: Quantity):
        """
        Very simple method that prints a warning if the radius is outside the range of data covered by the
        density or temperature profiles.

        :param Quantity rad: The radius to check.
        """
        if not rad.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("You can only check radii in units convertible to the radius units of "
                                      "the profile ({})".format(self.radii_unit.to_string()))

        if (self._temp_prof.annulus_bounds is not None and (rad > self._temp_prof.annulus_bounds[-1]).any()) \
                or (self._dens_prof.annulus_bounds is not None and (rad > self._dens_prof.annulus_bounds[-1]).any()):
            warn("Some radii are outside the data range covered by the temperature or density profiles, as such "
                 "you will be extrapolating based on the model fits.", stacklevel=2)


class Generic1D(BaseProfile1D):
    """
    A 1D profile product meant to hold profiles which have been dynamically generated by XSPEC profile fitting
    of models that I didn't build into XGA. It can also be used to make arbitrary profiles using external data.

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
    def __init__(self, radii: Quantity, values: Quantity, centre: Quantity, source_name: str, obs_id: str, inst: str,
                 y_axis_label: str, prof_type: str, radii_err: Quantity = None, values_err: Quantity = None,
                 associated_set_id: int = None, set_storage_key: str = None, deg_radii: Quantity = None):
        """
        The init of this subclass of BaseProfile1D, used by a dynamic XSPEC fitting process, or directly by a user,
        to set up an XGA profile with custom data.
        """

        super().__init__(radii, values, centre, source_name, obs_id, inst, radii_err, values_err, associated_set_id,
                         set_storage_key, deg_radii)
        self._prof_type = prof_type
        self._y_axis_name = y_axis_label








