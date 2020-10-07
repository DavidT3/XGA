#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 07/10/2020, 17:06. Copyright (c) David J Turner
from typing import Tuple
from warnings import warn

import numpy as np
from astropy.units import Quantity, UnitConversionError
from scipy.integrate import trapz

from ..products.base import BaseProfile1D


# TODO DOCSTRING EVERYTHING FFS
class SurfaceBrightness1D(BaseProfile1D):
    def __init__(self, radii: Quantity, values: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None, background: Quantity = None):
        super().__init__(radii, values, source_name, obs_id, inst, radii_err, values_err)

        if type(background) != Quantity:
            raise TypeError("The background variables must be an astropy quantity.")

        # Set the internal type attribute to brightness profile
        self._prof_type = "brightness"

        # Check that the background passed by the user is the same unit as values
        if background is not None and background.unit == values.unit:
            self._background = background
        elif background is not None and background.unit != values.unit:
            raise UnitConversionError("The background unit must be the same as the values unit.")
        # If no background is passed then the internal background attribute stays at 0 as it was set in
        #  BaseProfile1D


class GasDensity1D(BaseProfile1D):
    def __init__(self, radii: Quantity, values: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None):
        super().__init__(radii, values, source_name, obs_id, inst, radii_err, values_err)

        # Actually imposing limits on what units are allowed for the radii and values for this - just
        #  to make things like the gas mass integration easier and more reliable. Also this is for mass
        #  density, not number density.
        if not radii.unit.is_equivalent("Mpc"):
            raise UnitConversionError("Radii unit cannot be converted to kpc")

        if not values.unit.is_equivalent("solMass / Mpc3"):
            raise UnitConversionError("Values unit cannot be converted to solMass / Mpc3")

        # These are the allowed realisation types (in addition to whatever density models there are
        self._allowed_real_types = ["inv_abel_model", "inv_abel_data"]

        self._prof_type = "gas_density"

    def add_realisation(self, real_type: str, radii: Quantity, realisation: Quantity, conf_level: int = 90):
        if real_type not in self._allowed_real_types:
            raise ValueError("{r} is not an acceptable realisation type, this profile object currently supports"
                             " the following; {a}".format(r=real_type, a=", ".join(self._allowed_real_types)))
        elif real_type in self._realisations:
            warn("There was already a realisation of this type stored in this profile, it has been overwritten.")

        if radii.shape[0] != realisation.shape[0]:
            raise ValueError("First axis of radii and realisation arrays must be the same length.")

        # Check that the radii units are alright
        if not radii.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("The supplied radii cannot be converted to the radius unit"
                                      " of this profile ({u})".format(u=self.radii_unit.to_string()))
        else:
            radii = radii.to(self.radii_unit)

        # Check that the realisation unit are alright
        if not realisation.unit.is_equivalent(self.values_unit):
            raise UnitConversionError("The supplied realisation cannot be converted to the values unit"
                                      " of this profile ({u})".format(u=self.values_unit.to_string()))
        else:
            realisation = realisation.to(self.values_unit)

        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        # Calculates the mean model value at each radius step
        model_mean = np.mean(realisation, axis=1)
        # Then calculates the values for the upper and lower limits (defined by the
        #  confidence level) for each radii
        model_lower = np.percentile(realisation, lower, axis=1)
        model_upper = np.percentile(realisation, upper, axis=1)

        self._realisations[real_type] = {"mod_real": realisation, "mod_radii": radii, "conf_level": conf_level,
                                         "mod_real_mean": model_mean, "mod_real_lower": model_lower,
                                         "mod_real_upper": model_upper}

    def gas_mass(self, real_type: str, outer_rad: Quantity, conf_level: int = 90) -> Tuple[Quantity]:
        if real_type not in self._realisations:
            raise ValueError("{r} is not an acceptable realisation type, this profile object currently has realisations"
                             " stored for".format(r=real_type, a=", ".join(list(self._realisations.keys()))))

        if not outer_rad.unit.is_equivalent(self.radii_unit):
            raise UnitConversionError("The supplied outer radius cannot be converted to the radius unit"
                                      " of this profile ({u})".format(u=self.radii_unit.to_string()))
        else:
            outer_rad = outer_rad.to(self.radii_unit)

        if real_type not in self._good_model_fits:
            real_info = self._realisations[real_type]

            allowed_ind = np.where(real_info["mod_radii"] <= outer_rad)[0]
            trunc_rad = real_info["mod_radii"][allowed_ind].to("Mpc")
            trunc_real = real_info["mod_real"].to("solMass / Mpc3")[allowed_ind, :] * trunc_rad[..., None]**2

            gas_masses = Quantity(4*np.pi*trapz(trunc_real.value.T, trunc_rad.value), "solMass")
        else:
            raise NotImplementedError("Cannot integrate models yet")

        upper = 50 + (conf_level / 2)
        lower = 50 - (conf_level / 2)

        gas_mass_mean = np.mean(gas_masses)
        gas_mass_lower = np.percentile(gas_masses, lower)
        gas_mass_upper = np.percentile(gas_masses, upper)

        return gas_mass_mean, gas_mass_mean - gas_mass_lower, gas_mass_upper - gas_mass_mean, gas_masses

    def gas_mass_profile(self):
        raise NotImplementedError("Haven't done this yet sozzle")


class GasMass1D(BaseProfile1D):
    def __init__(self, radii: Quantity, values: Quantity, source_name: str, obs_id: str, inst: str,
                 radii_err: Quantity = None, values_err: Quantity = None):
        super().__init__(radii, values, source_name, obs_id, inst, radii_err, values_err)
        self._prof_type = "gas_mass"











