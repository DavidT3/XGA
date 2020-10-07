#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 07/10/2020, 10:50. Copyright (c) David J Turner

from astropy.units import Quantity, UnitConversionError

from ..products.base import BaseProfile1D


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














