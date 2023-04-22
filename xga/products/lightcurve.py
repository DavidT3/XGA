#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 22/04/2023, 15:19. Copyright (c) The Contributors

from astropy.units import Quantity

from xga.products import BaseProduct


class LightCurve(BaseProduct):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 lo_en: Quantity, hi_en: Quantity, time_bin_size: Quantity, pattern_expr: str):
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd)

        self._time_bin = time_bin_size
        self._pattern_expr = pattern_expr

        self._energy_bounds = (lo_en, hi_en)
