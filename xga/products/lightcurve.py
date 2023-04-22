#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 22/04/2023, 17:33. Copyright (c) The Contributors

from astropy.units import Quantity

from xga.products import BaseProduct


class LightCurve(BaseProduct):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 central_coord: Quantity, inn_rad: Quantity, out_rad: Quantity, lo_en: Quantity, hi_en: Quantity,
                 time_bin_size: Quantity, pattern_expr: str, region: bool = False):
        # Unfortunate local import to avoid circular import errors
        from xga.sas import check_pattern

        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd)

        self._prod_type = "lightcurve"

        self._time_bin = time_bin_size
        self._pattern_expr, self._pattern_name = check_pattern(pattern_expr)

        self._energy_bounds = (lo_en, hi_en)

        # Storing the central coordinate of this spectrum
        self._central_coord = central_coord

        # Storing the region information
        self._inner_rad = inn_rad
        self._outer_rad = out_rad
        # And also the shape of the region
        if self._inner_rad.isscalar:
            self._shape = 'circular'
        else:
            self._shape = 'elliptical'

        # This describes whether this spectrum was generated directly from a region present in a region file
        self._region = region

        lc_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}"
        if not self._region and self.inner_rad.isscalar:
            lc_storage_name = lc_storage_name.format(ra=self.central_coord[0].value, dec=self.central_coord[1].value,
                                                     ri=self._inner_rad.value, ro=self._outer_rad.value)
        elif not self._region and not self._inner_rad.isscalar:
            inn_rad_str = 'and'.join(self._inner_rad.value.astype(str))
            out_rad_str = 'and'.join(self._outer_rad.value.astype(str))
            lc_storage_name = lc_storage_name.format(ra=self.central_coord[0].value, dec=self.central_coord[1].value,
                                                     ri=inn_rad_str, ro=out_rad_str)
        else:
            lc_storage_name = "region"

        # Won't include energy bounds in this right now because I think the update_products method will do that
        #  part for us - _{l}-{u}keV l=lo_en.value, u=hi_en.value
        lc_storage_name += "_timebin{tb}_pattern{p}".format(tb=time_bin_size, p=self._pattern_name)

        # And we save the completed key to an attribute
        self._storage_key = lc_storage_name

    @property
    def storage_key(self) -> str:
        """
        This property returns the storage key which this object assembles to place the LightCurve in
        an XGA source's storage structure. The key is based on the properties of the LightCurve, and
        some configuration options, and is basically human-readable.

        :return: String storage key.
        :rtype: str
        """
        return self._storage_key

    @property
    def central_coord(self) -> Quantity:
        """
        This property provides the central coordinates (RA-Dec) of the region that this light curve
        was generated from.

        :return: Astropy quantity object containing the central coordinate in degrees.
        :rtype: Quantity
        """
        return self._central_coord

    @property
    def shape(self) -> str:
        """
        Returns the shape of the outer edge of the region this light curve was generated from.

        :return: The shape (either circular or elliptical).
        :rtype: str
        """
        return self._shape

    @property
    def inner_rad(self) -> Quantity:
        """
        Gives the inner radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this light curve was generated.

        :return: The inner radius(ii) of the region.
        :rtype: Quantity
        """
        return self._inner_rad

    @property
    def outer_rad(self):
        """
        Gives the outer radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this light curve was generated.

        :return: The outer radius(ii) of the region.
        :rtype: Quantity
        """
        return self._outer_rad
