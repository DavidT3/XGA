#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 22/01/2024, 00:05. Copyright (c) The Contributors
import re
from datetime import datetime
from typing import Union, List, Tuple
from warnings import warn

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.units import Quantity, Unit, UnitConversionError
from fitsio import FITS, FITSHDR, read_header
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from xga.exceptions import FailedProductError, IncompatibleProductError, NotAssociatedError, TelescopeNotAssociatedError
from xga.products import BaseProduct, BaseAggregateProduct
from xga.utils import dict_search, PRETTY_TELESCOPE_NAMES


class LightCurve(BaseProduct):
    """
    This is the XGA LightCurve product class, which is used to interface with X-ray lightcurves generated
    for a variety of sources. It provides simple access to data and information about the lightcurve, fitting
    capabilities, and the ability to easily create lightcurve visualisations.

    :param str path: The path to the lightcurve.
    :param str obs_id: The ObsID from which this lightcurve was generated.
    :param str instrument: The instrument from which this lightcurve.
    :param str stdout_str: The stdout from the generation process.
    :param str stderr_str: The stderr for the generation process.
    :param str gen_cmd: The generation command for the lightcurve.
    :param Quantity central_coord: The central coordinate of the region from which this lightcurve was extracted.
    :param Quantity inn_rad: The inner radius of the lightcurve region.
    :param Quantity out_rad: The outer radius of the lightcurve region.
    :param Quantity lo_en: The lower energy bound for this lightcurve.
    :param Quantity hi_en: The upper energy bound for this lightcurve.
    :param Quantity time_bin_size: The time bin size used to generate the lightcurve.
    :param str pattern_expr: The event selection pattern used to generate the lightcurve.
    :param bool region: Whether this was generated from a region in a region file
    :param bool is_back_sub: Whether this lightcurve is background subtracted or not.
    :param str telescope: The telescope that this product is derived from. Default is None.
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 central_coord: Quantity, inn_rad: Quantity, out_rad: Quantity, lo_en: Quantity, hi_en: Quantity,
                 time_bin_size: Quantity, pattern_expr: str, region: bool = False, is_back_sub: bool = True,
                 telescope: str = None):
        """
        This is the XGA LightCurve product class, which is used to interface with X-ray lightcurves generated
        for a variety of sources. It provides simple access to data and information about the lightcurve, fitting
        capabilities, and the ability to easily create lightcurve visualisations.

        :param str path: The path to the lightcurve.
        :param str obs_id: The ObsID from which this lightcurve was generated.
        :param str instrument: The instrument from which this lightcurve.
        :param str stdout_str: The stdout from the generation process.
        :param str stderr_str: The stderr for the generation process.
        :param str gen_cmd: The generation command for the lightcurve.
        :param Quantity central_coord: The central coordinate of the region from which this lightcurve was extracted.
        :param Quantity inn_rad: The inner radius of the lightcurve region.
        :param Quantity out_rad: The outer radius of the lightcurve region.
        :param Quantity lo_en: The lower energy bound for this lightcurve.
        :param Quantity hi_en: The upper energy bound for this lightcurve.
        :param Quantity time_bin_size: The time bin size used to generate the lightcurve.
        :param str pattern_expr: The event selection pattern used to generate the lightcurve.
        :param bool region: Whether this was generated from a region in a region file
        :param bool is_back_sub: Whether this lightcurve is background subtracted or not.
        :param str telescope: The telescope that this product is derived from. Default is None.
        """

        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, telescope=telescope)

        # Set the product type
        self._prod_type = "lightcurve"

        # Store the size of the time binning used to generate the lightcurve as an attribute
        self._time_bin = time_bin_size.to('s')

        # Unfortunate local import to avoid circular import errors
        from xga.generate.common import check_pattern
        self._pattern_expr, self._pattern_name = check_pattern(pattern_expr, telescope)

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

        lc_storage_name = "bound_{l}-{u}_ra{ra}_dec{dec}_ri{ri}_ro{ro}"
        if not self._region and self.inner_rad.isscalar:
            lc_storage_name = lc_storage_name.format(ra=self.central_coord[0].value, dec=self.central_coord[1].value,
                                                     ri=self._inner_rad.value, ro=self._outer_rad.value,
                                                     l=self.energy_bounds[0].to('keV').value,
                                                     u=self.energy_bounds[1].to('keV').value)
        elif not self._region and not self._inner_rad.isscalar:
            inn_rad_str = 'and'.join(self._inner_rad.value.astype(str))
            out_rad_str = 'and'.join(self._outer_rad.value.astype(str))
            lc_storage_name = lc_storage_name.format(ra=self.central_coord[0].value, dec=self.central_coord[1].value,
                                                     ri=inn_rad_str, ro=out_rad_str,
                                                     l=self.energy_bounds[0].to('keV').value,
                                                     u=self.energy_bounds[1].to('keV').value)
        else:
            lc_storage_name = "region"

        # Add the time bin information
        lc_storage_name += "_timebin{tb}_pattern{p}".format(tb=time_bin_size.value, p=self._pattern_name)

        # And we save the completed key to an attribute
        self._storage_key = lc_storage_name

        # The definition of this product has defined whether the data in 'RATE' are background subtracted or not, and
        #  that info is stored in this attribute - changes how we read in data
        self._is_back_sub = is_back_sub

        # Here we set up attributes to store the various information we can pull from a light curve file - they
        #  are all initially set to None because we only read the information into memory if the user actually
        #  calls one of the properties which uses one of these attributes
        # This will store the background subtracted count rate and uncertainty
        self._bck_sub_cnt_rate = None
        self._bck_sub_cnt_rate_err = None
        # This should store the values (in counts/per second) for the source, instrumental-effect-corrected, count-
        #  rate of the light curve
        self._src_cnt_rate = None
        # The count-rate should be accompanied by uncertainties, and they will live in this attribute
        self._src_cnt_rate_err = None
        # This stores an important quantity, the reference time for the light curve time values
        self._ref_time = None
        # This stores the 'time system' - so for default XMM lightcurves for instance it will be TT (terrestrial time)
        self._time_sys = None
        # Here we store the x-axis, the time steps which the count-rates are attributed to
        self._time = None
        # The fractional exposure time of the livetime for each data point
        self._frac_exp = None
        # The background count-rate and count-rate uncertainty, scaled to account for the area difference between
        #  the source and background regions
        self._bck_cnt_rate = None
        self._bck_cnt_rate_err = None

        # These attributes store 2D quantities that describe the start and end points of the good-time-intervals for
        #  the source and background light-curves. The zeroth column is start, and the other column is stop
        self._src_gti = None
        self._bck_gti = None

        # In these attributes we store the XMM recorded start and stop times for this light curve, as well as the
        #  place where the time values were assigned (i.e. relative to satellite or barycentre of the solar system)
        self._time_start = None
        self._time_stop = None
        self._time_assign = None

        # This just stores whether the data have been read into memory or not (set by the _read_on_demand method)
        self._read_in = False

        # This is used to store the header of the main data table, but again it is only read in its entirety if the
        #  user actually asks for it through the 'header' property
        self._header = None

    # Defining properties first
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

        :return: Astropy Quantity object containing the central coordinate in degrees.
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
    def outer_rad(self) -> Quantity:
        """
        Gives the outer radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this light curve was generated.

        :return: The outer radius(ii) of the region.
        :rtype: Quantity
        """
        return self._outer_rad

    @property
    def time_bin_size(self) -> Quantity:
        """
        Gives the time bin size used to generate the lightcurve.

        :return: The time bin size used to generate the lightcurve.
        :rtype: Quantity
        """
        return self._time_bin

    @property
    def count_rate(self) -> Quantity:
        """
        Returns the background-subtracted instrumental-effect-corrected count-rates for this light curve.

        :return: Background-subtracted instrumental-effect-corrected count-rates, in units of ct/s.
        :rtype: Quantity
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._bck_sub_cnt_rate

    @property
    def count_rate_err(self) -> Quantity:
        """
        Returns the background-subtracted instrumental-effect-corrected count-rate uncertainties for this light curve.

        :return: Background-subtracted instrumental-effect-corrected count-rate uncertainties, in units of ct/s.
        :rtype: Quantity
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._bck_sub_cnt_rate_err

    @property
    def src_count_rate(self) -> Quantity:
        """
        Returns the source instrumental-effect-corrected count-rates for this light curve.

        :return: Source instrumental-effect-corrected count-rates, in units of ct/s.
        :rtype: Quantity
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._src_cnt_rate

    @property
    def src_count_rate_err(self) -> Quantity:
        """
        Returns the source instrumental-effect-corrected count-rate uncertainties for this light curve.

        :return: Source instrumental-effect-corrected count-rate uncertainties, in units of ct/s.
        :rtype: Quantity
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._src_cnt_rate_err

    @property
    def ref_time(self) -> Time:
        """
        Returns the reference time for this lightcurve, which is what the 'time' values are calculated from.

        :return: An Astropy Time object that defines the reference time for this lightcurve.
        :rtype: Time
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._ref_time

    @property
    def time_system(self) -> str:
        """
        Returns the time system for this lightcurve; e.g. TT or terrestrial time.

        :return: The time system.
        :rtype: str
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._time_sys

    @property
    def time(self) -> Quantity:
        """
        Returns the time steps that correspond to the count-rates measured for this light curve

        :return: Background-subtracted and instrumental-effect-corrected count-rate uncertainties, in units of seconds.
        :rtype: Quantity
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._time

    @property
    def datetime(self) -> datetime:
        """
        Returns the time steps for this light curve, but in a datetime format, and no longer relative to a
        reference time.

        :return: The absolute datetimes that the time steps correspond to.
        :rtype: np.ndarray(datetime)
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return (self.ref_time + TimeDelta(self.time, format='sec', scale=self.time_system.lower())).to_datetime()

    @property
    def bck_count_rate(self) -> Quantity:
        """
        Returns the background count-rates for this light curve.

        :return: Background count-rates, in units of ct/s.
        :rtype: Quantity
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._bck_cnt_rate

    @property
    def bck_count_rate_err(self) -> Quantity:
        """
        Returns the background count-rate uncertainties for this light curve.

        :return: Background count-rate uncertainties, in units of ct/s.
        :rtype: Quantity
        """
        # The version of read on demand for this class will itself check to see if the data have already been
        #  read in, so we don't need to check that here - this method will read our LC data from disk into this
        #  XGA product instance
        self._read_on_demand()

        return self._bck_cnt_rate_err

    @property
    def frac_exp(self) -> Quantity:
        """
        The fractional exposure time for each entry in this light curve.

        :return: Fractional exposure.
        :rtype: Quantity
        """
        # Make sure the file is read in
        self._read_on_demand()
        return self._frac_exp

    @property
    def src_gti(self) -> Quantity:
        """
        Returns a 2D quantity with start (column 0) and end (column 1) times for the good-time-intervals of the
        source light curve.

        :return: A 2D astropy quantity with start (column 0) and end (column 1) times for the source
            good-time-intervals (in seconds).
        :rtype: Quantity
        """
        # Make sure the GTI information is read into memory
        self._read_on_demand()
        return self._src_gti

    @property
    def bck_gti(self) -> Quantity:
        """
        Returns a 2D quantity with start (column 0) and end (column 1) times for the good-time-intervals of the
        background light curve.

        :return: A 2D astropy quantity with start (column 0) and end (column 1) times for the background
            good-time-intervals (in seconds).
        :rtype: Quantity
        """
        # Make sure the GTI information is read into memory
        self._read_on_demand()
        return self._bck_gti

    @property
    def start_time(self) -> Quantity:
        """
        A property getter to access the recorded start time for this light curve.

        :return: Light curve start time, in seconds.
        :rtype: Quantity
        """
        self._read_on_demand()
        return self._time_start

    @property
    def stop_time(self) -> Quantity:
        """
        A property getter to access the recorded stop time for this light curve.

        :return: Light curve stop time, in seconds.
        :rtype: Quantity
        """
        self._read_on_demand()
        return self._time_stop

    @property
    def start_datetime(self) -> datetime:
        """
        A property getter to access the recorded start datetime for this light curve.

        :return: Light curve start datetime.
        :rtype: datetime
        """
        self._read_on_demand()

        return (self.ref_time + TimeDelta(self._time_start, format='sec', scale=self.time_system.lower())).to_datetime()

    @property
    def stop_datetime(self) -> datetime:
        """
        A property getter to access the recorded stop datetime for this light curve.

        :return: Light curve stop datetime.
        :rtype: datetime
        """
        self._read_on_demand()
        return (self.ref_time + TimeDelta(self._time_stop, format='sec', scale=self.time_system.lower())).to_datetime()

    @property
    def time_assign(self) -> str:
        """
        A property getter to access the physical location that the assigned times are based on.

        :return: The TASSIGN entry of the light curve file.
        :rtype: str
        """
        self._read_on_demand()
        return self._time_assign

    @property
    def header(self) -> FITSHDR:
        """
        Property getter allowing access to the main fits header of the light curve.

        :return: The header of the primary data table (RATE) of the light curve that was read in.
        :rtype: FITSHDR
        """
        if self._header is None and self.usable:
            self._header = read_header(self.path)
        elif not self.usable:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot access "
                                     "the header from it; reason given is {}.".format(reasons))

        return self._header

    # Then define internal methods
    def _read_on_demand(self):
        """
        This method is called by properties that deliver data to the user, either directly or via other methods of
        this class, such as view(). It will ensure that the data haven't already been read from the source file into
        memory, and that the source file has actually been classed as usable, and then read the relevant data into
        attributes of this class.
        """
        # TODO The way this is laid out is a bit weird

        # Usable flag to check that nothing went wrong in the light-curve generation, and the _read_in flag to
        #  check that we haven't already read this in to memory - no sense doing it again
        if self.usable and not self._read_in:
            with FITS(self.path) as all_lc:
                # This chunk reads out the various columns of the 'RATE' entry in the light curve file, storing
                #  them in suitably unit-ed astropy quantities
                if self._is_back_sub:
                    # TODO I should email the XMM help desk about this and double check
                    self._bck_sub_cnt_rate = Quantity(all_lc['RATE'].read_column('RATE'), 'ct/s')
                    if 'ERROR' in all_lc['RATE'].get_colnames():
                        self._bck_sub_cnt_rate_err = Quantity(all_lc['RATE'].read_column('ERROR'), 'ct/s')
                    else:
                        self._bck_sub_cnt_rate_err = Quantity(all_lc['RATE'].read_column('RATE_ERR'), 'ct/s')

                    if np.isnan(self._bck_sub_cnt_rate).any():
                        good_ent = np.where(~np.isnan(self._bck_sub_cnt_rate))
                        self._bck_sub_cnt_rate = self._bck_sub_cnt_rate[good_ent]
                        self._bck_sub_cnt_rate_err = self._bck_sub_cnt_rate_err[good_ent]
                    else:
                        good_ent = np.arange(0, len(self._bck_sub_cnt_rate))
                else:
                    # If we weren't told that the rate data are background subtracted when the light curve was
                    #  declared, then we store the values in the source count rate attributes
                    self._src_cnt_rate = Quantity(all_lc['RATE'].read_column('RATE'), 'ct/s')
                    if 'ERROR' in all_lc['RATE'].get_colnames():
                        self._src_cnt_rate_err = Quantity(all_lc['RATE'].read_column('ERROR'), 'ct/s')
                    else:
                        self._src_cnt_rate_err = Quantity(all_lc['RATE'].read_column('RATE_ERR'), 'ct/s')

                    if np.isnan(self._src_cnt_rate).any():
                        good_ent = np.where(~np.isnan(self._src_cnt_rate))
                        self._src_cnt_rate = self._src_cnt_rate[good_ent]
                        self._src_cnt_rate_err = self._src_cnt_rate_err[good_ent]
                    else:
                        good_ent = np.arange(0, len(self._src_cnt_rate))

                self._time = Quantity(all_lc['RATE'].read_column('TIME'), 's')[good_ent]
                self._frac_exp = Quantity(all_lc['RATE'].read_column('FRACEXP'))[good_ent]
                if "BACKV" in all_lc['RATE'].read_column('RATE'):
                    self._bck_cnt_rate = Quantity(all_lc['RATE'].read_column('BACKV'), 'ct/s')[good_ent]
                    self._bck_cnt_rate_err = Quantity(all_lc['RATE'].read_column('BACKE'), 'ct/s')[good_ent]

                else:
                    self._bck_cnt_rate = Quantity(np.full(len(all_lc['RATE'].read_column('TIME')),
                                                          np.NaN), 'ct/s')[good_ent]
                    self._bck_cnt_rate_err = Quantity(np.full(len(all_lc['RATE'].read_column('TIME')),
                                                              np.NaN), 'ct/s')[good_ent]

                # Grab the start, stop, and time assign values from the overall header of the light curve
                hdr = all_lc['RATE'].read_header()
                self._time_start = Quantity(hdr['TSTART'], 's')
                self._time_stop = Quantity(hdr['TSTOP'], 's')
                if 'TASSIGN' in hdr:
                    self._time_assign = hdr['TASSIGN']
                else:
                    self._time_assign = None
                self._ref_time = Time(hdr['MJDREF'], format='mjd')
                self._time_sys = hdr['TIMESYS']

                # TODO NEED TO ASK EROSITA TEAM IF THE COMBINED LIGHTCURVES ARE MEANT TO HAVE GTIs WRITTEN
                # I now read in the GTIs after dealing with the header keywords because I might need to construct
                #  my own GTI entry if there is no specific GTI in the lightcurve
                if self.telescope == 'xmm':
                    # Here we read out the beginning and end times of the GTIs for source and background
                    self._src_gti = Quantity([all_lc['SRC_GTIS'].read_column('START'),
                                              all_lc['SRC_GTIS'].read_column('STOP')], 's').T
                    self._bck_gti = Quantity([all_lc['BKG_GTIS'].read_column('START'),
                                              all_lc['BKG_GTIS'].read_column('STOP')], 's').T

                elif self.telescope == 'erosita' and 'SRCGTI{}'.format(self.instrument[-1]) in all_lc:
                    self._src_gti = Quantity([all_lc['SRCGTI{}'.format(self.instrument[-1])].read_column('START'),
                                              all_lc['SRCGTI{}'.format(self.instrument[-1])].read_column('STOP')],
                                             's').T
                    self._bck_gti = self._src_gti
                else:
                    self._src_gti = Quantity([[self._time_start.value, self._time_stop.value]], 's')
                    self._bck_gti = self._src_gti

            # TODO add calculation for error prop of src-bck or bck+bckcorr
            # And set this attribute to make sure that no further reading in is done
            self._read_in = True

        elif not self.usable:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("Failed to generate this product successfully, so you cannot access "
                                     "data from it; reason given is {}.".format(reasons))

    def overlap_check(self, lightcurves: Union['LightCurve', List['LightCurve']]) -> Union[np.ndarray, bool]:
        """
        A simple method which checks whether a passed LightCurve (or list of lightcurves) overlap temporally with
        this lightcurve.

        :param LightCurve/List[LightCurve] lightcurves: A LightCurve, or a list of LightCurves, to check for overlap
            with this LightCurve.
        :rtype: np.ndarray/bool
        :return: A boolean value (or an array of boolean values if multiple LightCurve instances were passed) which
            is True if the passed LightCurve temporally overlaps with this one, and False if it does not.
        """
        # Make sure that the input is iterable to normalise the behaviour later on
        if isinstance(lightcurves, LightCurve):
            lightcurves = [lightcurves]

        # Grabs the start and stop datetimes (which take into account the reference times of each lightcurve) for
        #  the passed lightcurves, puts them all in non-scalar quantities
        starts = np.array([lc.start_datetime for lc in lightcurves])
        ends = np.array([lc.stop_datetime for lc in lightcurves])

        # Simply constructs a boolean array that tells us if each lightcurve starts in, finishes in, or completely
        #  encloses the light curve we're checking against
        overlap = ((starts >= self.start_datetime) & (starts < self.stop_datetime)) | \
                  ((ends >= self.start_datetime) & (ends < self.stop_datetime)) | \
                  ((starts <= self.start_datetime) & (ends >= self.stop_datetime))

        if len(overlap) == 1:
            overlap = overlap[0]

        return overlap

    # Then define user-facing methods
    def get_view(self, ax: Axes, time_unit: Union[str, Unit] = Unit('s'), lo_time_lim: Quantity = None,
                 hi_time_lim: Quantity = None, colour: str = 'black', plot_sep: bool = False,
                 src_colour: str = 'tab:cyan', bck_colour: str = 'firebrick', custom_title: str = None,
                 label_font_size: int = 15, title_font_size: int = 18, highlight_bad_times: bool = True) -> Axes:
        """
        A method that allows the user to retrieve a populated lightcurve visualisation axes, in a form that allows
        them to then add their own plots in additon to what has been automatically constructed. This is an alternative
        to the view method, which calls this method and then displays the visualisation as constructed here.

        :param Axes ax: The matplotlib axes that should be populated with a lightcurve visualization.
        :param str/Unit time_unit: The unit to be used for the time axis.
        :param Quantity lo_time_lim: The lower x-limit (i.e. lower time limit) of the data to be displayed.
        :param Quantity hi_time_lim: The upper x-limit (i.e. upper time limit) of the data to be displayed.
        :param str colour: The colour to be used to plot data points (if background and source lightcurves are not
            plotted separately).
        :param bool plot_sep: Should the source and background lightcurves be plotted separately. Default is False.
        :param str src_colour: The colour to be used to plot source lightcurve data points, if plot_sep is True.
        :param str bck_colour: The colour to be used to plot background lightcurve data points, if plot_sep is True.
        :param str custom_title: A title to be added to the axes, which would override the automatically constructed
            figure title.
        :param int label_font_size: The fontsize to be used for labels.
        :param int title_font_size: The fontsize to be used for the title.
        :param bool highlight_bad_times: Should periods of time that are NOT within a GTI be highlighted?
            Default is True.
        :return: The input Axes, but populated with a lightcurve visualisation.
        :rtype: Axes
        """
        if isinstance(time_unit, str):
            time_unit = Unit(time_unit)

        if not self.time.unit.is_equivalent(time_unit):
            raise UnitConversionError("You have supplied a 'time_unit' that cannot be converted to seconds.")

        time_x = self.time.to(time_unit) - self.start_time.to(time_unit)

        if lo_time_lim is None:
            lo_time_lim = time_x.min()
        elif lo_time_lim is not None and lo_time_lim.unit.is_equivalent(time_unit):
            lo_time_lim = lo_time_lim.to(time_unit)

        if hi_time_lim is None:
            hi_time_lim = time_x.max()
        elif hi_time_lim is not None and hi_time_lim.unit.is_equivalent(time_unit):
            hi_time_lim = hi_time_lim.to(time_unit)

        if plot_sep:
            if self.src_count_rate is None:
                raise ValueError("This light-curve is background subtracted, so we cannot plot the total and "
                                 "background separately.")
            ax.errorbar(time_x.value, self.src_count_rate.value, yerr=self.src_count_rate_err.value, capsize=2,
                        color=src_colour, label='Source', fmt='x')
            ax.errorbar(time_x.value, self.bck_count_rate.value, yerr=self.bck_count_rate_err.value, capsize=2,
                        color=bck_colour, label='Background', fmt='x')
        else:
            ax.errorbar(time_x.value, self.count_rate.value, yerr=self.count_rate_err.value, capsize=2,
                        color=colour, label='Background subtracted', fmt='x')

        if highlight_bad_times and (len(self.src_gti) != 1 or self.src_gti[0, 0] != self.start_time
                                    or self.src_gti[0, 1] != self.stop_time):
            for ind in range(len(self.src_gti)):
                if ind == 0:
                    # This is where the first GTI DOESN'T begin with the start of the observation - so there is a
                    #  bad period between the start of the LC time coverage and the first GTI start
                    if (self.src_gti[ind, 0] - self.start_time).to('s') != 0:
                        bad_start = Quantity(0, 's')
                    else:
                        bad_start = self.src_gti[ind, 1] - self.start_time.to(time_unit)

                    # Here the first BAD time interval goes from the end of the first GTI to the start of the
                    #  next one, OR to the end of the obs if there isn't a next one
                    if len(self.src_gti) != 1:
                        bad_stop = self.src_gti[ind + 1, 0] - self.start_time.to(time_unit)
                    else:
                        bad_stop = self.stop_time - self.start_time

                    # bad_stop = self.src_gti[ind, 0] - self.start_time.to(time_unit)
                    label = "Bad time interval"
                elif ind == (len(self.src_gti)-1):
                    bad_start = self.src_gti[ind, 1]
                    bad_stop = self.stop_time - self.start_time
                    label = ""
                else:
                    bad_start = self.src_gti[ind, 1] - self.start_time.to(time_unit)
                    bad_stop = self.src_gti[ind+1, 0] - self.start_time.to(time_unit)
                    label = ""

                ax.axvspan(bad_start.value, bad_stop.value, color='firebrick', alpha=0.3, label=label)

        if custom_title is not None:
            ax.set_title(custom_title, fontsize=title_font_size)
        elif self.src_name is not None:
            ax.set_title("{s} {t} {o} {i} {l}-{u}keV Lightcurve".format(s=self.src_name, t='XMM', o=self.obs_id,
                                                                        i=self.instrument.upper(),
                                                                        l=self.energy_bounds[0].to('keV').value,
                                                                        u=self.energy_bounds[1].to('keV').value),
                         fontsize=title_font_size)
        else:
            ax.set_title("{t} {o} {i} {l}-{u}keV Lightcurve".format(s=self.src_name, t='XMM', o=self.obs_id,
                                                                    i=self.instrument.upper(),
                                                                    l=self.energy_bounds[0].to('keV').value,
                                                                    u=self.energy_bounds[1].to('keV').value),
                         fontsize=title_font_size)

        if lo_time_lim < time_x.min():
            warn('The lower time limit is smaller than the lowest time value, it has been set to the '
                 'lowest available value.', stacklevel=2)
            lo_time_lim = time_x.min()
        if hi_time_lim > time_x.max():
            warn('The upper time limit is higher than the greatest time value, it has been set to the '
                 'greatest available value.', stacklevel=2)
            hi_time_lim = time_x.max()

        ax.minorticks_on()
        ax.tick_params(direction='in', which='both', right=True, top=True)

        # Setting the axis limits
        ax.set_xlim(lo_time_lim.value, hi_time_lim.value)

        ax.set_xlabel("Relative Time [{}]".format(time_unit.to_string('latex')), fontsize=label_font_size)
        ax.set_ylabel("Count-rate [{}]".format(self.count_rate.unit.to_string('latex')), fontsize=label_font_size)
        ax.legend(loc='best')

        return ax

    def view(self, figsize: tuple = (14, 6), time_unit: Union[str, Unit] = Unit('s'),
             lo_time_lim: Quantity = None, hi_time_lim: Quantity = None, colour: str = 'black',
             plot_sep: bool = False, src_colour: str = 'tab:cyan', bck_colour: str = 'firebrick',
             custom_title: str = None, label_font_size: int = 15, title_font_size: int = 18,
             highlight_bad_times: bool = True):
        """
        A method that creates and displays a visualisation of this lightcurve.

        :param tuple figsize: The figure size to use for this lightcurve visualisation.
        :param str/Unit time_unit: The unit to be used for the time axis.
        :param Quantity lo_time_lim: The lower x-limit (i.e. lower time limit) of the data to be displayed.
        :param Quantity hi_time_lim: The upper x-limit (i.e. upper time limit) of the data to be displayed.
        :param str colour: The colour to be used to plot data points (if background and source lightcurves are not
            plotted separately).
        :param bool plot_sep: Should the source and background lightcurves be plotted separately. Default is False.
        :param str src_colour: The colour to be used to plot source lightcurve data points, if plot_sep is True.
        :param str bck_colour: The colour to be used to plot background lightcurve data points, if plot_sep is True.
        :param str custom_title: A title to be added to the axes, which would override the automatically constructed
            figure title.
        :param int label_font_size: The fontsize to be used for labels.
        :param int title_font_size: The fontsize to be used for the title.
        :param bool highlight_bad_times: Should periods of time that are NOT within a GTI be highlighted?
            Default is True.
        """
        # Create figure object
        fig = plt.figure(figsize=figsize)

        ax = plt.gca()

        ax = self.get_view(ax, time_unit, lo_time_lim, hi_time_lim, colour, plot_sep, src_colour, bck_colour,
                           custom_title, label_font_size, title_font_size, highlight_bad_times)
        plt.tight_layout()
        # Display the image
        plt.show()

        # Wipe the figure
        plt.close("all")


class AggregateLightCurve(BaseAggregateProduct):
    """
    The init method for the AggregateLightCurve class, performs checks and organises the light-curves which
    have been passed in, for easy retrieval. It also allows for analysis to be performed on the combined
    data, and for visualisations to be created.

    This class is designed to package light-curves generated for the same source, with the same settings, and
    for the same energy bounds - if interested in the time varying behaviours of multiple energy bands then
    the HardnessCurve and AggregateHardnessCurve products should be used. It can take light-curves from different
    instruments, and will deal with them simultaneously rather than stacking them.

    Light curves that are part of an AggregateLightCurve will be separated into 'time chunks', where a time chunk
    is a period that has uninterrupted coverage. For instance, three XMM observations separated by a year each would
    be in three different time chunks, but if there were a fourth observation that was taken by another telescope
    and happened concurrently (even if it didn't start and end at the same time) with the first XMM
    observation, then it would be in the same time chunk.

    :param Union[List[LightCurve], np.ndarray] lightcurves: A list or array of LightCurve objects that are to be
        collated in an AggregateLightCurve. These must be for the same source, and generated with the same
        settings.
    """
    def __init__(self, lightcurves: Union[List[LightCurve], np.ndarray]):
        """
        The init method for the AggregateLightCurve class, performs checks and organises the light-curves which
        have been passed in, for easy retrieval. It also allows for analysis to be performed on the combined
        data, and for visualisations to be created.

        This class is designed to package light-curves generated for the same source, with the same settings, and
        for the same energy bounds - if interested in the time varying behaviours of multiple energy bands then
        the HardnessCurve and AggregateHardnessCurve products should be used. It can take light-curves from different
        instruments, and will deal with them simultaneously rather than stacking them.

        Light curves that are part of an AggregateLightCurve will be separated into 'time chunks', where a time chunk
        is a period that has uninterrupted coverage. For instance, three XMM observations separated by a year each would
        be in three different time chunks, but if there were a fourth observation that was taken by another telescope
        and happened concurrently (even if it didn't start and end at the same time) with the first XMM
        observation, then it would be in the same time chunk.

        :param Union[List[LightCurve], np.ndarray] lightcurves: A list or array of LightCurve objects that are to be
            collated in an AggregateLightCurve. These must be for the same source, and generated with the same
            settings.
        """
        if isinstance(lightcurves, list):
            lightcurves = np.array(lightcurves)
        elif not isinstance(lightcurves, np.ndarray):
            raise TypeError("The AggregateLightCurve argument must be either a list of LightCurve objects, or a "
                            "numpy array of LightCurve Objects.")

        # Check that all the elements of the list/array are actually LightCurve objects
        if any([not isinstance(lc, LightCurve) for lc in lightcurves]):
            raise TypeError("You cannot pass a list/array containing an object that is not an instance of LightCurve.")

        # Obviously we need to make sure there are enough light curves to aggregate
        if len(lightcurves) < 2:
            raise ValueError("At least two light curves must be passed in order to declare an AggregateLightCurve.")

        # We need to do some checking on the input light-curves, to make sure they're for the same source, region, and
        #  energy bands - the first of the light curves is selected as the comparison point. We're going to check that
        #  each of them has the same central coordinates, region shape, inner radius, outer radius, and energy bounds
        comp_lc = lightcurves[0]
        if not all([np.array_equal(comp_lc.central_coord, lc.central_coord) for lc in lightcurves[1:]]):
            raise IncompatibleProductError("Central coordinates of all lightcurves passed to AggregateLightCurve must "
                                           "be the same.")
        elif not all([lc.shape == comp_lc.shape for lc in lightcurves[1:]]):
            raise IncompatibleProductError("Region shape of all lightcurves passed to AggregateLightCurve must "
                                           "be the same.")
        elif not all([np.array_equal(comp_lc.inner_rad, lc.inner_rad) for lc in lightcurves[1:]]):
            raise IncompatibleProductError("Inner radii of lightcurves passed to AggregateLightCurve must "
                                           "be the same.")
        elif not all([np.array_equal(comp_lc.outer_rad, lc.outer_rad) for lc in lightcurves[1:]]):
            raise IncompatibleProductError("Outer radii of lightcurves passed to AggregateLightCurve must "
                                           "be the same.")
        # Past me made the energy_bounds return a tuple for some reason - just need to convert to a Quantity to compare
        elif not all([np.array_equal(Quantity(comp_lc.energy_bounds), Quantity(lc.energy_bounds))
                      for lc in lightcurves[1:]]):
            raise IncompatibleProductError("The energy bounds of lightcurves passed to AggregateLightCurve must "
                                           "be the same - if interested in the time varying behaviours of multiple "
                                           "energy bands then the HardnessCurve and AggregateHardnessCurve products "
                                           "should be used.")
        # Checks that the time bin sizes used to generate the lightcurves are the same
        elif not all([lc.time_bin_size == comp_lc.time_bin_size for lc in lightcurves[1:]]):
            raise IncompatibleProductError("Time bin sizes of lightcurves passed to AggregateLightCurve must "
                                           "be the same.")
        elif len(list(set([lc.src_name for lc in lightcurves]))) > 1:
            raise ValueError("Some of the passed light curves were not generated for the same object; their "
                             "source names must match.")

        # Pulls out all the ObsIDs and instruments associated with these light curves
        obs_ids = list(set([lc.obs_id for lc in lightcurves]))
        insts = list(set([lc.instrument for lc in lightcurves]))

        if len(obs_ids) == 1:
            obs_id_to_pass = obs_ids[0]
        else:
            obs_id_to_pass = 'combined'

        if len(insts) == 1:
            inst_to_pass = insts[0]
        else:
            inst_to_pass = 'combined'

        # This just sorts the lightcurves by their start time, for earliest to latest
        start_sort = np.argsort(Quantity([lc.start_time for lc in lightcurves]))
        lightcurves = lightcurves[start_sort]

        super().__init__([lc.path for lc in lightcurves], 'lightcurve', obs_id_to_pass, inst_to_pass)

        # Storing the name of the object for which we've generated light curves - we've already checked that all the
        #  input light curves have the same src name.
        self._src_name = lightcurves[0].src_name

        # Make sure to add the energy bounds to the attribute - we require and know that they are the same for
        #  all the lightcurves that have been passed.
        self._energy_bounds = lightcurves[0].energy_bounds

        # This stores the ObsIDs and their instruments, for the lightcurves associated with this object
        self._rel_obs = {}
        # This array determines which lightcurves have overlapping temporal coverage
        overlapping = np.full((len(lightcurves), len(lightcurves)), False)
        for lc_ind, lc in enumerate(lightcurves):
            # If a telescope is already present in the rel_obs dict then we need to add the current ObsID as part
            #  of a new list, otherwise we need to make sure we add the current lightcurve ObsID and instrument, or
            #  just instrument if the ObsID is already there
            if lc.telescope not in self._rel_obs:
                self._rel_obs[lc.telescope] = {lc.obs_id: [lc.instrument]}
            elif lc.telescope in self._rel_obs and lc.obs_id not in self._rel_obs[lc.telescope]:
                self._rel_obs[lc.telescope][lc.obs_id] = [lc.instrument]
            elif lc.telescope in self._rel_obs and lc.obs_id in self._rel_obs[lc.telescope]:
                self._rel_obs[lc.telescope][lc.obs_id].append(lc.instrument)

            # Use the LightCurve overlap checking method to figure out which of our set of light curves overlaps with
            #  the current one. Checking like this does include the current light curve which obviously will overlap,
            #  but we're setting up an overlap matrix and those values will be the diagonal, so we don't mind
            cur_overlap = lc.overlap_check(lightcurves)
            overlapping[lc_ind, :] = cur_overlap

        # Get the entries one up from the diagonal, we'll use them to split the light curves up into time chunks, which
        #  is where there are gaps between coverage. This works because there will be False overlap entries in the
        #  shifted diagonal of the matrix, and those are cases where there is a break in the observations
        split = np.diag(overlapping, 1)
        split = np.insert(split, 0, False)

        # Here we want to index the light curves into time chunk groupings
        groupings = np.zeros(len(lightcurves), dtype=int)
        # This gives us the indices of where we need to split the light curves into chunks, as we know that the light
        #  curves are already sorted into the correct temporal order
        split_inds = np.argwhere(~split).T[0]
        # Just sets up the time chunk ID 1D matrix for our light curves
        for split_ind_ind, split_ind in enumerate(split_inds):
            if split_ind_ind != (len(split_inds)-1):
                groupings[split_ind: split_inds[split_ind_ind+1]] = split_ind_ind
            else:
                # As I want it to go to the very end of the array
                groupings[split_ind:] = split_ind_ind

        self._time_chunk_ids = np.arange(0, len(split_inds))

        self._component_products = {tel: {} for tel in self.telescopes}
        # Maybe there is a more elegant, in-line, way of doing this, but I cannot be bothered to think of it
        for lc_ind, lc in enumerate(lightcurves):
            rel_grp = groupings[lc_ind]
            if lc.obs_id not in self._component_products[lc.telescope]:
                self._component_products[lc.telescope][lc.obs_id] = {lc.instrument: {rel_grp: lc}}
            elif (lc.obs_id in self._component_products[lc.telescope] and
                  lc.instrument not in self._component_products[lc.telescope][lc.obs_id]):
                self._component_products[lc.telescope][lc.obs_id][lc.instrument] = {rel_grp: lc}
            elif (lc.obs_id in self._component_products[lc.telescope] and
                  lc.instrument in self._component_products[lc.telescope][lc.obs_id]):
                self._component_products[lc.telescope][lc.obs_id][lc.instrument][rel_grp] = lc

        # This is all helps to set the storage key as the same as the LightCurve, but we do account for different
        #  patterns accepted for different instruments - as mos1 and 2 should be treated the same we don't look at
        #  the specific MOS instrument, same with eROSITA telescope modules.
        self._patterns = {tel: {} for tel in self.telescopes}
        for lc in lightcurves:
            patt = lc.storage_key.split('_pattern')[-1]
            # Turns mos1 and mos2 into just mos, and tm1-7 into tm
            rel_inst = re.sub(r'\d+', '', lc.instrument)
            if rel_inst not in self._patterns[lc.telescope]:
                self._patterns[lc.telescope][rel_inst] = patt
            elif lc.instrument in self._patterns[lc.telescope] and self._patterns[lc.telescope][rel_inst] != patt:
                raise IncompatibleProductError(
                    "Lightcurves for the same instrument ({t}-{i}) must have the same event "
                    "selection pattern.".format(t=lc.telescope, i=rel_inst.upper()))

        patts = [tel + "_".join([pk + 'pattern' + pv for pk, pv in pd.items()]) for tel, pd in self._patterns.items()]
        # This is what the AggregateLightCurve will be stored under in an XGA source product storage structure.
        self._storage_key = lightcurves[0].storage_key.split('_pattern')[0] + '_' + '_'.join(patts)

    # Start by defining properties, then internal (protected) methods, and then user facing methods
    @property
    def obs_ids(self) -> dict:
        """
        A property of this spectrum set that details which ObsIDs of which telescopes have contributed lightcurves
        to this object.

        :return: A dictionary where the keys are telescope names and the values are lists of ObsIDs
        :rtype: dict
        """
        return {t: list(self._rel_obs[t].keys()) for t in self._rel_obs}

    @property
    def instruments(self) -> dict:
        """
        A property of this aggregate light curve that details which ObsIDs and instruments of which telescopes have
        contributed lightcurves to this object. The top level keys are telescopes, lower level keys are ObsIDs, and
        the values are lists of instruments.

        :return: A dictionary where the top level keys are telescopes, the lower level keys are ObsIDs, and
            their values are lists of related instruments.
        :rtype: dict
        """
        return self._rel_obs

    @property
    def telescopes(self) -> List[str]:
        """
        Property getter for telescopes that are associated with this aggregate light curve.

        :return: A list of telescope names with valid data related to this aggregate light curve.
        :rtype: List[str]
        """
        return list(self._rel_obs.keys())

    @property
    def src_name(self) -> str:
        """
        Method to return the name of the object a product is associated with. The product becomes
        aware of this once it is added to a source object.

        :return: The name of the source object this product is associated with.
        :rtype: str
        """
        return self._src_name

    # This needs a setter, as this property only becomes not-None when the product is added to a source object.
    @src_name.setter
    def src_name(self, name: str):
        """
        Property setter for the src_name attribute of a product, should only really be called by a source object,
        not by a user.

        :param str name: The name of the source object associated with this product.
        """
        self._src_name = name
        for p in self.all_lightcurves:
            p.src_name = name

    @property
    def all_lightcurves(self) -> List[LightCurve]:
        """
        Simple extra wrapper for get_lightcurve that allows the user to retrieve every single lightcurve associated
        with this AggregateLightCurve instance, for all time chunk IDs, telescopes, ObsIDs, and Instruments.

        :return: A list of every single lightcurve associated with this object.
        :rtype: List[LightCurve]
        """
        # This stores the flattened set of all lightcurves
        all_lcs = []
        # Iterate through the time chunks associated with this AggregateLightCurve, as we need to pass an ID to
        #  get_lightcurves to retrieve lightcurves
        for tc_i in self.time_chunk_ids:
            # If there is only one light curve per time chunk then get_lightcurves will just return a
            #  LightCurve object, but I always want it to be a list
            rel_lcs = self.get_lightcurves(tc_i)
            if isinstance(rel_lcs, LightCurve):
                rel_lcs = [rel_lcs]

            # Add the list of lightcurves for the current time chunk to the overall flattened lightcurve list
            all_lcs += rel_lcs

        return all_lcs

    @property
    def central_coord(self) -> Quantity:
        """
        This property provides the central coordinates (RA-Dec) of the region that this set of light curves
        was generated from.

        :return: Astropy Quantity object containing the central coordinate in degrees.
        :rtype: Quantity
        """
        return self.all_lightcurves[0].central_coord

    @property
    def shape(self) -> str:
        """
        Returns the shape of the outer edge of the region this set of light curves was generated from.

        :return: The shape (either circular or elliptical).
        :rtype: str
        """
        return self.all_lightcurves[0].shape

    @property
    def inner_rad(self) -> Quantity:
        """
        Gives the inner radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this set of light curves was generated.

        :return: The inner radius(ii) of the region.
        :rtype: Quantity
        """
        return self.all_lightcurves[0].inner_rad

    @property
    def outer_rad(self):
        """
        Gives the outer radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this set of light curves was generated.

        :return: The outer radius(ii) of the region.
        :rtype: Quantity
        """
        return self.all_lightcurves[0].outer_rad

    @property
    def time_bin_size(self) -> Quantity:
        """
        Gives the time bin size used to generate this set of light curves.

        :return: The time bin size used to generate this set of light curves.
        :rtype: Quantity
        """
        return self.all_lightcurves[0].time_bin_size

    @property
    def time_chunk_ids(self) -> np.ndarray:
        """
        Getter for the time chunk IDs associated with this AggregateLightCurve. Light curves that are part of an
        AggregateLightCurve will be separated into 'time chunks', where a time chunk is a period that has
        uninterrupted coverage. For instance, three XMM observations separated by a year each would be in three
        different time chunks, but if there were a fourth observation that was taken by another telescope and
        happened concurrently (even if it didn't start and end at the same time) with the first XMM observation, then
        it would be in the same time chunk.

        :return: np.ndarray
        :rtype: An array of integer time chunk identifiers, ordered from earlier to later times.
        """
        return self._time_chunk_ids

    @property
    def num_time_chunks(self) -> int:
        """
        Getter for the number of time chunks associated with this AggregateLightCurve. Light curves that are part
        of an AggregateLightCurve will be separated into 'time chunks', where a time chunk is a period that has
        uninterrupted coverage. For instance, three XMM observations separated by a year each would be in three
        different time chunks, but if there were a fourth observation that was taken by another telescope and
        happened concurrently (even if it didn't start and end at the same time) with the first XMM observation, then
        it would be in the same time chunk.

        :return: np.ndarray
        :rtype: An array of integer time chunk identifiers, ordered from earlier to later times.
        """
        return len(self._time_chunk_ids)

    @property
    def time_chunks(self) -> Quantity:
        """
        A getter for the start and stop times of the time chunks associated with this AggregateLightCurve. The left
        hand column are start times, and the right hand column are stop times. These are the earliest and latest
        times of coverage for all the observations in the particular time chunk.

        :return: A Nx2 non-scalar Astropy Quantity, where the left hand column are chunk start times, and the
            right hand column are chunk stop times.
        :rtype: Quantity
        """
        # Stores the chunk start and stop times until they are turned into a non-scalar Quantity at the end
        chunk_bounds = []
        # Iterates through the time chunks associated with this object
        for tc_id in self.time_chunk_ids:
            # Fetching lightcurves, making sure the return is a list
            rel_lcs = self.get_lightcurves(tc_id)
            if isinstance(rel_lcs, LightCurve):
                rel_lcs = [rel_lcs]
                # Finds the minimum start_time (i.e. the earliest start time) for lightcurves in this time chunk, and
                #  the maximum (i.e. latest) stop time - these define the bounds of this chunk
            tc_start = min(Quantity([lc.start_time for lc in rel_lcs]))
            tc_end = max(Quantity([lc.stop_time for lc in rel_lcs]))
            chunk_bounds.append(Quantity([tc_start, tc_end]))

        return Quantity(chunk_bounds)

    @property
    def datetime_chunks(self) -> np.ndarray:
        """
        A getter for the start and stop datetimes of the time chunks associated with this AggregateLightCurve. The
        left hand column are start datetimes, and the right hand column are stop datetimes. These are the earliest
        and latest times of coverage for all the observations in the particular time chunk.

        :return: A Nx2 array of datetime objects, where the left hand column are chunk start datetimes, and the
            right hand column are chunk stop datetimes.
        :rtype: np.ndarray(datetime)
        """
        # This behaves exactly the same as time_chunks, but works on datetime objects instead.
        chunk_bounds = []
        for tc_id in self.time_chunk_ids:
            rel_lcs = self.get_lightcurves(tc_id)
            if isinstance(rel_lcs, LightCurve):
                rel_lcs = [rel_lcs]
            tc_start = min([lc.start_datetime for lc in rel_lcs])
            tc_end = max([lc.stop_datetime for lc in rel_lcs])
            chunk_bounds.append([tc_start, tc_end])

        return np.array(chunk_bounds)

    @property
    def storage_key(self) -> str:
        """
        This property returns the storage key which this object assembles to place the AggregateLightCurve in
        an XGA source's storage structure. The key is based on the properties of the AggregateLightCurve, and
        some of the configuration options, and is basically human-readable.

        :return: String storage key.
        :rtype: str
        """
        return self._storage_key

    @property
    def event_selection_patterns(self) -> dict:
        """
        The event selection patterns used for different telescope instruments that are associated with
        this AggregateLightCurve.

        :return: A dictionary where top level keys are telescope names, lower level keys are instrument names, and
            values are event selection patterns.
        :rtype: dict
        """
        return self._patterns

    # Then define user-facing methods
    def get_lightcurves(self, time_chunk_id: int, obs_id: str = None,
                        inst: str = None, telescope: str = None) -> Union[List[LightCurve], LightCurve]:
        """
        This is the getter for the lightcurves stored in the AggregateLightCurve data storage structure. They can
        be retrieved based on ObsID and instrument.

        :param int time_chunk_id: The time chunk identifier to retrieve lightcurves for.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied.
        :param str inst: Optionally, a specific instrument to search for can be supplied.
        :param str telescope: Optionally, a specific telescope to search for can be supplied.
        :return: List of matching lightcurves, or just a LightCurve object if one match is found.
        :rtype: Union[List[LightCurve], LightCurve]
        """
        def unpack_list(to_unpack: list):
            """
            A recursive function to go through every layer of a nested list and flatten it all out. It
            doesn't return anything because to make life easier the 'results' are appended to a variable
            in the namespace above this one.

            :param list to_unpack: The list that needs unpacking.
            """
            # Must iterate through the given list
            for entry in to_unpack:
                # If the current element is not a list then all is chill, this element is ready for appending
                # to the final list
                if not isinstance(entry, list):
                    out.append(entry)
                else:
                    # If the current element IS a list, then obviously we still have more unpacking to do,
                    # so we call this function recursively.
                    unpack_list(entry)

        if time_chunk_id not in self.time_chunk_ids:
            tc_str = ", ".join(self.time_chunk_ids.astype(str))
            raise IndexError("{i} is not a time chunk ID associated with this AggregateLightCurve object. "
                             "Allowed time chunk IDs are; {a}".format(i=time_chunk_id, a=tc_str))
        elif telescope not in self.telescopes and telescope is not None:
            raise TelescopeNotAssociatedError("{0} is not associated with chunk {1} of this "
                                              "AggregateLightCurve.".format(telescope, time_chunk_id))
        elif ((telescope is not None and telescope in self.telescopes) and
              (obs_id not in self.obs_ids[telescope] and obs_id is not None)):
            raise NotAssociatedError("ObsID {o} is not associated with telescope {t} for chunk {tc} of this "
                                     "AggregateLightCurve.".format(o=obs_id, tc=time_chunk_id, t=telescope))
        elif ((telescope is not None and telescope in self.telescopes) and
              (obs_id is not None and obs_id in self.obs_ids) and
              (inst is not None and inst not in self.instruments[obs_id])):
            raise NotAssociatedError("Instrument {i} is not associated with {t}-{o} for time chunk {tc} of this "
                                     "AggregateLightCurve.".format(o=obs_id, i=inst, tc=time_chunk_id, t=telescope))

        matches = []
        for match in dict_search(time_chunk_id, self._component_products):
            out = []
            unpack_list(match)
            if ((telescope == out[0] or telescope is None) and (obs_id == out[1] or obs_id is None)
                    and (inst == out[2] or inst is None)):
                matches.append(out[-1])

        # Here I only return the object if one match was found
        if len(matches) == 1:
            matches = matches[0]
        elif len(matches) == 0:
            # This probably means that an instrument was requested without a specific ObsID and it doesn't exist for
            #  the specified time chunk
            raise NotAssociatedError("The requested data are not associated with time chunk {} of this "
                                     "AggregateLightCurve.".format(time_chunk_id))
        return matches

    def get_data(self, inst: str, date_time: bool = False) -> Tuple[Quantity, Quantity, Union[TimeDelta, np.ndarray]]:
        """
        A get method to retrieve all count-rate and timing data for a particular instrument from this
        AggregateLightCurve. The data are in the correct temporal order.

        :param str inst: The instrument for which to retrieve the overall count-rate and time data.
        :param bool date_time: Whether the time data should be returned as an array of datetimes (not the default), or
            an Astropy TimeDelta object with the time as a different from MJD 50814.0 in seconds (the default).
        :return: The count rate data, count rate uncertainty data, and time data for the selected instrument. These
            are in the correct temporal order.
        :rtype: Tuple[Quantity, Quantity, Union[TimeDelta, np.ndarray]]
        """
        # These store the countrates, errors, and times that we pull out for the chosen instrument for all
        #  time chunks
        cr_data = []
        cr_err_data = []
        t_data = []
        # Iterate through the time chunk IDs associated with this object
        for tc_id in self.time_chunk_ids:
            try:
                # Grab the light curves, but catch if there isn't an entry for the chosen instrument for this
                #  time chunk and handle it gracefully
                rel_lcs = self.get_lightcurves(tc_id, inst=inst)
            except NotAssociatedError:
                continue

            # Append the current time chunk's chosen instrument's count rate data and error to their lists
            cr_data.append(rel_lcs.count_rate)
            cr_err_data.append(rel_lcs.count_rate_err)

            # Do the same with the datetime
            cur_dt = rel_lcs.datetime
            t_data.append(cur_dt)

        # Combine the datetime arrays in t_data into one array
        t_data = np.concatenate(t_data)
        # If the user wants the time data as a TimeDelta from the reference MJD time then calculate that
        if not date_time:
            t_data = (Time(t_data) - Time(50814.0, format='mjd')).sec

        # Concatenate the count rate data and error into one quantity each and return everything
        return np.concatenate(cr_data), np.concatenate(cr_err_data), t_data

    def get_view(self, fig: Figure, inst: str = None, custom_title: str = None, label_font_size: int = 18,
                 title_font_size: int = 20, inst_cmap: str = 'viridis', y_lims: Quantity = None,
                 time_chunk_ids: Union[int, List[int]] = None) -> Tuple[dict, Figure]:
        """
        A get method for a populated visualisation of the light curves present in this AggregateLightCurve.

        :param Figure fig: The matplotlib Figure object to create axes on, and thus make the plot.
        :param str inst: A specific instrument to display data for. Default is None, in which case all instruments
            are plotted.
        :param str custom_title: A custom title to add to the visualisation - the default is None, in which case a
            title containing the source name and energy band will be generated.
        :param int label_font_size: The font size for axes labels, default is 18.
        :param int title_font_size: The font size for the title, default is 20.
        :param str inst_cmap: The colormap from which we draw colours to uniquely identify different instruments
            plotted in this get_view method.
        :param Quantity y_lims: The lower and upper limits that should be applied to the y-axis of this plot. The
            default is None, in which case they will be determined automatically based on the data.
        :param int/List[int] time_chunk_ids: This parameter can be used to control which time chunks are plotted on
            this AggregateLightCurve view. The default is None, in which case all time chunks are plotted; however
            the user may also pass a list of chunk IDs (or a single chunk ID) to limit the data that are shown.
        :return: A dictionary of axes objects that have been added, and the figure object that was passed in.
        :rtype: Tuple[dict, Figure]
        """

        # We check the input for the time_chunk_ids argument first, because not only does it determine the data that
        #  we plot, but it determines how we set up the figure
        if time_chunk_ids is not None and isinstance(time_chunk_ids, int):
            # The 'all_rel_lcs' variable is actually only used for auto-setting the y-axis limits, the light curve
            #  retrieval for plotting will happen separately (I just found it more convenient that way, even if this
            #  isn't particularly elegant).
            all_rel_lcs = self.get_lightcurves(time_chunk_ids, inst=inst)
            if isinstance(all_rel_lcs, LightCurve):
                all_rel_lcs = [all_rel_lcs]
            time_chunk_ids = np.array([time_chunk_ids])
        elif time_chunk_ids is not None and isinstance(time_chunk_ids, (list, np.ndarray)):
            # This has to be done in a for-loop rather than a list comprehension, because I have to be able to catch
            #  not associated errors
            # all_rel_lcs = [lc for tc_id in time_chunk_ids for lc in self.get_lightcurves(time_chunk_id=tc_id)]
            all_rel_lcs = []
            for tc_id in time_chunk_ids:
                try:
                    # Possible that this will return a single lightcurve, rather than a list of them
                    cur_lcs = self.get_lightcurves(time_chunk_id=tc_id, inst=inst)
                    # So we make sure that it IS a list
                    if isinstance(cur_lcs, LightCurve):
                        cur_lcs = [cur_lcs]
                    all_rel_lcs += cur_lcs
                except NotAssociatedError:
                    pass
            time_chunk_ids = np.array(time_chunk_ids)
        elif time_chunk_ids is not None:
            raise TypeError("Only integers and lists of integers may be passed for the 'time_chunk_ids' argument.")
        else:
            all_rel_lcs = self.all_lightcurves
            time_chunk_ids = self.time_chunk_ids

        # This sets the fraction of the total x-width of the figure that is set between each axes
        buffer_frac = 0.008
        # This calculates the total time length of all time chunks - note that we are selecting those specified by the
        #  time_chunk_ids array, which by default is all time chunks, but can be set by the user
        chunk_len = (self.datetime_chunks[:, 1] - self.datetime_chunks[:, 0])
        chunk_len = np.array([float(cl.total_seconds()) for cl in chunk_len])[time_chunk_ids]

        # Then finds what fraction of the total coverage each time chunk covers, taking into account the buffer (again
        #  this is for those time chunks specified by the time chunk id argument, with the default being all chunks)
        chunk_frac = chunk_len / (chunk_len.sum() + buffer_frac*len(chunk_len)-1)

        # Proportion of vertical to horizontal extent of the slanted line that breaks the separate axes
        break_slant = 1.3
        # Set properties of the break lines
        break_kwargs = dict(marker=[(-1, -break_slant), (1, break_slant)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        # To store the to-be-setup axes in
        axes_dict = {}
        # This is added to each iteration so that the next axes knows what x-position to start at
        cumu_x_pos = 0
        # Iterate through the selected time chunks, each will have a sub-axes
        for tc_id_ind, tc_id in enumerate(time_chunk_ids):
            # Grab the fraction of time coverage for this time chunk (e.g. size of this axes)
            rel_frac = chunk_frac[tc_id_ind]

            # We deal with axes differently depending on whether they are the first time chunk in a series, one in
            #  the middle of a series of time chunks, or the last. We also have to account for the fact that there
            #  could only be one or two time chunks. This all defines whether we plot break lines, which axes
            #  boundaries are visible, which axis tick labels are shown etc.
            if tc_id_ind == 0:
                # Adding a new axes, the size defined by the fractional coverage time, and the position by
                #  cumu_x_pos (though that should always be zero for tc_id == 0). It extends the full height of
                #  the figure
                axes_dict[tc_id_ind] = fig.add_axes([cumu_x_pos, 0.0, rel_frac, 1])
                # Set up a y-label - other axes won't have this because they share the y-axis and we only want to
                #  label the first one
                y_lab = "Count-rate [{}]".format(self.all_lightcurves[0].count_rate.unit.to_string('latex'))
                axes_dict[tc_id_ind].set_ylabel(y_lab, fontsize=label_font_size)

                # We set the upper and lower y-axis limits based on the maximum and minimum count rates across all
                #  the lightcurves that are to be plotted, as the y-axis is shared - if the user hasn't specified
                #  their own y-axis limits
                if y_lims is None:
                    low_lim = min([np.nanmin(lc.count_rate-lc.count_rate_err) for lc in all_rel_lcs]).value*0.95
                    upp_lim = max([np.nanmax(lc.count_rate+lc.count_rate_err) for lc in all_rel_lcs]).value*1.05
                else:
                    # The user has specified axis limits, so we make sure to convert them to the y-axis unit
                    low_lim, upp_lim = y_lims.to(self.all_lightcurves[0].count_rate.unit).value
                axes_dict[tc_id_ind].set_ylim(low_lim, upp_lim)

                # If there is more than one time chunk, we turn off the line on the right hand side of this initial
                #  axes - so there is no unsightly barrier between it and the next axes - we also add slanted lines
                #  to indicate a break in the y-axis
                if self.num_time_chunks != 1:
                    axes_dict[tc_id_ind].spines.right.set_visible(False)
                    axes_dict[tc_id_ind].plot([1, 1], [1, 0], transform=axes_dict[tc_id_ind].transAxes, **break_kwargs)

                # We make sure the ticks look how we want them
                axes_dict[tc_id_ind].tick_params(direction='in', which='both', right=False, left=True, top=True)

            # In this case we are at a time chunk that is not the first, and not the last
            elif tc_id_ind != (self.num_time_chunks - 1):
                # Add the axes at the correct position, making sure to share the y-axis with the first axes
                axes_dict[tc_id_ind] = fig.add_axes([cumu_x_pos, 0.0, rel_frac, 1], sharey=axes_dict[0])
                # Both the left hand and the right hand axis lines are turned off
                axes_dict[tc_id_ind].spines.left.set_visible(False)
                axes_dict[tc_id_ind].spines.right.set_visible(False)
                # We make sure to setup the ticks as we want them - making sure that the y-axis is not labelled for
                #  this one, and that the left and right ticks are turned off
                axes_dict[tc_id_ind].tick_params(direction='in', which='both', right=False, left=False, top=True,
                                                 labelleft=False)
                # This is what sets up the break lines on the left and right hand sides of the x-axis, at the top
                #  and bottom
                axes_dict[tc_id_ind].plot([1, 1], [1, 0], transform=axes_dict[tc_id_ind].transAxes, **break_kwargs)
                axes_dict[tc_id_ind].plot([0, 0], [0, 1], transform=axes_dict[tc_id_ind].transAxes, **break_kwargs)

            # Finally this is triggered when we're at the last time chunks
            else:
                axes_dict[tc_id_ind] = fig.add_axes([cumu_x_pos, 0.0, rel_frac, 1], sharey=axes_dict[0])
                # The right hand axis line is visible, but not the left
                axes_dict[tc_id_ind].spines.right.set_visible(True)
                axes_dict[tc_id_ind].spines.left.set_visible(False)
                # And we make sure to add the tick setup and ensure that the final break lines on the left are drawn
                axes_dict[tc_id_ind].tick_params(direction='in', which='both', right=True, left=False, top=True,
                                                 labelleft=False)
                axes_dict[tc_id_ind].plot([0, 0], [0, 1], transform=axes_dict[tc_id_ind].transAxes, **break_kwargs)

            # Iterate the cumulative position
            cumu_x_pos += (rel_frac+buffer_frac)
            # And turn on minor ticks, because I prefer how that looks
            axes_dict[tc_id_ind].minorticks_on()

            # Setting the x-axis limits, based on the known time coverage of the time chunk
            axes_dict[tc_id_ind].set_xlim(self.datetime_chunks[tc_id, 0], self.datetime_chunks[tc_id, 1])

        # Create a single x-axis label for all axes, it looks ugly and is unnecessary to have one for each
        fig.text(0.5, -0.04, "Time", ha='center', fontsize=label_font_size)

        # Fetching the colormap object specified by the user - this is what we draw colours from for each instrument
        #  involved in this AggregateLightCurve. We want them all to be different (of course), but we also want
        #  particular instruments to have consistent colouring across subplots (i.e. time chunks)
        rel_cmap = plt.get_cmap(inst_cmap)
        # This goes through all the instruments for all the ObsIDs for all the telescopes, constructing a list of
        #  unique instrument names
        uniq_insts = sorted(list(set([i for tel in self.instruments for oi in self.instruments[tel]
                                      for i in self.instruments[tel][oi]])))
        # Then we simply loop through the instruments, normalising their index in the list by the total number (we want
        #  to feed values between zero and one into the colormap), and get the colours out
        inst_colours = {inst: rel_cmap(inst_ind / (len(uniq_insts)-1)) for inst_ind, inst in enumerate(uniq_insts)}

        # Now we need to populate our carefully set up axes with DATA
        for tc_id_ind, tc_id in enumerate(time_chunk_ids):
            ax = axes_dict[tc_id_ind]
            try:
                # That this is the first part of the try-except, and won't trigger the except, is quite deliberate. It
                #  sets up the x-axis tick labels, even if there are no data for the requested instrument (if the
                #  user has even requested a specific instrument) - makes the plot look nice
                # The labels will have hour, minute, date, month (shortened word so as not to be ambigious with
                #  american date system), and year
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Hh-%Mm %d-%b-%Y'))
                for label in ax.get_xticklabels(which='major'):
                    label.set(y=label.get_position()[1] - 0.03, rotation=40, horizontalalignment='right')

                # This is what _might_ trigger the NotAssociatedError
                rel_lcs = self.get_lightcurves(tc_id, inst=inst)
            except NotAssociatedError:
                continue

            # Make sure rel_lcs, even if it only contains a single light curve
            if isinstance(rel_lcs, LightCurve):
                rel_lcs = [rel_lcs]

            # Now we cycle through the light curves for the current time chunk and add them to the plot
            for rel_lc in rel_lcs:
                ident = "{t} {o}-{i}".format(t=PRETTY_TELESCOPE_NAMES[rel_lc.telescope], o=rel_lc.obs_id,
                                             i=rel_lc.instrument)
                ax.errorbar(rel_lc.datetime, rel_lc.count_rate.value, yerr=rel_lc.count_rate_err.value,
                            capsize=2, label=ident, fmt='x', color=inst_colours[rel_lc.instrument])

            ax.legend(loc='best')

        # Check if the user has defined a custom title, and if not then we build one and add it to the plot
        if custom_title is not None:
            fig.suptitle(custom_title, fontsize=title_font_size, y=1.05)
        elif self.src_name is not None:
            fig.suptitle("{s} {l}-{u}keV Aggregate Lightcurve".format(s=self.src_name,
                                                                      l=self.energy_bounds[0].to('keV').value,
                                                                      u=self.energy_bounds[1].to('keV').value),
                         fontsize=title_font_size, y=1.05)
        else:
            fig.suptitle("{l}-{u}keV Aggregate Lightcurve".format(l=self.energy_bounds[0].to('keV').value,
                                                                  u=self.energy_bounds[1].to('keV').value),
                         fontsize=title_font_size, y=1.05)

        return axes_dict, fig

    def view(self, figsize: tuple = (14, 6), inst: str = None, custom_title: str = None, label_font_size: int = 15,
             title_font_size: int = 18, inst_cmap: str = 'viridis', y_lims: Quantity = None,
             time_chunk_ids: Union[int, List[int]] = None):
        """
        This method creates a combined visualisation of all the light curves associated with this object (apart from
        when you specify a single instrument, then it uses all the light curves from that instrument). The data are
        displayed in the correct temporal order, with the x-axis labels indicating the date and time rather than the
        mission specific internal time.

        :param tuple figsize: The size of the visualisation figure, default is (14, 6). Adjusting this value is the
            best way to achieve nice looking plots when labels are overlapping, particularly when there are many
            observations and time chunks to plot in the x-direction.
        :param str inst: A specific instrument to display data for. Default is None, in which case all instruments
            are plotted.
        :param str custom_title: A custom title to add to the visualisation - the default is None, in which case a
            title containing the source name and energy band will be generated.
        :param int label_font_size: The font size for axes labels, default is 18.
        :param int title_font_size: The font size for the title, default is 20.
        :param str inst_cmap: The colormap from which we draw colours to uniquely identify different instruments
            plotted in this view method.
        :param Quantity y_lims: The lower and upper limits that should be applied to the y-axis of this plot. The
            default is None, in which case they will be determined automatically based on the data.
        :param int/List[int] time_chunk_ids: This parameter can be used to control which time chunks are plotted on
            this AggregateLightCurve view. The default is None, in which case all time chunks are plotted; however
            the user may also pass a list of chunk IDs (or a single chunk ID) to limit the data that are shown.
        """
        # Create figure object
        fig = plt.figure(figsize=figsize)

        ax_dict, fig = self.get_view(fig, inst, custom_title, label_font_size, title_font_size, inst_cmap, y_lims,
                                     time_chunk_ids)

        # plt.tight_layout()
        # Display the plot
        plt.show()

        # Wipe the figure
        plt.close("all")
