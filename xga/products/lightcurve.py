#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 08/11/2023, 12:38. Copyright (c) The Contributors
from datetime import datetime
from typing import Union, List
from warnings import warn

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.units import Quantity, Unit, UnitConversionError
from fitsio import FITS, FITSHDR, read_header
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from xga.exceptions import FailedProductError, IncompatibleProductError, NotAssociatedError
from xga.products import BaseProduct, BaseAggregateProduct
from xga.utils import dict_search


class LightCurve(BaseProduct):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 central_coord: Quantity, inn_rad: Quantity, out_rad: Quantity, lo_en: Quantity, hi_en: Quantity,
                 time_bin_size: Quantity, pattern_expr: str, region: bool = False, is_back_sub: bool = True):
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

        # Won't include energy bounds in this right now because I think the update_products method will do that
        #  part for us - bound_{l}-{u}keV l=lo_en.value, u=hi_en.value
        lc_storage_name += "_timebin{tb}_pattern{p}".format(tb=time_bin_size, p=self._pattern_name)

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
                    self._bck_sub_cnt_rate_err = Quantity(all_lc['RATE'].read_column('ERROR'), 'ct/s')

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
                    self._src_cnt_rate_err = Quantity(all_lc['RATE'].read_column('ERROR'), 'ct/s')

                    if np.isnan(self._src_cnt_rate).any():
                        good_ent = np.where(~np.isnan(self._src_cnt_rate))
                        self._src_cnt_rate = self._src_cnt_rate[good_ent]
                        self._src_cnt_rate_err = self._src_cnt_rate_err[good_ent]
                    else:
                        good_ent = np.arange(0, len(self._src_cnt_rate))

                self._time = Quantity(all_lc['RATE'].read_column('TIME'), 's')[good_ent]
                self._frac_exp = Quantity(all_lc['RATE'].read_column('FRACEXP'))[good_ent]
                self._bck_cnt_rate = Quantity(all_lc['RATE'].read_column('BACKV'), 'ct/s')[good_ent]
                self._bck_cnt_rate_err = Quantity(all_lc['RATE'].read_column('BACKE'), 'ct/s')[good_ent]

                # Here we read out the beginning and end times of the GTIs for source and background
                self._src_gti = Quantity([all_lc['SRC_GTIS'].read_column('START'),
                                          all_lc['SRC_GTIS'].read_column('STOP')], 's').T
                self._bck_gti = Quantity([all_lc['BKG_GTIS'].read_column('START'),
                                          all_lc['BKG_GTIS'].read_column('STOP')], 's').T

                # Grab the start, stop, and time assign values from the overall header of the light curve
                hdr = all_lc['RATE'].read_header()
                self._time_start = Quantity(hdr['TSTART'], 's')
                self._time_stop = Quantity(hdr['TSTOP'], 's')
                self._time_assign = hdr['TASSIGN']
                self._ref_time = Time(hdr['MJDREF'], format='mjd')
                self._time_sys = hdr['TIMESYS']

            # TODO add calculation for error prop of src-bck or bck+bckcorr
            # And set this attribute to make sure that no further reading in is done
            self._read_in = True

        elif not self.usable:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot access "
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

        starts = Quantity([lc.start_time for lc in lightcurves])
        ends = Quantity([lc.stop_time for lc in lightcurves])

        overlap = ((starts >= self.start_time) & (starts < self.stop_time)) | \
                  ((ends >= self.start_time) & (ends < self.stop_time)) | \
                  ((starts <= self.start_time) & (ends >= self.stop_time))

        if len(overlap) == 1:
            overlap = overlap[0]

        return overlap

    # Then define user-facing methods
    def get_view(self, ax: Axes, time_unit: Union[str, Unit] = Unit('s'), lo_time_lim: Quantity = None,
                 hi_time_lim: Quantity = None, colour: str = 'black', plot_sep: bool = False,
                 src_colour: str = 'tab:cyan', bck_colour: str = 'firebrick', custom_title: str = None,
                 label_font_size: int = 15, title_font_size: int = 18, highlight_bad_times: bool = True):

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

        if highlight_bad_times:
            for ind in range(len(self.src_gti)-2):
                if ind == 0 and (self.src_gti[ind, 0]-self.start_time).to('s') != 0:
                    bad_start = Quantity(0, time_unit)
                    bad_stop = self.src_gti[ind, 0] - self.start_time.to(time_unit)
                else:
                    bad_start = self.src_gti[ind, 1] - self.start_time.to(time_unit)
                    bad_stop = self.src_gti[ind+1, 0] - self.start_time.to(time_unit)

                ax.axvspan(bad_start.value, bad_stop.value, color='firebrick', alpha=0.3)
            ax.axvspan(self.src_gti[-2, 1].value - self.start_time.to(time_unit).value,
                       self.src_gti[-1, 0].value - self.start_time.to(time_unit).value, color='firebrick', alpha=0.3,
                       label='Bad time interval')

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

        # Make sure to add the energy bounds to the attribute - we require and know that they are the same for
        #  all the lightcurves that have been passed.
        self._energy_bounds = lightcurves[0].energy_bounds

        self._rel_obs = {}
        overlapping = np.full((len(lightcurves), len(lightcurves)), False)
        for lc_ind, lc in enumerate(lightcurves):

            if lc.obs_id not in self._rel_obs:
                self._rel_obs[lc.obs_id] = [lc.instrument]
            else:
                self._rel_obs[lc.obs_id].append(lc.instrument)

            cur_overlap = lc.overlap_check(lightcurves)
            overlapping[lc_ind, :] = cur_overlap

        split = np.diag(overlapping, 1)
        split = np.insert(split, 0, False)

        groupings = np.zeros(len(lightcurves), dtype=int)
        split_inds = np.argwhere(~split).T[0]
        for split_ind_ind, split_ind in enumerate(split_inds):
            if split_ind_ind != (len(split_inds)-1):
                groupings[split_ind: split_inds[split_ind_ind+1]] = split_ind_ind
            else:
                # As I want it to go to the very end of the array
                groupings[split_ind:] = split_ind_ind

        self._time_chunk_ids = np.arange(0, len(split_inds))

        # Maybe there is a more elegant, in-line, way of doing this, but I cannot be bothered to think of it
        for lc_ind, lc in enumerate(lightcurves):
            rel_grp = groupings[lc_ind]
            if lc.obs_id not in self._component_products:
                self._component_products[lc.obs_id] = {lc.instrument: {rel_grp: lc}}
            elif lc.obs_id in self._component_products and lc.instrument not in self._component_products[lc.obs_id]:
                self._component_products[lc.obs_id][lc.instrument] = {rel_grp: lc}
            elif lc.obs_id in self._component_products and lc.instrument in self._component_products[lc.obs_id]:
                self._component_products[lc.obs_id][lc.instrument][rel_grp] = lc

    @property
    def obs_ids(self) -> list:
        """
        A property of this spectrum set that details which ObsIDs have contributed lightcurves to this object.

        :return: A list of ObsIDs.
        :rtype: list
        """
        return list(self._rel_obs.keys())

    @property
    def instruments(self) -> dict:
        """
        A property of this spectrum set that details which ObsIDs and instruments have contributed lightcurves
        to this object. The top level keys are ObsIDs, and the values are lists of instruments.

        :return: A dictionary of lists, with the top level keys being ObsIDs, and the lists
            containing instruments associated with those ObsIDs.
        :rtype: dict
        """
        return self._rel_obs

    @property
    def all_lightcurves(self) -> List[LightCurve]:
        """
        Simple extra wrapper for get_lightcurve that allows the user to retrieve every single lightcurve associated
        with this AggregateLightCurve instance, for all time chunk IDs, ObsIDs, and Instruments.

        :return: A list of every single lightcurve associated with this object.
        :rtype: List[LightCurve]
        """
        all_lcs = []
        for tc_i in self.time_chunk_ids:
            # If there is only one light curve per time chunk then get_lightcurves will just return an object
            rel_lcs = self.get_lightcurves(tc_i)
            if isinstance(rel_lcs, LightCurve):
                rel_lcs = [rel_lcs]

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
        chunk_bounds = []
        for tc_id in self.time_chunk_ids:
            rel_lcs = self.get_lightcurves(tc_id)
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
        chunk_bounds = []
        for tc_id in self.time_chunk_ids:
            rel_lcs = self.get_lightcurves(tc_id)
            tc_start = min([lc.start_datetime for lc in rel_lcs])
            tc_end = max([lc.stop_datetime for lc in rel_lcs])
            chunk_bounds.append([tc_start, tc_end])

        return np.array(chunk_bounds)

    def get_lightcurves(self, time_chunk_id: int, obs_id: str = None,
                        inst: str = None) -> Union[List[LightCurve], LightCurve]:
        """
        This is the getter for the lightcurves stored in the AggregateLightCurve data storage structure. They can
        be retrieved based on ObsID and instrument.

        :param int time_chunk_id: The time chunk identifier to retrieve lightcurves for.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied.
        :param str inst: Optionally, a specific instrument to search for can be supplied.
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
        elif obs_id not in self.obs_ids and obs_id is not None:
            raise NotAssociatedError("{0} is not associated with this AggregateLightCurve.".format(obs_id))
        elif (obs_id is not None and obs_id in self.obs_ids) and \
                (inst is not None and inst not in self.instruments[obs_id]):
            raise NotAssociatedError("Instrument {1} is not associated with {0}".format(obs_id, inst))

        matches = []
        for match in dict_search(time_chunk_id, self._component_products):
            out = []
            unpack_list(match)
            if (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None):
                matches.append(out[-1])

        # Here I only return the object if one match was found
        if len(matches) == 1:
            matches = matches[0]
        return matches

    # def get_count_rate

# Then define user-facing methods
    def get_view(self, fig: Figure, time_unit: Union[str, Unit] = Unit('s'),
                 lo_time_lim: Quantity = None,
                 hi_time_lim: Quantity = None, colour: str = 'black', plot_sep: bool = False,
                 src_colour: str = 'tab:cyan', bck_colour: str = 'firebrick', custom_title: str = None,
                 label_font_size: int = 18, title_font_size: int = 18, highlight_bad_times: bool = True):

        # TODO this will need a little bit of TLC once this and the multi-mission branch cross paths
        if isinstance(time_unit, str):
            time_unit = Unit(time_unit)

        if not self.all_lightcurves[0].time.unit.is_equivalent(time_unit):
            raise UnitConversionError("You have supplied a 'time_unit' that cannot be converted to seconds.")

        buffer_frac = 0.008
        chunk_len = (self.time_chunks[:, 1] - self.time_chunks[:, 0]).value
        chunk_frac = chunk_len / (chunk_len.sum() + buffer_frac*len(chunk_len)-1)

        break_slant = 1.3  # proportion of vertical to horizontal extent of the slanted line
        break_kwargs = dict(marker=[(-1, -break_slant), (1, break_slant)], markersize=12,
                            linestyle="none", color='k', mec='k', mew=1, clip_on=False)

        axes_dict = {}
        cumu_x_pos = 0
        for tc_id in self.time_chunk_ids:
            rel_frac = chunk_frac[tc_id]

            if tc_id == 0:
                axes_dict[tc_id] = fig.add_axes([cumu_x_pos, 0.0, rel_frac, 1])
                y_lab = "Count-rate [{}]".format(self.all_lightcurves[0].count_rate.unit.to_string('latex'))
                axes_dict[tc_id].set_ylabel(y_lab, fontsize=label_font_size)
                axes_dict[tc_id].spines.right.set_visible(False)

                low_lim = min([(lc.count_rate-lc.count_rate_err).min() for lc in self.all_lightcurves]).value*0.95
                upp_lim = max([(lc.count_rate+lc.count_rate_err).max() for lc in self.all_lightcurves]).value*1.05
                axes_dict[tc_id].set_ylim(low_lim, upp_lim)

                if self.num_time_chunks != 1:
                    axes_dict[tc_id].spines.right.set_visible(False)
                    axes_dict[tc_id].plot([1, 1], [1, 0], transform=axes_dict[tc_id].transAxes, **break_kwargs)

            elif tc_id != (self.num_time_chunks - 1):
                axes_dict[tc_id] = fig.add_axes([cumu_x_pos, 0.0, rel_frac, 1], sharey=axes_dict[0])
                axes_dict[tc_id].spines.left.set_visible(False)
                axes_dict[tc_id].spines.right.set_visible(False)
                axes_dict[tc_id].tick_params(direction='in', which='both', right=False, left=False, top=True,
                                             labelleft=False)
                axes_dict[tc_id].plot([1, 1], [1, 0], transform=axes_dict[tc_id].transAxes, **break_kwargs)
                axes_dict[tc_id].plot([0, 0], [0, 1], transform=axes_dict[tc_id].transAxes, **break_kwargs)

            else:
                axes_dict[tc_id] = fig.add_axes([cumu_x_pos, 0.0, rel_frac, 1], sharey=axes_dict[0])
                axes_dict[tc_id].spines.right.set_visible(True)
                axes_dict[tc_id].spines.left.set_visible(False)
                axes_dict[tc_id].tick_params(direction='in', which='both', right=True, left=False, top=True,
                                             labelleft=False)
                axes_dict[tc_id].plot([0, 0], [0, 1], transform=axes_dict[tc_id].transAxes, **break_kwargs)

            cumu_x_pos += (rel_frac+buffer_frac)

            axes_dict[tc_id].minorticks_on()

            # Setting the axis limits
            axes_dict[tc_id].set_xlim(self.datetime_chunks[tc_id, 0], self.datetime_chunks[tc_id, 1])

            # axes_dict[tc_id].set_ylabel("Count-rate [{}]".format(self.all_lightcurves[0].count_rate.unit.to_string('latex')),
            #               fontsize=label_font_size)

        # axes_dict[0].yaxis.set_tick_params(which='both', labelleft=True)

        # fig.supxlabel("Time [{}]".format(time_unit.to_string('latex')), fontsize=label_font_size)
        fig.text(0.5, 0.01, "Time", ha='center', fontsize=label_font_size)

        for tc_id in self.time_chunk_ids:
            ax = axes_dict[tc_id]
            rel_lcs = self.get_lightcurves(tc_id)
            if isinstance(rel_lcs, LightCurve):
                rel_lcs = [rel_lcs]

            for rel_lc in rel_lcs:
                ident = "{t} {o}-{i}".format(t='XMM', o=rel_lc.obs_id, i=rel_lc.instrument)
                if not plot_sep:
                    ax.errorbar(rel_lc.datetime, rel_lc.count_rate.value, yerr=rel_lc.count_rate_err.value,
                                capsize=2, label=ident, fmt='x')

                else:
                    raise NotImplementedError("Not decided whether I will add this feature yet")

                    # ax.errorbar(time_x.value, self.src_count_rate.value, yerr=self.src_count_rate_err.value, capsize=2,
                    #             color=src_colour, label='Source', fmt='x')
                    # ax.errorbar(time_x.value, self.bck_count_rate.value, yerr=self.bck_count_rate_err.value, capsize=2,
                    #             color=bck_colour, label='Background', fmt='x')

                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Hh-%Mm %d-%b-%Y'))
                for label in ax.get_xticklabels(which='major'):
                    label.set(y=label.y*0.9, rotation=40, horizontalalignment='right')

                ax.legend(loc='best')

            # print(ax.get_yticklabels())

        plt.show()
        import sys
        sys.exit()
        # time_x = self.time.to(time_unit) - self.start_time.to(time_unit)

        # if lo_time_lim is None:
        #     lo_time_lim = time_x.min()
        # elif lo_time_lim is not None and lo_time_lim.unit.is_equivalent(time_unit):
        #     lo_time_lim = lo_time_lim.to(time_unit)
        #
        # if hi_time_lim is None:
        #     hi_time_lim = time_x.max()
        # elif hi_time_lim is not None and hi_time_lim.unit.is_equivalent(time_unit):
        #     hi_time_lim = hi_time_lim.to(time_unit)

        # if plot_sep:
        #     if self.src_count_rate is None:
        #         raise ValueError("This light-curve is background subtracted, so we cannot plot the total and "
        #                          "background separately.")
        #     ax.errorbar(time_x.value, self.src_count_rate.value, yerr=self.src_count_rate_err.value, capsize=2,
        #                 color=src_colour, label='Source', fmt='x')
        #     ax.errorbar(time_x.value, self.bck_count_rate.value, yerr=self.bck_count_rate_err.value, capsize=2,
        #                 color=bck_colour, label='Background', fmt='x')
        # else:
        #     ax.errorbar(time_x.value, self.count_rate.value, yerr=self.count_rate_err.value, capsize=2,
        #                 color=colour, label='Background subtracted', fmt='x')

        # if highlight_bad_times:
        #     for ind in range(len(self.src_gti)-2):
        #         if ind == 0 and (self.src_gti[ind, 0]-self.start_time).to('s') != 0:
        #             bad_start = Quantity(0, time_unit)
        #             bad_stop = self.src_gti[ind, 0] - self.start_time.to(time_unit)
        #         else:
        #             bad_start = self.src_gti[ind, 1] - self.start_time.to(time_unit)
        #             bad_stop = self.src_gti[ind+1, 0] - self.start_time.to(time_unit)
        #
        #         ax.axvspan(bad_start.value, bad_stop.value, color='firebrick', alpha=0.3)
        #     ax.axvspan(self.src_gti[-2, 1].value - self.start_time.to(time_unit).value,
        #                self.src_gti[-1, 0].value - self.start_time.to(time_unit).value, color='firebrick', alpha=0.3,
        #                label='Bad time interval')

        # if custom_title is not None:
        #     ax.set_title(custom_title, fontsize=title_font_size)
        # elif self.src_name is not None:
        #     ax.set_title("{s} {t} {o} {i} {l}-{u}keV Lightcurve".format(s=self.src_name, t='XMM', o=self.obs_id,
        #                                                                 i=self.instrument.upper(),
        #                                                                 l=self.energy_bounds[0].to('keV').value,
        #                                                                 u=self.energy_bounds[1].to('keV').value),
        #                  fontsize=title_font_size)
        # else:
        #     ax.set_title("{t} {o} {i} {l}-{u}keV Aggregate Lightcurve".format(s=self.src_name, t='XMM', o=self.obs_id,
        #                                                                       i=self.instrument.upper(),
        #                                                                       l=self.energy_bounds[0].to('keV').value,
        #                                                                       u=self.energy_bounds[1].to('keV').value),
        #                  fontsize=title_font_size)

        # if lo_time_lim < time_x.min():
        #     warn('The lower time limit is smaller than the lowest time value, it has been set to the '
        #          'lowest available value.', stacklevel=2)
        #     lo_time_lim = time_x.min()
        # if hi_time_lim > time_x.max():
        #     warn('The upper time limit is higher than the greatest time value, it has been set to the '
        #          'greatest available value.', stacklevel=2)
        #     hi_time_lim = time_x.max()

        # ax.minorticks_on()
        # ax.tick_params(direction='in', which='both', right=True, top=True)
        #
        # # Setting the axis limits
        # ax.set_xlim(self.time_chunks[0, 0].value, self.time_chunks[-1, 1].value)
        #
        # ax.set_xlabel("Time [{}]".format(time_unit.to_string('latex')), fontsize=label_font_size)
        # ax.set_ylabel("Count-rate [{}]".format(self.all_lightcurves[0].count_rate.unit.to_string('latex')),
        #               fontsize=label_font_size)
        # ax.legend(loc='best')

        return ax

    def view(self, figsize: tuple = (14, 6), time_unit: Union[str, Unit] = Unit('s'),
             lo_time_lim: Quantity = None, hi_time_lim: Quantity = None, colour: str = 'black',
             plot_sep: bool = False, src_colour: str = 'tab:cyan', bck_colour: str = 'firebrick',
             custom_title: str = None, label_font_size: int = 15, title_font_size: int = 18,
             highlight_bad_times: bool = True):

        # Create figure object
        fig = plt.figure(figsize=figsize)

        ax = self.get_view(fig, time_unit, lo_time_lim, hi_time_lim, colour, plot_sep, src_colour, bck_colour,
                           custom_title, label_font_size, title_font_size, highlight_bad_times)
        plt.tight_layout()
        # Display the image
        plt.show()

        # Wipe the figure
        plt.close("all")
