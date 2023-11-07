#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 07/11/2023, 09:37. Copyright (c) The Contributors
from typing import Union
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from astropy.units import Quantity, Unit, UnitConversionError
from fitsio import FITS, FITSHDR, read_header

from xga.exceptions import FailedProductError
from xga.products import BaseProduct


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

            # TODO add calculation for error prop of src-bck or bck+bckcorr
            # And set this attribute to make sure that no further reading in is done
            self._read_in = True

        elif not self.usable:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot access "
                                     "data from it; reason given is {}.".format(reasons))

    # Then define user-facing methods
    def view(self, figsize: tuple = (14, 6), time_unit: Union[str, Unit] = Unit('s'),
             lo_time_lim: Quantity = None, hi_time_lim: Quantity = None, colour: str = 'black',
             plot_sep: bool = False, src_colour: str = 'tab:cyan', bck_colour: str = 'firebrick',
             custom_title: str = None, label_font_size: int = 15, title_font_size: int = 18,
             highlight_bad_times: bool = True):

        # TODO set this up like view of image, make a get_view - just in case that might be useful at some point
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

        plt.figure(figsize=figsize)
        if plot_sep:
            if self.src_count_rate is None:
                raise ValueError("This light-curve is background subtracted, so we cannot plot the total and "
                                 "background separately.")
            plt.errorbar(time_x.value, self.src_count_rate.value, yerr=self.src_count_rate_err.value, capsize=2,
                         color=src_colour, label='Source', fmt='x')
            plt.errorbar(time_x.value, self.bck_count_rate.value, yerr=self.bck_count_rate_err.value, capsize=2,
                         color=bck_colour, label='Background', fmt='x')
        else:
            plt.errorbar(time_x.value, self.count_rate.value, yerr=self.count_rate_err.value, capsize=2,
                         color=colour, label='Background subtracted', fmt='x')

        if highlight_bad_times:
            for ind in range(len(self.src_gti)-2):
                if ind == 0 and (self.src_gti[ind, 0]-self.start_time).to('s') != 0:
                    bad_start = Quantity(0, time_unit)
                    bad_stop = self.src_gti[ind, 0] - self.start_time.to(time_unit)
                else:
                    bad_start = self.src_gti[ind, 1] - self.start_time.to(time_unit)
                    bad_stop = self.src_gti[ind+1, 0] - self.start_time.to(time_unit)

                plt.axvspan(bad_start.value, bad_stop.value, color='firebrick', alpha=0.3)
            plt.axvspan(self.src_gti[-2, 1].value - self.start_time.to(time_unit).value,
                        self.src_gti[-1, 0].value - self.start_time.to(time_unit).value, color='firebrick', alpha=0.3,
                        label='Bad time interval')

        if custom_title is not None:
            plt.title(custom_title, fontsize=title_font_size)
        else:
            pass

        if lo_time_lim < time_x.min():
            warn('The lower time limit is smaller than the lowest time value, it has been set to the '
                 'lowest available value.')
            lo_time_lim = time_x.min()
        if hi_time_lim > time_x.max():
            warn('The upper time limit is higher than the greatest time value, it has been set to the '
                 'greatest available value.')
            hi_time_lim = time_x.max()

        plt.minorticks_on()
        plt.tick_params(direction='in', which='both', right=True, top=True)

        # Setting the axis limits
        plt.xlim(lo_time_lim.value, hi_time_lim.value)

        plt.xlabel(" Relative Time [{}]".format(time_unit.to_string('latex')), fontsize=label_font_size)
        plt.ylabel("Count-rate [{}]".format(self.count_rate.unit.to_string('latex')), fontsize=label_font_size)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()
        plt.close('all')




# class AggregateLightCurve(BaseAggregateProduct)