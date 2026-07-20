#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 7/20/26, 12:55 PM. Copyright (c) The Contributors.

import os.path
from typing import List, Tuple, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from astropy import wcs
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, HDUList
from astropy.table import Table
from astropy.units import Quantity, UnitConversionError
from astropy.wcs import WCS

from exceptions import XGADeveloperError
from xga import MISSION_COL_DB, DEFAULT_IMAGE_BINNING, ALT_INST_NAMES
from xga.exceptions import ProductGenerationError
from xga.products.base import BaseProduct
from xga.products.phot import Image

LIM_KEY_MAP = {'sky': ('xsiz', 'ysiz'),
               'det': ('detxsiz', 'detysiz'),
               'raw': ('rawxsiz', 'rawysiz')}

WCS_PREFIX_ALTS = {'TCDLT': 'CDELT',
                   'TCRPX': 'CRPIX',
                   'TCRVL': 'CRVAL',
                   'TCTYP': 'CTYPE'}


class EventList(BaseProduct):
    """
    A product class for event lists, it stores information about the event list.

    :param str path: The path to the event list file, OR an S3-bucket (or S3-bucket-like) path/url to stream
            the event list data from.
    :param str obs_id: The ObsID related to the event list being declared.
    :param str instrument: The instrument related to the event list being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the event list.
    :param str telescope: The telescope that is the source of this event list. The default is None.
    :param List[str] obs_ids: The obs ids that were combined to make this event list. The default is None.
    :param bool force_remote: Used to force the product instantiation to treat the passed path string as a url to
            a remote dataset, and to use fsspec to read/stream the data.
    :param dict fsspec_kwargs: Optional arguments that can be passed fsspec when reading or streaming remote
        datasets - e.g. to pass credentials to access an S3 bucket. Default value is None, which sets the
        argument to {"anon": True}, making it instantly compatible with NASA archive S3 buckets.
    :param Quantity energy_per_channel: An Astropy Quantity (units of channel/eV, or equivalent) that describes
        the mean energy difference between PI channels. The default is None, in which case the EventList
        instance uses a 'standard' value (not all missions/instruments will have a default value defined).
        Specifying a value by passing a Quantity will override any default value that may be available.
    :param str sky_x_col: The name of the column containing X-axis spatial coordinates for the sky coordinate
        system. The default is None, in which case XGA will attempt to determine it from the mission database.
    :param str sky_y_col: The name of the column containing Y-axis spatial coordinates for the sky coordinate
        system. The default is None, in which case XGA will attempt to determine it from the mission database.
    :param str det_x_col: The name of the column containing X-axis spatial coordinates for the detector coordinate
        system. The default is None, in which case XGA will attempt to determine it from the mission database.
    :param str det_y_col: The name of the column containing Y-axis spatial coordinates for the detector coordinate
        system. The default is None, in which case XGA will attempt to determine it from the mission database.
    :param str raw_x_col: The name of the column containing X-axis spatial coordinates for the raw coordinate
        system. The default is None, in which case XGA will attempt to determine it from the mission database.
    :param str raw_y_col: The name of the column containing Y-axis spatial coordinates for the raw coordinate
        system. The default is None, in which case XGA will attempt to determine it from the mission database.
    :param str en_col: The name of the column containing energy/channel information. The default is None, in which
        case XGA will attempt to determine the column name from the mission database.
    :param str evt_tab_name: The name of the FITS table containing the event data. The default is None, in which
        case XGA will attempt to determine the table name from the mission database.
    :param bool check_exists: Controls whether the product instantiation process checks for the file
        path's existence or not. Default is True, in which case a check will be performed. However, if declaring
        many products from the same directory/directory structure, it can be more performant to run listdir
        or scandir and confirm files exist externally, than one by one in each product declaration.
    """

    def __init__(self, path: str, obs_id: Optional[str] = None, instrument: Optional[str] = None,
                 stdout_str: Optional[str] = None, stderr_str: Optional[str] = None, gen_cmd: Optional[str] = None,
                 telescope: Optional[str] = None, obs_ids: Optional[List[str]] = None, force_remote: bool = False,
                 fsspec_kwargs: Optional[dict] = None, energy_per_channel: Optional[Quantity] = None,
                 sky_x_col: Optional[str] = None, sky_y_col: Optional[str] = None,
                 det_x_col: Optional[str] = None, det_y_col: Optional[str] = None,
                 raw_x_col: Optional[str] = None, raw_y_col: Optional[str] = None,
                 en_col: Optional[str] = None, evt_tab_name: Optional[str] = None,
                 check_exists: bool = True):
        """
        The init method of the EventList class, a product class for event lists, it stores information about
        the event list.

        :param str path: The path to the event list file, OR an S3-bucket (or S3-bucket-like) path/url to stream
            the event list data from.
        :param str obs_id: The ObsID related to the event list being declared.
        :param str instrument: The instrument related to the event list being declared.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the event list.
        :param str telescope: The telescope that is the source of this event list. The default is None.
        :param List[str] obs_ids: The obs ids that were combined to make this event list. The default is None.
        :param bool force_remote: Used to force the product instantiation to treat the passed path string as a url to
            a remote dataset, and to use fsspec to read/stream the data.
        :param dict fsspec_kwargs: Optional arguments that can be passed fsspec when reading or streaming remote
            datasets - e.g. to pass credentials to access an S3 bucket. Default value is None, which sets the
            argument to {"anon": True}, making it instantly compatible with NASA archive S3 buckets.
        :param Quantity energy_per_channel: An Astropy Quantity (units of channel/eV, or equivalent) that describes
            the mean energy difference between PI channels. The default is None, in which case the EventList
            instance uses a 'standard' value (not all missions/instruments will have a default value defined).
            Specifying a value by passing a Quantity will override any default value that may be available.
        :param str sky_x_col: The name of the column containing X-axis spatial coordinates for the sky coordinate
            system. The default is None, in which case XGA will attempt to determine it from the mission database.
        :param str sky_y_col: The name of the column containing Y-axis spatial coordinates for the sky coordinate
            system. The default is None, in which case XGA will attempt to determine it from the mission database.
        :param str det_x_col: The name of the column containing X-axis spatial coordinates for the detector coordinate
            system. The default is None, in which case XGA will attempt to determine it from the mission database.
        :param str det_y_col: The name of the column containing Y-axis spatial coordinates for the detector coordinate
            system. The default is None, in which case XGA will attempt to determine it from the mission database.
        :param str raw_x_col: The name of the column containing X-axis spatial coordinates for the raw coordinate
            system. The default is None, in which case XGA will attempt to determine it from the mission database.
        :param str raw_y_col: The name of the column containing Y-axis spatial coordinates for the raw coordinate
            system. The default is None, in which case XGA will attempt to determine it from the mission database.
        :param str en_col: The name of the column containing energy/channel information. The default is None, in which
            case XGA will attempt to determine the column name from the mission database.
        :param str evt_tab_name: The name of the FITS table containing the event data. The default is None, in which
            case XGA will attempt to determine the table name from the mission database.
        :param bool check_exists: Controls whether the product instantiation process checks for the file
            path's existence or not. Default is True, in which case a check will be performed. However, if declaring
            many products from the same directory/directory structure, it can be more performant to run listdir
            or scandir and confirm files exist externally, than one by one in each product declaration.
        """
        # Call the BaseProduct init, sets up some attributes
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, telescope=telescope,
                         force_remote=force_remote, fsspec_kwargs=fsspec_kwargs, check_exists=check_exists)
        self._prod_type = "events"
        # These store the header of the event list fits file (if read in), as well as the main table of event
        #  information (again if read in).
        self._header = None
        self._data = None
        # Also include another header attribute, specifically for the event table header
        self._event_header = None

        # These store the user-provided column and table names, as well as WCS key overrides
        self._sky_x_col = sky_x_col
        self._sky_y_col = sky_y_col
        self._det_x_col = det_x_col
        self._det_y_col = det_y_col
        self._raw_x_col = raw_x_col
        self._raw_y_col = raw_y_col
        self._en_col = en_col
        self._evt_tab_name = evt_tab_name

        # These attributes will store information about the currently loaded data, but also all the data that COULD
        #  be loaded. The idea being that we can tightly control which columns are being loaded and presented as
        #  pandas dataframes. Just converting the whole events table isn't guaranteed to work (some include
        #  columns that are arrays, which Pandas will not abide).
        # We do not believe we can stream a subset of columns, so the purpose of these features is to save on
        #  memory usage.
        # This will be a boolean flag, if True then only a subset of columns from the event list table has
        #  been loaded, if False then they all have.
        self._data_col_subset = None
        # Contains the names of ALL the columns in the events table that could be loaded
        self._all_col_names = None

        # We attempt to automatically derive the telescope, ObsID, and instrument (if they haven't been
        #  passed by the user) from the event list header
        if telescope is None:
            # Some older missions (like Einstein) can store the telescope name, ObsID, and instrument in the header
            #  of the event table, rather than the primary header. We try-except our attempt to read that
            #  information from the primary header, and failover to trying to read it from the event table header.
            try:
                self._tele = self.header['TELESCOP']

            # This contains a bodge, because we've come across a circular problem - we don't yet know the
            #  event table name, because we don't know the telescope name and can't look it up. That then means
            #  we have to assume the event table name is 'EVENTS' (which is a decent assumption), and attempt to
            #  read it in to get the telescope name.
            except KeyError:
                # This doesn't use the property setter because we are still in the init, and we want to
                #  temporarily set it to 'EVENTS' to try and find the telescope name
                self._evt_tab_name = 'EVENTS'

                self._tele = self.event_header['TELESCOP']

                # We now reset the event table name, and the event header attribute, so that the following code
                #  will continue as normal
                self._evt_tab_name = evt_tab_name
                self._event_header = None

        # We have to do the same for the instrument
        if instrument is None:
            try:
                self._inst = self.header['INSTRUME']
            except KeyError:
                # Same bodge as above
                self._evt_tab_name = 'EVENTS'

                self._inst = self.event_header['INSTRUME']

                # We now reset the event table name, and the event header attribute, so that the following code
                #  will continue as normal
                self._evt_tab_name = evt_tab_name
                self._event_header = None

        # Most missions call the table that contains event information "EVENTS", but it isn't a given - ROSAT, for
        #  instance, calls it STDEVT - obviously very important that we get this right
        if self.telescope.upper() not in MISSION_COL_DB:
            warn(f"The {self.telescope} telescope cannot be found in the XSELECT mission database file, so "
                 f"the name of the table containing event information is assumed to be 'EVENTS'.", stacklevel=2)
            self._evt_tab_name = "EVENTS"
        # In cases where individual instruments have entries for this, we'll use them
        elif (self._inst.upper() in MISSION_COL_DB[self.telescope.upper()] and
              'events' in MISSION_COL_DB[self.telescope.upper()][self._inst.upper()]):
            self._evt_tab_name = MISSION_COL_DB[self.telescope.upper()][self._inst.upper()]['events']
        # Otherwise we'll look for the top-level events entry for the mission
        elif 'events' in MISSION_COL_DB[self.telescope.upper()]:
            self._evt_tab_name = MISSION_COL_DB[self.telescope.upper()]['events']

        # And now we know we have the right event table name, we'll automatically determine the ObsID and instrument
        #  from the header, if they haven't been passed by the user.
        if obs_id is None:
            try:
                self._obs_id = self.header['OBS_ID']
            except KeyError:
                try:
                    self._obs_id = self.event_header['OBS_ID']
                # This is trying to catch a behaviour that might be unique to Einstein
                except KeyError:
                    self._obs_id = self.event_header['XS-OBSID']


        # Checking the formatting of the obs_ids argument
        if obs_ids is not None and (not isinstance(obs_ids, List) or
                                    (isinstance(obs_ids, List) and not all(isinstance(obs, str) for obs in obs_ids))):
            raise ValueError("The 'obs_ids' argument must be a list of strings.")
        self._obs_ids = obs_ids

        # The user may want to use WCSes to convert between different coordinate systems (sky to RA-Dec for
        #  instance), so when they are constructed they will be assigned to these attributes
        self._radec_sky_wcs = None

        # We allow the user to specify an energy per channel, which is used to convert between event channel and
        #  event energy, when performing operations such as generating images. If the user doesn't specify it
        #  then we'll try and infer the quantity using information from the mission database, but that operation
        #  will take place in the ev_per_channel property, as we don't want to raise an error because we can't find
        #  the necessary information unless the user actually needs that information.
        if energy_per_channel is not None and not energy_per_channel.unit.is_equivalent('eV/chan'):
            raise UnitConversionError("The 'energy_per_channel' argument must be an astropy Quantity with "
                                      "units convertible to eV/chan.")
        elif energy_per_channel is not None:
            self._ev_per_channel = energy_per_channel.to('eV/chan')
        else:
            self._ev_per_channel = None

    # ------------- Define properties -------------
    @property
    def obs_ids(self) -> list:
        """
        Property getter for the ObsIDs that are involved in this Eventlist, if combined. Otherwise
        will return a list with one element, the single relevant ObsID.

        :return: List of ObsIDs involved in this EventList.
        :rtype: list
        """

        return self._obs_ids

    # This absolutely doesn't get a setter considering it's the header object
    @property
    def header(self) -> fits.Header:
        """
        The primary header object of this event list.

        :return: The primary header of the event list.
        :rtype: fits.Header
        """
        # Reads the header into memory (though this method does check to see if it already exists)
        self._read_header_on_demand()
        return self._header

    @header.deleter
    def header(self):
        """
        Property deleter for the header of this EventList instance. The self._header attribute is removed from
        memory, and then self._header is explicitly set to None so that self._read_header_on_demand() will be
        triggered if you ever want the header from this object again.
        """
        del self._header
        self._header = None

    # This absolutely doesn't get a setter considering it's the header object
    @property
    def event_header(self) -> fits.Header:
        """
        The header object of the events table in this event list.

        :return: The event table header of the event list.
        :rtype: fits.Header
        """
        # This will read the header in if it does not already exist
        self._read_header_on_demand('event')
        return self._event_header

    @event_header.deleter
    def event_header(self):
        """
        Property deleter for the event table header of this EventList instance. The self._event_header attribute is
        removed from memory, and then self._event_header is explicitly set to None so that
        self._read_header_on_demand() will be triggered if you ever want the header from this object again.
        """
        del self._event_header
        self._event_header = None

    @property
    def data(self) -> pd.DataFrame:
        """
        Returns the primary events table included in this event list.

        :return: The contents of the primary data table of the event list.
        :rtype: pd.DataFrame
        """
        # If the header attribute is None then we know we have to read the header in
        if self._data is None:
            self._read_data_on_demand()
        return self._data

    @data.deleter
    def data(self):
        """
        Property deleter for the data of this EventList instance. The self._data attribute is removed from
        memory, and then self._data is explicitly set to None so that self._read_data_on_demand() will be
        triggered if you ever want the header from this object again.
        """
        del self._data
        self._data = None
        self._data_col_subset = None

    @property
    def radec_sky_wcs(self) -> wcs.WCS:
        """
        WCS information that relates this event list's 'sky' coordinate system to RA-Dec coordinates.

        :return: The WCS information that relates this event list's 'sky' coordinate system to RA-Dec coordinates.
        :rtype: astropy.wcs.WCS
        """
        # If we haven't already, we need to construct the WCS now
        if self._radec_sky_wcs is None:
            self._radec_sky_wcs = self._build_wcs(self.sky_x_col, self.sky_y_col, 'sky')
        return self._radec_sky_wcs

    @property
    def deg_per_sky(self) -> Quantity:
        """
        The angular size of a 'pixel' in the sky coordinate system, in both the x and y directions (though
        they are often the same).

        This information is extracted from the the Sky-RA/Dec WCS (accessible through the 'radec_sky_wcs'
        property of this EventList).

        :return: A two-entry non-scalar property, with the first entry being the x-direction sky pixel
            scale and the second being the y-direction sky pixel scale.
        :rtype: Quantity
        """

        return np.abs(Quantity(self.radec_sky_wcs.wcs.cdelt, 'deg/pix'))

    @property
    def sky_pix_lims(self) -> Tuple[Quantity, Quantity]:
        """
        The X and Y pixel limits of the sky coordinate system. As this information
        is extracted from the 'radec_sky_wcs' of this EventList instance, and not all
        event lists contain the necessary FITS header entries to determine coordinate
        system limits, a ValueError can be raised.

        :return: Two non-scalar quantities, with the first representing the lower and upper allowed values
            for the primary coordinate (usually sky) coordinate system x-axis, and the second being for the y-axis.
        :rtype: Tuple[Quantity, Quantity]
        """
        if self.radec_sky_wcs.pixel_bounds is None:
            raise ValueError("The Sky coordinate system pixel limits could not be automatically determined from the "
                             "FITS header of this EventList - instead, we suggest finding the minimum and maximum "
                             "values of the relevant X and Y columns.")

        return (Quantity(self.radec_sky_wcs.pixel_bounds[0], 'pix'),
                    Quantity(self.radec_sky_wcs.pixel_bounds[1], 'pix'))

    @property
    def ev_per_channel(self) -> Quantity:
        """
        The mapping between channel values in the energy column of the notebook, and an absolute energy
        value in eV. This is used in the construction of images and lightcurves from event lists.

        :param Quantity new_val: Passed to the ev_per_channel property setter, the new energy-channel
            mapping value in the form of an astropy quantity in units of eV/chan.
        :return: An astropy quantity, in units of eV/chan, representing the mapping between channel and energy.
        :rtype: Quantity
        """
        if self._ev_per_channel is None:
            # If the user didn't specify the energy-channel conversion, we'll try to derive it by identifying the
            #  fits table headers that define the limits of the channel coordinate system, then pulling that
            #  information from the event table header

            # Check whether the telescope has information in the mission file we maintain (derived from XSELECT's
            #  mission database file) - if it does then we'll use that to specify the header columns that contain
            #  the relevant WCS information.
            if self.telescope.upper() in MISSION_COL_DB:
                # Read out the mission database file's entry for the current mission
                rel_miss_info = MISSION_COL_DB[self.telescope.upper()]

                # Try to identify the TTYPE ind associated with the channel 'coordinate system'
                ecol_ttype_ind = [hdr_key.split('TTYPE')[-1] for hdr_key, hdr_val in self.event_header.items()
                                  if hdr_val == rel_miss_info['ecol'] and 'TTYPE' in hdr_key]
                # Check for multiple entries - if there are then something has gone awry
                if len(ecol_ttype_ind) > 1:
                    raise KeyError("Multiple TTYPE entries found for the energy column 'coordinate system' "
                                   "in the event table header.")
                else:
                    ecol_ttype_ind = ecol_ttype_ind[0]

                if ('phamax' in rel_miss_info and rel_miss_info['phamax'] != 'TLMAX' and
                        rel_miss_info['phamax'] in self.event_header):
                    max_ecol = self.event_header[rel_miss_info['phamax']]
                    # We have to assume that the minimum value is 0 in this case, as we don't know what
                    #  header name to look out for
                    min_ecol = 0
                elif 'phamax' in rel_miss_info and (rel_miss_info['phamax'] == 'TLMAX' or
                                                    rel_miss_info['phamax'] not in self.event_header):
                    max_ecol = self.event_header['TLMAX' + ecol_ttype_ind]
                    # If there is a matching TLMIN entry, we'll use that information to describe the minimum
                    #  value of the energy column
                    if 'TLMIN' + ecol_ttype_ind in rel_miss_info:
                        min_ecol = self.event_header['TLMIN' + ecol_ttype_ind]
                    else:
                        min_ecol = 0

                # TODO Will need to implement said default values.
                raise NotImplementedError("Default values for 'ev_per_channel' are not yet implemented, please pass "
                                          "a value to the 'ev_per_channel' argument when instantiating the EventList.")

        return self._ev_per_channel

    @ev_per_channel.setter
    def ev_per_channel(self, new_val: Quantity):
        """
        The mapping between channel values in the energy column of the notebook, and an absolute energy
        value in eV. This is used in the construction of images and lightcurves from event lists.

        :param Quantity new_val: Passed to the ev_per_channel property setter, the new energy-channel
            mapping value in the form of an astropy quantity in units of eV/chan.
        :return: An astropy quantity, in units of eV/chan, representing the mapping between channel and energy.
        :rtype: Quantity
        """
        # Validity checks on the input
        if not isinstance(new_val, Quantity):
            raise ValueError("The 'new_val' argument must be an Astropy quantity.")
        elif not new_val.unit.is_equivalent('eV/chan'):
            raise UnitConversionError("The 'new_val' argument must be convertible to units of 'eV/chan'.")

        # Converting to the expected units
        self._ev_per_channel = new_val.to('eV/chan')

    @property
    def sky_x_col(self) -> str:
        """
        The name of the event list column containing the Sky X coordinates.

        :return: The sky X column name.
        :rtype: str
        """
        if self._sky_x_col is None:
            if self.telescope.upper() in MISSION_COL_DB:
                self._sky_x_col = MISSION_COL_DB[self.telescope.upper()]['x']
            else:
                raise ValueError(f"The sky X column name cannot be determined for {self.telescope}, please provide "
                                 f"it manually using the 'sky_x_col' argument when instantiating the EventList, or by "
                                 f"setting this object's '.sky_x_col' property.")
        return self._sky_x_col

    @sky_x_col.setter
    def sky_x_col(self, value: str):
        self._sky_x_col = value
        # Clear the cached radec WCS as the columns have changed
        self._radec_sky_wcs = None

    @property
    def sky_y_col(self) -> str:
        """
        The name of the event list column containing the Sky Y coordinates.

        :return: The sky Y column name.
        :rtype: str
        """
        if self._sky_y_col is None:
            if self.telescope.upper() in MISSION_COL_DB:
                self._sky_y_col = MISSION_COL_DB[self.telescope.upper()]['y']
            else:
                raise ValueError(f"The sky Y column name cannot be determined for {self.telescope}, please provide "
                                 f"it manually using the 'sky_y_col' argument when instantiating the EventList, or by "
                                 f"setting this object's '.sky_y_col' property.")
        return self._sky_y_col

    @sky_y_col.setter
    def sky_y_col(self, value: str):
        self._sky_y_col = value
        # Clear the cached radec WCS as the columns have changed
        self._radec_sky_wcs = None

    @property
    def det_x_col(self) -> str:
        """
        The name of the event list column containing the Detector X coordinates.

        :return: The detector X column name.
        :rtype: str
        """
        if self._det_x_col is None:
            if (self.telescope.upper() in MISSION_COL_DB and
                    'detx' in MISSION_COL_DB[self.telescope.upper()]):
                self._det_x_col = MISSION_COL_DB[self.telescope.upper()]['detx']
            else:
                raise ValueError(f"The detector X column name cannot be determined for {self.telescope}, please provide "
                                 f"it manually using the 'det_x_col' argument when instantiating the EventList, or by "
                                 f"setting this object's '.det_x_col' property.")
        return self._det_x_col

    @det_x_col.setter
    def det_x_col(self, value: str):
        self._det_x_col = value

    @property
    def det_y_col(self) -> str:
        """
        The name of the event list column containing the Detector Y coordinates.

        :return: The detector Y column name.
        :rtype: str
        """
        if self._det_y_col is None:
            if (self.telescope.upper() in MISSION_COL_DB and
                    'dety' in MISSION_COL_DB[self.telescope.upper()]):
                self._det_y_col = MISSION_COL_DB[self.telescope.upper()]['dety']
            else:
                raise ValueError(f"The detector Y column name cannot be determined for {self.telescope}, please provide "
                                 f"it manually using the 'det_y_col' argument when instantiating the EventList, or by "
                                 f"setting this object's '.det_y_col' property.")
        return self._det_y_col

    @det_y_col.setter
    def det_y_col(self, value: str):
        self._det_y_col = value

    @property
    def raw_x_col(self) -> str:
        """
        The name of the event list column containing the Raw X coordinates.

        :return: The raw X column name.
        :rtype: str
        """
        if self._raw_x_col is None:
            if (self.telescope.upper() in MISSION_COL_DB and
                    'rawx' in MISSION_COL_DB[self.telescope.upper()]):
                self._raw_x_col = MISSION_COL_DB[self.telescope.upper()]['rawx']
            else:
                raise ValueError(f"The raw X column name cannot be determined for {self.telescope}, please provide "
                                 f"it manually using the 'raw_x_col' argument when instantiating the EventList, or by "
                                 f"setting this object's '.raw_x_col' property.")
        return self._raw_x_col

    @raw_x_col.setter
    def raw_x_col(self, value: str):
        self._raw_x_col = value

    @property
    def raw_y_col(self) -> str:
        """
        The name of the event list column containing the Raw Y coordinates.

        :return: The raw Y column name.
        :rtype: str
        """
        if self._raw_y_col is None:
            if (self.telescope.upper() in MISSION_COL_DB and
                    'rawy' in MISSION_COL_DB[self.telescope.upper()]):
                self._raw_y_col = MISSION_COL_DB[self.telescope.upper()]['rawy']
            else:
                raise ValueError(f"The raw Y column name cannot be determined for {self.telescope}, please provide "
                                 f"it manually using the 'raw_y_col' argument when instantiating the EventList, or by "
                                 f"setting this object's '.raw_y_col' property.")
        return self._raw_y_col

    @raw_y_col.setter
    def raw_y_col(self, value: str):
        self._raw_y_col = value

    @property
    def en_col(self) -> str:
        """
        The name of the event list column containing energy/channel information.

        :return: The energy column name.
        :rtype: str
        """
        if self._en_col is None:
            if self.telescope.upper() in MISSION_COL_DB:
                self._en_col = MISSION_COL_DB[self.telescope.upper()]['ecol']
            else:
                raise ValueError(f"The energy column name cannot be determined for {self.telescope}, please provide "
                                 f"it manually using the 'en_col' argument when instantiating the EventList, or by "
                                 f"setting this object's '.en_col' property.")
        return self._en_col

    @en_col.setter
    def en_col(self, value: str):
        self._en_col = value

    @property
    def evt_tab_name(self) -> str:
        """
        The name of the FITS table containing the event data.

        :return: The event table name.
        :rtype: str
        """
        if self._evt_tab_name is None:
            # If it's not manually set, we try to find it in the mission database
            if self.telescope.upper() in MISSION_COL_DB:
                rel_miss_info = MISSION_COL_DB[self.telescope.upper()]
                # In cases where individual instruments have entries for this, we'll use them
                if (self.instrument.upper() in rel_miss_info and
                        'events' in rel_miss_info[self.instrument.upper()]):
                    self._evt_tab_name = rel_miss_info[self.instrument.upper()]['events']
                # Otherwise we'll look for the top-level events entry for the mission
                elif 'events' in rel_miss_info:
                    self._evt_tab_name = rel_miss_info['events']
                else:
                    self._evt_tab_name = "EVENTS"
            else:
                self._evt_tab_name = "EVENTS"
        return self._evt_tab_name

    @evt_tab_name.setter
    def evt_tab_name(self, value: str):
        self._evt_tab_name = value

    # --------- Define internal functions ---------
    def _build_wcs(self, position_type: str = 'sky') -> wcs.WCS:
        """
        Internal factory method to construct a WCS object for arbitrary columns and position types.

        :param str position_type: The coordinate system type, 'sky', 'det', or 'raw'. Default is 'sky'.
        :return: A constructed Astropy WCS object.
        :rtype: wcs.WCS
        """
        # Check whether the telescope has information in the mission file we maintain
        has_db_info = self.telescope.upper() in MISSION_COL_DB

        # Determine which database keys to look for based on position type
        if position_type not in LIM_KEY_MAP:
            raise KeyError(f"Value of 'position_type' ({position_type}) is not valid. It must "
                           f"be one of {list(LIM_KEY_MAP.keys())}.")
        else:
            db_lim_keys = LIM_KEY_MAP[position_type]

        # We attempt to find the standard WCS header entries
        try:
            rel_wcs_keys = self.get_wcs_keys(position_type)

            # cdelt_keys = self._get_wcs_keys("TCDLT", x_col, y_col)
            # crpix_keys = self._get_wcs_keys("TCRPX", x_col, y_col)
            # crval_keys = self._get_wcs_keys("TCRVL", x_col, y_col)
            # ctype_keys = self._get_wcs_keys("TCTYP", x_col, y_col)

            # Time to assemble the WCS!
            out_wcs = WCS(naxis=2)
            out_wcs.wcs.cdelt = [self.event_header[cdelt_keys[0]], self.event_header[cdelt_keys[1]]]
            out_wcs.wcs.crpix = [self.event_header[crpix_keys[0]], self.event_header[crpix_keys[1]]]
            out_wcs.wcs.crval = [self.event_header[crval_keys[0]], self.event_header[crval_keys[1]]]
            out_wcs.wcs.ctype = [self.event_header[ctype_keys[0]], self.event_header[ctype_keys[1]]]

            max_sky_x, max_sky_y = None, None
            if has_db_info:
                rel_miss_info = MISSION_COL_DB[self.telescope.upper()]
                if db_lim_keys[0] in rel_miss_info and rel_miss_info[db_lim_keys[0]] in self.event_header:
                    max_sky_x = self.event_header[rel_miss_info[db_lim_keys[0]]]
                if db_lim_keys[1] in rel_miss_info and rel_miss_info[db_lim_keys[1]] in self.event_header:
                    max_sky_y = self.event_header[rel_miss_info[db_lim_keys[1]]]

            # If we still don't have limits, we try TLMAX
            if max_sky_x is None:
                x_ind = [hdr_key.split('TTYPE')[-1]
                         for hdr_key, hdr_val in self.event_header.items()
                         if hdr_val == x_col and 'TTYPE' in hdr_key]
                if len(x_ind) == 1:
                    max_sky_x = self.event_header.get('TLMAX' + x_ind[0], None)

            if max_sky_y is None:
                y_ind = [hdr_key.split('TTYPE')[-1]
                         for hdr_key, hdr_val in self.event_header.items()
                         if hdr_val == y_col and 'TTYPE' in hdr_key]
                if len(y_ind) == 1:
                    max_sky_y = self.event_header.get('TLMAX' + y_ind[0], None)

            # We only add pixel limits to the WCS if they are available for both
            #  X and Y axes - otherwise any limits will be imposed at analysis time
            #  (e.g. in generate_image, where the fallback is to find the maximum
            #  value of the X and Y columns
            if max_sky_x is not None and max_sky_y is not None:
                out_wcs.pixel_bounds = [(0, int(max_sky_x)), (0, int(max_sky_y))]

        except (KeyError, ValueError, TypeError) as err:
            raise ProductGenerationError(f"The requested WCS ({position_type}) cannot "
                                         f"be constructed for this event list: {err}")
            # Fallback to a simple Physical WCS if celestial mapping is missing
            # out_wcs = WCS(naxis=2)
            # out_wcs.wcs.crpix = [1, 1]
            # out_wcs.wcs.crval = [1, 1]
            # out_wcs.wcs.cdelt = [1, 1]
            # out_wcs.wcs.ctype = [x_col, y_col]
            # # Try to get basic limits
            # x_ind = [hdr_key.split('TTYPE')[-1] for hdr_key, hdr_val in self.event_header.items()
            #          if hdr_val == x_col and 'TTYPE' in hdr_key]
            # y_ind = [hdr_key.split('TTYPE')[-1] for hdr_key, hdr_val in self.event_header.items()
            #          if hdr_val == y_col and 'TTYPE' in hdr_key]
            # mx = self.event_header.get('TLMAX' + x_ind[0], 0) if len(x_ind) == 1 else 0
            # my = self.event_header.get('TLMAX' + y_ind[0], 0) if len(y_ind) == 1 else 0
            # out_wcs.pixel_bounds = [(0, int(mx)), (0, int(my))]

        return out_wcs

    def _read_header_on_demand(self, table: Optional[str] = None):
        """
        This will read the primary event list header into memory, without loading the data from the event
        list main table. That way the user can get access to the summary information stored in the header
        without wasting a lot of memory.

        :param Optional[str] table: Optionally defines which table's header to read; default is to read the primary header. Other
            options that may be passed are 'event', which loads and stores the event table header in _event_header.
        """
        if table is None or table == 'primary':
            read_type = 'primary'
            table = 0
        elif table == 'event':
            read_type = 'event'
            table = self._evt_tab_name
        else:
            raise ValueError("The 'table' argument must be either 'primary' or 'event'.")

        # We could likely treat the remote and local file access identically, but we're doing it this way for
        #  now out of an abundance of caution - I don't know how local files would behave using fsspec
        if (read_type == 'primary' and self._header is None) or (read_type == 'event' and self._event_header is None):
            # We alter the loading behaviours of astropy fits.open depending on whether this event list
            #  is pointed at a local file or not
            pass_use_fsspec = False if self._local_file else True
            pass_fsspec_kw = None if self._local_file else self.fsspec_kwargs
            try:
                # Reads only the header information
                with fits.open(self.path, lazy_load_hdus=True, use_fsspec=pass_use_fsspec,
                               fsspec_kwargs=pass_fsspec_kw) as fitso:

                    out_hdr = fitso[table].header
                    if read_type == 'primary':
                        self._header = out_hdr
                    elif read_type == 'event':
                        self._event_header = out_hdr

            except OSError:
                if self._local_file:
                    raise FileNotFoundError("{f} primary header cannot be opened. This product (of type {t}) is "
                                            "associated with {s}.".format(f=self.path, s=self.src_name, t=self.type))
                else:
                    raise FileNotFoundError("The remote file's ({f}) primary header cannot be opened. This "
                                            "product (of type {t}) is associated "
                                            "with {s}.".format(f=self.path, s=self.src_name, t=self.type))

    def _read_data_on_demand(self, columns: Optional[List[str]] = None):
        """
        This will read the event list table into memory, allowing for the loading of a specific subset of columns, as
        well as streaming data from remote files.
        """
        # This is rather inelegant, but if we already have the whole set of column names saved in an attribute (which
        #  happens down below the first time the events table is accessed in any way); we'll check here if the
        #  columns passed by the user are actually in the table. If we don't have that info this same check is
        #  performed after a read of the events HDU
        if self._all_col_names is not None and columns is not None:
            # If there are any passed columns which aren't in the event list columns, we'll find them here and raise
            #  an exception (usefully telling the user which columns are bad and which columns they have to choose
            #  from).
            bad_cols = [cc for cc in columns if cc not in self._all_col_names]
            if len(bad_cols) > 0:
                raise ValueError("The following column(s) are not available in this event "
                                 "list; {c}. Please choose from; {a}.".format(c=",".join(bad_cols),
                                                                              a=",".join(self._all_col_names)))

        # In this case some data have already been loaded, but only a subset of columns, and a different subset
        #  to what is being requested via the 'columns' argument now
        if (self._data_col_subset is not None and self._data_col_subset and
                columns is not None and set(list(self._data.colnames)) != set(columns)):
            if all([cc in self._data.colnames for cc in columns]):
                run_load = False
            else:
                # We'll update the columns argument so that the already loaded columns are loaded again - this is
                #  a cumulative loading process in that regard
                columns = list(set(columns + list(self._data.colnames)))
                # Do we need to load anything
                run_load = True
                data_col_subset = True
        # Here we have already loaded the whole event list table, and we aren't going to take any
        #  columns away, even though only a subset has been requested this time, so we don't do anything
        elif self._data_col_subset is not None and not self._data_col_subset and columns is not None:
            run_load = False
            data_col_subset = False
            pass
        # No data have been loaded yet, and we're loading a subset of columns
        elif self._data_col_subset is None and columns is not None:
            data_col_subset = True
            run_load = True
        # No data have been loaded yet, and we're loading the whole dataset
        elif self._data_col_subset is None and columns is None:
            data_col_subset = False
            run_load = True
        else:
            run_load = False
            # raise XGADeveloperError("No user should see this, contact an XGA developer.")

        # Now we try to load the requested data into this EventList instance (into memory) if necessary
        if run_load:
            try:
                # We alter the loading behaviours of astropy fits.open depending on whether this event list
                #  is pointed at a local file or not
                pass_use_fsspec = False if self._local_file else True
                pass_fsspec_kw = None if self._local_file else self.fsspec_kwargs

                # Opening the event list fits file - we'll only grab the events data though
                with fits.open(self.path, lazy_load_hdus=True, use_fsspec=pass_use_fsspec,
                               fsspec_kwargs=pass_fsspec_kw) as fitso:
                    rel_tab = fitso[self._evt_tab_name]
                    # For posterity, and convenience, we'll store the whole set of available column names
                    if self._all_col_names is None:
                        self._all_col_names = list(rel_tab.columns.names)

                    # This is rather inelegant (see the top of this function for a similar check and an explanation)
                    if columns is not None:
                        # If there are any passed columns which aren't in the event list columns, we'll find them here
                        #  and raise an exception (usefully telling the user which columns are bad and which columns
                        #  they have to choose from).
                        bad_cols = [cc for cc in columns if cc not in self._all_col_names]
                        if len(bad_cols) > 0:
                            raise ValueError("The following column(s) are not available in this event list; "
                                             "{c}. Please choose from; {a}.".format(c=",".join(bad_cols),
                                                                                    a=",".join(self._all_col_names)))

                    # And finally, we read the event list data into this EventList instance - and if the user specified
                    #  a set of columns we load only those
                    if columns is not None:
                        self._data = Table(rel_tab.data)[columns]
                    else:
                        self._data = Table(rel_tab.data)

                    # And update the EventList's knowledge of it having a subset loaded
                    self._data_col_subset = data_col_subset

            except OSError:
                if self._local_file:
                    raise FileNotFoundError("{f} events data cannot be opened. This product (of type {t}) is "
                                            "associated with {s}.".format(f=self.path, s=self.src_name, t=self.type))
                else:
                    raise FileNotFoundError("The remote file's ({f}) events data cannot be opened. This product (of "
                                            " type {t}) is associated with {s}.".format(f=self.path, s=self.src_name,
                                                                                        t=self.type))

    # --------- Define external functions ---------
    def get_position_type_col_names(self, position_type: str) -> Tuple[str, str]:
        """
        Get method to retrieve the event list data table column names (both X and Y) which
        are currently assigned to the input position type. The input position type
        may be 'sky', 'det', or 'raw'.

        :param None position_type: The position type to fetch event list data table column
            names for; either 'sky', 'det', or 'raw'.
        :return: A tuple of the form (<x column name>, <y column name>)
        :rtype: Tuple[str, str]
        """
        position_type = position_type.lower()
        if position_type not in LIM_KEY_MAP:
            raise KeyError(f"Value of 'position_type' ({position_type}) is not valid. It must "
                           f"be one of {list(LIM_KEY_MAP.keys())}.")

        # Attempting to determine the columns to use for image generation
        if position_type == 'sky':
            x_col, y_col = self.sky_x_col, self.sky_y_col
        elif position_type == 'det':
            x_col, y_col = self.det_x_col, self.det_y_col
        elif position_type == 'raw':
            x_col, y_col = self.raw_x_col, self.raw_y_col
        else:
            raise XGADeveloperError(f"The {position_type} position type has been added "
                                    f"to the LIM_KEY_MAP constant, but not this function. "
                                    f"Contact the developers.")

        return x_col, y_col

    def get_wcs_keys(self, position_type: str, prefix: Optional[Union[str, List[str]]] = None) -> dict:
        """
        Get method to fetch the names of the FITS header keys required to construct a WCS, for the
        input position type. This method tries several different formats for the necessary key
        names, and will raise an error if the key cannot be found.

        To search for a specific key, pass the prefix(es) to the 'prefix' argument; e.g. `prefix="TCDLT"`, or
        `prefix=["TCDLT", "TCRPX"]`.

        :param str position_type: The coordinate system type, 'sky', 'det', or 'raw'.
        :param str/List[str]/None prefix: Manually specified prefix(es) of a WCS key(s); e.g. 'TCRPX'
            or ["TCDLT", "TCRPX"]. Default is None, in which case a standard set of prefixes defined by
            the keys of the `WCS_PREFIX_ALTS` constant are used.
        :return: A dictionary, with "x" and "y" top-level keys, and standard WCS header key
            prefixes (e.g. 'TCDLT') as lower level keys. Values are the corresponding keys for
            the specific position type provided.
        :rtype: dict
        """

        # Checking the prefix input, and using it to decide which keys we are looking for.
        if prefix is not None and isinstance(prefix, str):
            rel_prefixes = [prefix]
        elif prefix is not None and isinstance(prefix, list):
            rel_prefixes = prefix
        elif prefix is not None:
            raise TypeError("The 'prefix' argument must be either a string or a list of strings.")
        else:
            rel_prefixes = list(WCS_PREFIX_ALTS.keys())

        # Use the position type supplied by the user to fetch the correct x and y col names
        x_col, y_col = self.get_position_type_col_names(position_type)

        # We attempt to find the keys using TTYPE indices
        # For the x-col
        x_ind = [hdr_key.split('TTYPE')[-1]
                 for hdr_key, hdr_val in self.event_header.items()
                 if hdr_val == x_col and 'TTYPE' in hdr_key]
        # Then the y-col
        y_ind = [hdr_key.split('TTYPE')[-1]
                 for hdr_key, hdr_val in self.event_header.items()
                 if hdr_val == y_col and 'TTYPE' in hdr_key]

        ret_keys = {cur_prefix: {} for cur_prefix in rel_prefixes}
        if len(x_ind) == 1 and len(y_ind) == 1:
            for cur_prefix in rel_prefixes:
                cur_all_x_key_attempts = []
                cur_all_y_key_attempts = []

                # We try the standard prefix + index first
                cur_x_key = cur_prefix + x_ind[0]
                cur_y_key = cur_prefix + y_ind[0]
                cur_all_x_key_attempts.append(cur_x_key)
                cur_all_y_key_attempts.append(cur_y_key)

                # If that's not in the header, we try the prefix minus the 'T' + index
                if cur_x_key not in self.event_header:
                    cur_x_key = cur_prefix[1:] + x_ind[0]
                    cur_all_x_key_attempts.append(cur_x_key)
                else:
                    ret_keys[cur_prefix]["x"] = cur_x_key
                if cur_y_key not in self.event_header:
                    cur_y_key = cur_prefix[1:] + y_ind[0]
                    cur_all_y_key_attempts.append(cur_y_key)
                else:
                    ret_keys[cur_prefix]["y"] = cur_y_key

                if "x" in ret_keys[cur_prefix] and "y" in ret_keys[cur_prefix]:
                    continue

                # Second fallback is to check some slightly different names for these
                #  WCS keys that have been used by older missions
                if cur_x_key not in self.event_header and cur_prefix in WCS_PREFIX_ALTS:
                    cur_x_key = WCS_PREFIX_ALTS[cur_prefix] + x_ind[0]
                    cur_all_x_key_attempts.append(cur_x_key)
                else:
                    ret_keys[cur_prefix]["x"] = cur_x_key

                if cur_y_key not in self.event_header and cur_prefix in WCS_PREFIX_ALTS:
                    cur_y_key = WCS_PREFIX_ALTS[cur_prefix] + y_ind[0]
                    cur_all_y_key_attempts.append(cur_y_key)
                else:
                    ret_keys[cur_prefix]["y"] = cur_y_key

                if "x" in ret_keys[cur_prefix] and "y" in ret_keys[cur_prefix]:
                    continue

                # Now if the keys aren't there, we raise an error
                if cur_x_key not in self.event_header:
                    raise KeyError(f"The {cur_prefix}-type key for {x_col} cannot be found in the events "
                                   f"header. The following were tested; {cur_all_x_key_attempts}")
                else:
                    ret_keys[cur_prefix]["x"] = cur_x_key

                if cur_y_key not in self.event_header:
                    raise KeyError(f"The {cur_prefix}-type key for {y_col} cannot be found in the events "
                                   f"header. The following were tested; {cur_all_y_key_attempts}")
                else:
                    ret_keys[cur_prefix]["y"] = cur_y_key


        return ret_keys

    def get_columns_from_data(self, col_names: List[str]) -> pd.DataFrame:
        """
        This method allows you to retrieve specific columns from the event list table, without loading the whole table
        into memory.

        :param List[str] col_names: A list of column names to retrieve.
        :return: A pandas DataFrame containing the specified columns.
        :rtype: pd.DataFrame
        """
        # This will handle updating the loaded data, if another subset has already been loaded, and won't re-load
        #  data unless it really needs to. Running this will result in changes to _data
        self._read_data_on_demand(col_names)

        return self.data[col_names].to_pandas()

    def get_filtered_data(self, col_names: List[str], filt_operations: dict) -> pd.DataFrame:
        """
        A method to retrieve a filtered subset of the events table data - this is useful for the production of
        various X-ray products (images, lightcurves, etc.), as we rarely wish to use every single event.

        :param List[str] col_names: A list of column names to retrieve.
        :param dict filt_operations: A dictionary of filtering operations to apply to the event list data. The
            dictionary should be structured with column names as keys and filtering operations as values. The
            filtering operations can be specified either as strings (e.g. "> 5", "< 10") or as callable
            functions (e.g. lambda functions). Multiple operations on a single column should be provided as a
            list. For example - {'ENERGY': ['>100', '<1000'], 'X': [lambda x: x > 0]}
        :return: A pandas DataFrame containing the specified columns, filtered according to the operations
        :rtype: pd.DataFrame
        """
        # Check to make sure that all the filtering operations we're being asked to perform are
        #  on data columns that we're actually going to acquire
        filt_op_cols = np.array(list(filt_operations.keys()))
        filt_col_not_in_data = np.array([cur_col not in col_names for cur_col in filt_op_cols])
        if any(filt_col_not_in_data):
            miss_cols = ", ".join(filt_op_cols[filt_col_not_in_data])
            warn("Filtering operations are specified on columns not in 'col_names' ({mc}), the missing "
                 "columns will be added to the 'col_names' list.".format(mc=miss_cols), stacklevel=2)
            col_names += (filt_op_cols[filt_col_not_in_data]).tolist()

        # Make sure that all the filtering operations are specified in lists
        if any([not isinstance(filt_cmds, list) for filt_cmds in filt_operations.values()]):
            filt_operations = {filt_col: [filt_cmds] if not isinstance(filt_cmds, list) else filt_cmds
                               for filt_col, filt_cmds in filt_operations.items()}

        # Acquiring the specified columns
        rel_data = self.get_columns_from_data(col_names)

        # Setting up the overall mask that will be applied at the end of this function - this will be modified
        #  by each filtering operation.
        evt_mask = np.ones(len(rel_data), dtype=bool)
        # Iterating through the filtering operations
        for cur_filt_col, cur_filt_cmds in filt_operations.items():
            col_mask = np.ones(len(rel_data), dtype=bool)
            for cur_cmd in cur_filt_cmds:
                if isinstance(cur_cmd, str):
                    # Dynamically evaluate a string filtering command
                    col_mask &= eval(f"rel_data['{cur_filt_col}'] {cur_cmd}")
                elif callable(cur_cmd):
                    # Or apply a user-defined lambda function
                    col_mask &= cur_cmd(rel_data[cur_filt_col])
            # We now include the mask that resulted from the filtering operation on the current column
            #  into the overall mask
            evt_mask &= col_mask

        return rel_data[evt_mask]

    def unload(self, unload_data: bool = True, unload_header: bool = True):
        """
        This method allows you to safely remove the header and/or data information stored in memory.

        :param bool unload_data: Specifies whether the data should be unloaded from memory. Default is True, as the
            event list data is liable to take up far more memory than the header, meaning it is more likely to need to
            be removed.
        :param bool unload_header: Specifies whether the header should be unloaded from memory. Default is True.
        """
        # Doesn't make sense in this case, as the method wouldn't do anything - as it was probably a mistake to call
        #  the method like this I throw an error so the user knows
        if not unload_data and not unload_header:
            raise ValueError("At least one of the 'unload_data' and 'unload_header' arguments must be True.")

        # Pretty simple, if the user wants the data gone then we use the existing property delete method for data
        if unload_data:
            del self.data

        # And if they want the header gone then we use the property delete method for header
        if unload_header:
            del self.header

    def generate_image(self, bin_size: Optional[Quantity] = None, x_lims: Optional[Quantity] = None,
                       y_lims: Optional[Quantity] = None, lo_en: Optional[Quantity] = None,
                       hi_en: Optional[Quantity] = None, filt_operations: Optional[dict] = None,
                       save_path: Optional[str] = None, donor_image: Optional[Image] = None,
                       position_type: Optional[str] = None) -> Image:
        """
        Generate a 2D image from the event list data by binning events into pixels. The method allows control over
        binning size, spatial boundaries, energy filtering, and output file saving.

        :param Quantity bin_size: The size of bins to use when creating the image. Can be specified in pixels
            ('pix') or angular units (e.g. 'deg', 'arcmin'). If None, uses mission defaults or falls back to 1 pixel.
        :param Quantity x_lims: The x-axis boundaries of the generated image. Can be specified in pixels ('pix') or
            angular units (e.g. 'deg'). If None, uses full detector field of view.
        :param Quantity y_lims: The y-axis boundaries of the generated image. Can be specified in pixels ('pix') or
            angular units (e.g. 'deg'). If None, uses full detector field of view.
        :param Quantity lo_en: Lower energy boundary for event filtering. Must be in energy units (e.g. 'eV', 'keV').
            If specified, hi_en must also be specified.
        :param Quantity hi_en: Upper energy boundary for event filtering. Must be in energy units (e.g. 'eV', 'keV').
            If specified, lo_en must also be specified.
        :param dict filt_operations: A dictionary of filtering operations to apply to the event list data. The
            dictionary should be structured with column names as keys and filtering operations as values. The
            filtering operations can be specified either as strings (e.g. "> 5", "< 10") or as callable
            functions (e.g. lambda functions). Multiple operations on a single column should be provided as a
            list. For example - {'PI': ['>100', '<1000'], 'X': [lambda x: x > 0]}
        :param str save_path: Path to where the generated image should be saved as a FITS file. If
            None, then the image will exist only in memory, and will not be written to storage.
        :param Image donor_image: An existing XGA Image object whose WCS and grid will be used to project the
            event data into. If this is provided, bin_size, x_lims, and y_lims are ignored.
        :param str position_type: The coordinate system to use for the image generation, 'sky', 'det', or 'raw'.
            The default is None, in which case the mission's default imaging coordinate system is used.
        :return: An XGA Image object made up of the spatially binned event data and associated WCS information.
        :rtype: Image
        """
        position_type = position_type.lower()

        # Determine the position type to use
        if position_type is None and self.telescope.upper() in MISSION_COL_DB:
            position_type = MISSION_COL_DB[self.telescope.upper()].get('imagecoord').lower()
        elif position_type is None:
            position_type = 'sky'

        x_col, y_col = self.get_position_type_col_names(position_type)
        en_col = self.en_col

        # If the telescope is in the mission database, we can check for extra information
        if self.telescope.upper() in MISSION_COL_DB:
            rel_miss_info = MISSION_COL_DB[self.telescope.upper()]
            if rel_miss_info['imagecoord'] is None:
                raise ValueError(f"Observations taken by {self.telescope} may not contain spatial information.")

        # --------------------- Validating input configuration ---------------------
        # ---------------- Checking the save path ----------------
        # Checking that the directory in which the image should be saved (if the user has specified that
        #  it should be written to a file, and a directory is part of the save_path) actually exists
        if (save_path is not None and
                (os.path.dirname(save_path) != '' and not os.path.exists(os.path.dirname(save_path)))):
            raise FileNotFoundError("The directory in which the image is to be saved "
                                    "({d}) does not exist.".format(d=os.path.dirname(save_path)))

        # If None has been passed for the filtering operations, we'll turn it into an empty dictionary
        if filt_operations is None:
            filt_operations = {}
        # Otherwise we'll check that the user isn't trying to specify x_col, y_col, or en_col limits in
        #  the filtering operations dictionary
        elif x_col in filt_operations or y_col in filt_operations or en_col in filt_operations:
            raise ValueError("The filtering operations dictionary cannot contain keys spatial columns ({x}, {y}), or "
                             "the energy column ({e}), as these are controlled separately by this "
                             "method.".format(x=x_col, y=y_col, e=en_col))
        # --------------------------------------------------------

        # ------- Converting ints to assumed pixel coords --------
        # Making some arguments into quantities with an assumed unit if they were passed as integers.
        # If a simple integer is passed, we assume that it is a bin size in pixels
        if isinstance(bin_size, int):
            bin_size = Quantity(bin_size, 'pix')

        # Converting any non-quantity integer boundary limits to Quantity objects, assuming 'pix' units
        if (not isinstance(x_lims, Quantity) and
                (isinstance(x_lims, (list, np.ndarray)) and all([isinstance(xl, int) for xl in x_lims]))):
            x_lims = Quantity(x_lims, 'pix')
        if (not isinstance(y_lims, Quantity) and
                (isinstance(y_lims, (list, np.ndarray)) and all([isinstance(yl, int) for yl in y_lims]))):
            y_lims = Quantity(y_lims, 'pix')
        # --------------------------------------------------------

        # --------- Setting up x and y coordinate limits ---------
        # Parsing the user-specified data limits
        if x_lims is not None and x_lims.diff() <= 0:
            raise ValueError("The second element of 'x_lims' must be greater than the first.")
        elif x_lims is not None and (x_lims.unit.is_equivalent('deg')):
            mid_pos = self.radec_sky_wcs.all_pix2world(*Quantity([self.sky_pix_lims[0].mean(),
                                                        self.sky_pix_lims[1].mean()]).value, 1)
            low_x_lim = self.radec_sky_wcs.all_world2pix(x_lims[0].value, mid_pos[1], 1)[0]
            upp_x_lim = self.radec_sky_wcs.all_world2pix(x_lims[1].value, mid_pos[1], 1)[0]
            x_lims = np.sort(Quantity([low_x_lim, upp_x_lim]))
            x_lims[0] = np.floor(x_lims[0])
            x_lims[1] = np.ceil(x_lims[1])
        elif x_lims is None:
            try:
                x_lims = self.sky_pix_lims[0]
            except ValueError:
                # The final fallback - finding upper and lower limits using the data.
                #  We deliberately use the unfiltered data here, as we would rather
                #  err on the side of caution and have wider limits
                x_lims = Quantity([self.data[x_col].min(), self.data[x_col].max()], 'pix')
        #
        x_lims = x_lims.astype(int)

        if y_lims is not None and y_lims.diff() <= 0:
            raise ValueError("The second element of 'y_lims' must be greater than the first.")
        elif y_lims is not None and (y_lims.unit.is_equivalent('deg')):
            mid_pos = self.radec_sky_wcs.all_pix2world(*Quantity([self.sky_pix_lims[0].mean(),
                                                        self.sky_pix_lims[1].mean()]).value, 1)
            low_y_lim = self.radec_sky_wcs.all_world2pix(mid_pos[0], y_lims[0].value, 1)[1]
            upp_y_lim = self.radec_sky_wcs.all_world2pix(mid_pos[0], y_lims[1].value, 1)[1]
            y_lims = np.sort(Quantity([low_y_lim, upp_y_lim]))
            y_lims[0] = np.floor(y_lims[0])
            y_lims[1] = np.ceil(y_lims[1])
        elif y_lims is None:
            try:
                y_lims = self.sky_pix_lims[1]
            except ValueError:
                # The final fallback - finding upper and lower limits using the data.
                #  We deliberately use the unfiltered data here, as we would rather
                #  err on the side of caution and have wider limits
                y_lims = Quantity([self.data[y_col].min(), self.data[y_col].max()], 'pix')
        #
        x_lims = x_lims.astype(int)
        y_lims = y_lims.astype(int)
        # --------------------------------------------------------

        # ------------- Setting up the binning size --------------
        # Parsing the user-specified bin size, if indeed they did specify one. If not, then we
        #  pull the default size for the mission, and if that isn't available, then we default to a bin size of 1

        # Need to lower the telescope name for this check, as in XGA mission names are in lower case (not necessarily
        #  true in event file headers).
        if bin_size is None and self.telescope.lower() in DEFAULT_IMAGE_BINNING:
            # Read out the default bin size for this mission if this event list's instrument is a key in the image
            #  binning constant dictionary
            if self.instrument.lower() in DEFAULT_IMAGE_BINNING[self.telescope.lower()]:
                bin_size = Quantity(DEFAULT_IMAGE_BINNING[self.telescope.lower()][self.instrument.lower()], 'pix')

            # If the instrument isn't a key, it is possible that the instrument name in the event list is different
            #  to that used by XGA, so we check an 'alternative name' constant set up for this purpose
            elif [self.instrument.lower() in inst_alts for inst_alts in
                  ALT_INST_NAMES[self.telescope.lower()].values()]:
                # Inefficient considering we've already performed the check, but nicer code structure
                rel_xga_inst_name = [xga_inst for xga_inst, inst_alts in
                                     ALT_INST_NAMES[self.telescope.lower()].items()
                                     if self.instrument.lower() in inst_alts][0]
                if rel_xga_inst_name in DEFAULT_IMAGE_BINNING[self.telescope.lower()]:
                    bin_size = Quantity(DEFAULT_IMAGE_BINNING[self.telescope.lower()][rel_xga_inst_name], 'pix')
                else:
                    warn(f"No XGA default binning size has been set for the instrument '{self.instrument}' "
                         f"(please contact the developers), defaulting to a bin size of 1.", stacklevel=2)
                    bin_size = Quantity(1, 'pix')
        # The overall fallback, setting the binning to one
        elif bin_size is None:
            warn("No XGA default binning size has been set for the instrument '{i}' (please contact the "
                 "developers), defaulting to a bin size of 1.".format(i=self.instrument), stacklevel=2)
            bin_size = Quantity(1, 'pix')

        # We allow the bin_size argument to be in angular units, but make sure to translate it to pixels
        if bin_size.unit.is_equivalent('deg'):
            # We enforce square pixels by using the first element of this calculation - though
            #  in most cases the calculated bin size for x and y axes will be the same
            bin_size = np.ceil((bin_size / self.deg_per_sky).to('pix'))[0]
        # --------------------------------------------------------

        # ------------- Setting up the energy limits -------------
        # Initially check that both energy boundaries have been set
        check_en = [lo_en is not None, hi_en is not None]
        if any(check_en) and not all(check_en):
            raise ValueError("If either 'lo_en' or 'hi_en' are specified, both must be.")
        # Check that they are both in the correct units
        elif lo_en is not None and any([not lo_en.unit.is_equivalent('eV'), not hi_en.unit.is_equivalent('eV')]):
            raise UnitConversionError("Quantities passed to 'lo_en' and 'hi_en' must be convertible to eV.")
        # Check validity of lower and upper energy limits
        elif lo_en is not None and (lo_en >= hi_en):
            raise ValueError("Value passed to 'lo_en' must be less than or equal to 'hi_en'.")

        # Setup filtering operations for the events data
        if lo_en is not None:
            # Convert energy limits to channels if necessary
            lo_chan = (lo_en / self.ev_per_channel).decompose().value
            hi_chan = (hi_en / self.ev_per_channel).decompose().value
            filt_operations[en_col] = [f">={lo_chan}", f"<={hi_chan}"]
        # --------------------------------------------------------
        # --------------------------------------------------------------------------

        # ------------------ Generating an image from user input -------------------
        # After all of this converting and dealing with different potential inputs for bin_size, we store
        #  the final angular width/height of each pixel
        ang_bin_size = (bin_size*self.deg_per_sky).to('deg')[0].value
        # Make sure that the bin size is an integer
        bin_size = bin_size.astype(int)

        # Makes sure that any extra columns that the user wishes to filter on are in the set we request
        #  from the get_filtered_data method (they would be added by that method, but with an annoying
        #  warning that we do not need)
        cols_to_get = list(set([x_col, y_col, en_col] + list(filt_operations.keys())))
        rel_evt_data = self.get_filtered_data(cols_to_get, filt_operations)

        if donor_image is None:
            # We define bin edges such that they are centered on the sky pixel coordinates
            #  (e.g. for bin_size=1, a bin covering sky pixel 1 has edges [0.5, 1.5))
            x_bins = np.arange(x_lims.value[0] - 0.5, x_lims.value[1] + 0.5 + bin_size.value, bin_size.value)
            y_bins = np.arange(y_lims.value[0] - 0.5, y_lims.value[1] + 0.5 + bin_size.value, bin_size.value)

            # We bin the filtered event data into the histogram
            binned_data = np.histogram2d(rel_evt_data[y_col], rel_evt_data[x_col], bins=(y_bins, x_bins))[0]

            # Setting up the new WCS - we use the internal build method for the relevant columns
            base_wcs = self._build_wcs(x_col, y_col, position_type)
            im_wcs = WCS(naxis=2)
            im_wcs.wcs.cdelt = [np.sign(base_wcs.wcs.cdelt[0])*ang_bin_size,
                                    np.sign(base_wcs.wcs.cdelt[1])*ang_bin_size]

            # Calculate RA/Dec at the center of the first bin (origin=1) to set crval
            #  We use the average of the first and second bin edges to get the center
            center_x = (x_bins[0] + x_bins[1]) / 2
            center_y = (y_bins[0] + y_bins[1]) / 2
            min_bnd_radec = base_wcs.all_pix2world(center_x, center_y, 1)

            im_wcs.wcs.crpix = [1, 1]
            im_wcs.wcs.crval = [min_bnd_radec[0], min_bnd_radec[1]]
            im_wcs.wcs.ctype = [base_wcs.wcs.ctype[0], base_wcs.wcs.ctype[1]]

            # Set the lower and upper limits of the image pixel coordinate system (1-based)
            im_wcs.pixel_bounds = [(1, binned_data.shape[1]), (1, binned_data.shape[0])]

        else:
            # If a donor image is provided, we project the events onto its coordinate grid
            # First we transform the filtered event coordinates to RA and Dec (origin=1)
            ev_radec = self.radec_sky_wcs.all_pix2world(rel_evt_data[x_col], rel_evt_data[y_col], 1)
            # Then we transform those world coordinates to the pixel grid of the donor image (origin=1)
            ev_donor_pix = donor_image.radec_wcs.all_world2pix(ev_radec[0], ev_radec[1], 1)

            # We define bins based on the donor image's shape (again centering on pixels, edges are [0.5, 1.5))
            x_bins = np.arange(0.5, donor_image.shape[1] + 1.5, 1)
            y_bins = np.arange(0.5, donor_image.shape[0] + 1.5, 1)

            # We bin the calculated donor pixel coordinates into the grid
            binned_data = np.histogram2d(ev_donor_pix[1], ev_donor_pix[0], bins=(y_bins, x_bins))[0]

            # The WCS for this new image is simply inherited from the donor image
            im_wcs = donor_image.radec_wcs
        # --------------------------------------------------------------------------

        # -------------- Setting up XGA Image and saving if requested --------------
        new_header = [{'name': 'SIMPLE', 'value': 'T'},
                      {'name': 'BITPIX', 'value': binned_data.dtype.itemsize * 8},
                      {'name': 'NAXIS', 'value': 2},
                      {'name': 'NAXIS1', 'value': binned_data.shape[1]},
                      {'name': 'NAXIS2', 'value': binned_data.shape[0]}]

        new_im = Image({'data': binned_data, 'wcs': im_wcs, 'header': new_header}, self.obs_id,
                       self.instrument, "", "", "",
                       lo_en=lo_en, hi_en=hi_en, telescope=self.telescope)

        # We validated the 'save_path' argument earlier, so we'll just get on and save the file
        if save_path is not None:
            # Setting up the header that we'll feed into the HDU that will become the image file - the WCS
            #  is the most important part of that
            im_hdr = im_wcs.to_header()
            # Create a single-HDU fits file, just containing the image
            im_hdu = PrimaryHDU(binned_data, im_hdr)
            hdu_list = HDUList([im_hdu])
            hdu_list.writeto(save_path, overwrite=True)
        # --------------------------------------------------------------------------

        return new_im
