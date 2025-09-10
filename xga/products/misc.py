#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 09/09/2025, 22:59. Copyright (c) The Contributors
import os.path
from typing import List, Tuple

import numpy as np
import pandas as pd
from astropy import wcs
from astropy.io import fits
from astropy.io.fits import PrimaryHDU, HDUList
from astropy.table import Table
from astropy.units import Quantity, UnitConversionError
from astropy.wcs import WCS

from . import BaseProduct, Image
from .. import MISSION_COL_DB
from ..exceptions import XGADeveloperError


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
    """

    def __init__(self, path: str, obs_id: str = None, instrument: str = None, stdout_str: str = None,
                 stderr_str: str = None, gen_cmd: str = None, telescope: str = None, obs_ids: List[str] = None,
                 force_remote: bool = False, fsspec_kwargs: dict = None):
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
        :param bool force_remote: Used to force the product instantiation to treat the passed path string as a url to
            a remote dataset, and to use fsspec to read/stream the data.
        :param dict fsspec_kwargs: Optional arguments that can be passed fsspec when reading or streaming remote
            datasets - e.g. to pass credentials to access an S3 bucket. Default value is None, which sets the
            argument to {"anon": True}, making it instantly compatible with NASA archive S3 buckets.
        """
        # A validity check to help remind me to pass the telescope to the super-class init when this merges with
        #  multi-mission XGA
        if hasattr(super(), 'telescope'):
            raise XGADeveloperError("S3 streaming event lists have been merged into multi-mission XGA, and the "
                                    "call to BaseProduct init in EventList needs to be updated.")
        else:
            self._telescope = telescope

        # Call the BaseProduct init, sets up some attributes
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, None, force_remote, fsspec_kwargs)
        self._prod_type = "events"
        # These store the header of the event list fits file (if read in), as well as the main table of event
        #  information (again if read in).
        self._header = None
        self._data = None
        # Also include another header attribute, specifically for the event table header
        self._event_header = None

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
            self._tele = self.header['TELESCOP']
        if obs_id is None:
            self._obs_id = self.header['OBS_ID']
        if instrument is None:
            self._inst = self.header['INSTRUME']

        # Checking the formatting of the obs_ids argument
        if obs_ids is not None and (not isinstance(obs_ids, List) or
                                    (isinstance(obs_ids, List) and not all(isinstance(obs, str) for obs in obs_ids))):
            raise ValueError("The 'obs_ids' argument must be a list of strings.")
        self._obs_ids = obs_ids

        # Most missions call the table that contains event information "EVENTS", but it isn't a given - ROSAT for
        #  instance calls it STDEVT
        self._evt_tab_name = "EVENTS" if self.telescope.lower() not in MISSION_COL_DB \
            else MISSION_COL_DB[self.telescope.lower()]['events']

        # The user may want to use WCSes to convert between different coordinate systems (sky to RA-Dec for
        #  instance), so when they are constructed they will be assigned to these attributes
        self._radec_sky_wcs = None

        # Here we store the mapping between channel and energy, used for the generation of images and
        #  lightcurves. This can be pulled from the mission database file, or set by the user.
        self._ev_per_channel = None

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
        Property getter allowing access to the astropy fits header object of this event list.

        :return: The header of the primary data table of the event list.
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
        WCS information that relates this event list's 'sky' coordinate system (or the system that is primary and
        used for imaging positions) to RA-Dec coordinates.

        :return: The WCS information that relates this event list's 'sky' coordinate system to RA-Dec coordinates.
        :rtype: astropy.wcs.WCS
        """
        # If we haven't already, we need to construct the WCS now
        if self._radec_sky_wcs is None:
            # Check whether the telescope has information in the mission file we maintain (derived from XSELECT's
            #  mission database file) - if it does then we'll use that to specify the header columns that contain
            #  the relevant WCS information.
            if self.telescope.lower() in MISSION_COL_DB:
                rel_miss_info = MISSION_COL_DB[self.telescope.lower()]
                radec_wcs = WCS(naxis=2)
                radec_wcs.wcs.cdelt = [self.event_header[rel_miss_info['im_xdelt']],
                                       self.event_header[rel_miss_info['im_ydelt']]]
                radec_wcs.wcs.crpix = [self.event_header[rel_miss_info['im_xcritpix']],
                                       self.event_header[rel_miss_info['im_ycritpix']]]
                radec_wcs.wcs.crval = [self.event_header[rel_miss_info['im_xcritval']],
                                       self.event_header[rel_miss_info['im_ycritval']]]
                radec_wcs.wcs.ctype = [self.event_header[rel_miss_info['im_xproj']],
                                       self.event_header[rel_miss_info['im_yproj']]]

                x_lims = (self.event_header[rel_miss_info['im_xlim_low']],
                            self.event_header[rel_miss_info['im_xlim_upp']])
                y_lims = (self.event_header[rel_miss_info['im_ylim_low']],
                          self.event_header[rel_miss_info['im_ylim_upp']])
                # Set the lower and upper limits of the sky pixel coordinate system
                radec_wcs.pixel_bounds = [x_lims, y_lims]
                self._radec_sky_wcs = radec_wcs
            else:
                raise NotImplementedError("We cannot yet determine WCS information without header entry names "
                                          "being specified in the 'mission_event_column_name_map.json' file.")
        return self._radec_sky_wcs

    @property
    def deg_per_sky(self) -> Quantity:
        """
        Uses the Sky-RA/Dec WCS (accessible through the radec_sky_wcs property) to provide the angular
        size of a pixel in the sky coordinate system - both x and y directions are returned, though they
        are often the same.

        :return: A two-entry non-scalar property, with the first entry being the x-direction sky pixel
            scale and the second being the y-direction sky pixel scale.
        :rtype: Quantity
        """

        return np.abs(Quantity(self.radec_sky_wcs.wcs.cdelt, 'deg/pix'))

    @property
    def sky_pix_lims(self) -> Tuple[Quantity, Quantity]:
        """
        The X and Y pixel limits of the sky coordinate system (or the system that is primary and
        used for imaging positions).

        :return: Two non-scalar quantities, with the first representing the lower and upper allowed values
            for the primary coordinate (usually sky) coordinate system x-axis, and the second being for the y-axis.
        :rtype: Tuple[Quantity, Quantity]
        """
        return (Quantity(self.radec_sky_wcs.pixel_bounds[0], 'pix').astype(int),
                    Quantity(self.radec_sky_wcs.pixel_bounds[1], 'pix').astype(int))

    # @property
    # def ev_per_channel(self) -> Quantity:
    #     """
    #     The mapping between channel values in the energy column of the notebook, and an absolute energy
    #     value in eV. This is used in the construction of images and lightcurves from event lists.
    #
    #     :param Quantity new_val: Passed to the ev_per_channel property setter, the new energy-channel
    #         mapping value in the form of an astropy quantity in units of eV/chan.
    #     :return: An astropy quantity, in units of eV/chan, representing the mapping between channel and energy.
    #     :rtype: Quantity
    #     """
    #     if self._ev_per_channel is None:
    #         MISSION_COL_DB
    #
    #         self._ev_per_channel = None
    #     return self._ev_per_channel
    #
    # @ev_per_channel.setter
    # def ev_per_channel(self, new_val: Quantity):
    #     """
    #     The mapping between channel values in the energy column of the notebook, and an absolute energy
    #     value in eV. This is used in the construction of images and lightcurves from event lists.
    #
    #     :param Quantity new_val: Passed to the ev_per_channel property setter, the new energy-channel
    #         mapping value in the form of an astropy quantity in units of eV/chan.
    #     :return: An astropy quantity, in units of eV/chan, representing the mapping between channel and energy.
    #     :rtype: Quantity
    #     """
    #     # Validity checks on the input
    #     if not isinstance(new_val, Quantity):
    #         raise ValueError("The 'new_val' argument must be an astropy quantity.")
    #     elif not new_val.unit.is_equivalent('eV/chan'):
    #         raise UnitConversionError("The 'new_val' argument must be in units of eV/chan.")
    #
    #     # Converting to the expected units
    #     self._ev_per_channel = new_val.to('eV/chan')

    def _read_header_on_demand(self, table: str = None):
        """
        This will read the primary event list header into memory, without loading the data from the event
        list main table. That way the user can get access to the summary information stored in the header
        without wasting a lot of memory.

        :param table: Optionally defines which table's header to read; default is to read the primary header. Other
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

    def _read_data_on_demand(self, columns: List[str] = None):
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

    def generate_image(self, bin_size: Quantity = None, x_lims: Quantity = None, y_lims: Quantity = None,
                       lo_en: Quantity = None, hi_en: Quantity = None, donor_image: Image = None,
                       save_path: str = None):
        """

        :return:
        :rtype:
        """
        raise NotImplementedError("Intrinsic generation of images from EventList instances is not fully implemented.")
        #
        if self.telescope.lower() in MISSION_COL_DB:
            rel_miss_info = MISSION_COL_DB[self.telescope.lower()]
            if rel_miss_info['imagecoord'] is None:
                raise ValueError("Observations taken by {t} may not contain spatial "
                                 "information.".format(t=self.telescope))
            elif rel_miss_info['imagecoord'] == 'SKY':
                x_col = rel_miss_info["x"]
                y_col = rel_miss_info["y"]
            elif rel_miss_info['imagecoord'] == "DET":
                x_col = rel_miss_info["detx"]
                y_col = rel_miss_info["dety"]
            #
            en_col = rel_miss_info['ecol']
        else:
            raise NotImplementedError("'{t}' does not have an mission DB entry, and manual specification is not "
                                      "supported yet.".format(t=self.telescope))
            x_col = "X"
            y_col = "Y"

        ###################### Validating input configuration ######################
        ################## Checking the save path ##################
        # Checking that the directory in which the image should be saved (if the user has specified that
        #  it should be written to a file, and a directory is part of the save_path) actually exists
        if (save_path is not None and
                (os.path.dirname(save_path) != '' and not os.path.exists(os.path.dirname(save_path)))):
            raise FileNotFoundError("The directory in which the image is to be saved "
                                    "({d}) does not exist.".format(d=os.path.dirname(save_path)))
        ############################################################

        ######### Converting ints to assumed pixel coords ##########
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
        ############################################################

        ########## Setting up x and y coordinate limits ############
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
            x_lims = self.sky_pix_lims[0]
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
            y_lims = self.sky_pix_lims[1]
        #
        x_lims = x_lims.astype(int)
        y_lims = y_lims.astype(int)
        ############################################################

        ############### Setting up the binning size ################
        # Parsing the user-specified bin size, if indeed they did specify one - if not, then we
        #  pull the default size for the mission
        if bin_size is None and ('default_im_binsize' not in MISSION_COL_DB[self.telescope.lower()] or
                                 MISSION_COL_DB[self.telescope.lower()]['default_im_binsize'] is None):
            raise ValueError("No default image bin size is defined for {t} in the mission database file, please "
                             "pass the 'bin_size' argument".format(t=self.telescope))
        elif bin_size is None and MISSION_COL_DB[self.telescope.lower()]['default_im_binsize'] is not None:
            bin_size = Quantity(MISSION_COL_DB[self.telescope.lower()]['default_im_binsize'])

        #
        if bin_size.unit.is_equivalent('deg'):
            # We enforce square pixels by using the first element of this calculation - though
            #  in most cases the calculated bin size for x and y axes will be the same
            bin_size = np.ceil((bin_size / self.deg_per_sky).to('pix'))[0]
        ############################################################

        ############### Setting up the energy limits ###############
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

        # phamax - mission database file keyword for the maximum allowable channel value

        ############################################################

        ############################################################################

        # After all of this converting and dealing with different potential inputs for bin_size, we store
        #  the final angular width/height of each pixel
        ang_bin_size = (bin_size*self.deg_per_sky).to('deg')[0].value

        #
        bin_size = bin_size.astype(int)
        rel_evt_data = self.get_columns_from_data([x_col, y_col, en_col])

        x_bins = np.arange(x_lims.value[0], x_lims.value[1]+bin_size.value, bin_size.value)
        y_bins = np.arange(y_lims.value[0], y_lims.value[1]+bin_size.value, bin_size.value)

        binned_data = np.histogram2d(rel_evt_data[y_col], rel_evt_data[x_col], bins=(y_bins, x_bins))[0]

        # Setting up the new WCS
        im_wcs = WCS(naxis=2)
        im_wcs.wcs.cdelt = [np.sign(self.event_header[rel_miss_info['im_xdelt']])*ang_bin_size,
                                np.sign(self.event_header[rel_miss_info['im_ydelt']])*ang_bin_size]

        min_bnd_radec = self.radec_sky_wcs.all_pix2world(x_bins[0], y_bins[0], 1)
        im_wcs.wcs.crpix = [1, 1]
        im_wcs.wcs.crval = [min_bnd_radec[0], min_bnd_radec[1]]

        im_wcs.wcs.ctype = [self.event_header[rel_miss_info['im_xproj']],
                               self.event_header[rel_miss_info['im_yproj']]]

        # Set the lower and upper limits of the sky pixel coordinate system
        im_wcs.pixel_bounds = [x_lims.value, y_lims.value]

        # We validated the 'save_path' argument earlier, so we'll just get on and save the file
        if save_path is not None:
            # Setting up the header that we'll feed into the HDU that will become the image file - the WCS
            #  is the most important part of that
            im_hdr = im_wcs.to_header()
            # Create a single-HDU fits file, just containing the image
            im_hdu = PrimaryHDU(binned_data, im_hdr)
            hdu_list = HDUList([im_hdu])
            hdu_list.writeto(save_path)

        return binned_data, im_wcs


