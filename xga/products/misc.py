#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 27/08/2025, 16:45. Copyright (c) The Contributors
from typing import List

import pandas as pd
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from . import BaseProduct
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

    def generate_image(self):
        """

        :return:
        :rtype:
        """
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
            # WCS
            radec_wcs = WCS(naxis=2)

            radec_wcs.wcs.cdelt = [self.event_header[rel_miss_info['im_xdelt']],
                                   self.event_header[rel_miss_info['im_ydelt']]]
            radec_wcs.wcs.crpix = [self.event_header[rel_miss_info['im_xcritpix']],
                                   self.event_header[rel_miss_info['im_ycritpix']]]
            radec_wcs.wcs.crval = [self.event_header[rel_miss_info['im_xcritval']],
                                   self.event_header[rel_miss_info['im_ycritval']]]
            radec_wcs.wcs.ctype = [self.event_header[rel_miss_info['im_xproj']],
                                   self.event_header[rel_miss_info['im_yproj']]]
        else:
            x_col = "X"
            y_col = "Y"

        print(radec_wcs)



