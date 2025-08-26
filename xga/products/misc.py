#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 26/08/2025, 18:51. Copyright (c) The Contributors
from typing import List

import fitsio
import pandas as pd
from astropy.io import fits
from fitsio import FITSHDR

from . import BaseProduct
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

        # We attempt to automatically derive the telescope, ObsID, and instrument (if they haven't been
        #  passed by the user) from the event list header
        if telescope is None:
            self._telescope = self.header['TELESCOP']
        if obs_id is None:
            self._obs_id = self.header['OBSID']
        if instrument is None:
            self._instrument = self.header['INSTRUME']

        # Checking the formatting of the obs_ids argument
        if obs_ids is not None and (not isinstance(obs_ids, List) or
                                    (isinstance(obs_ids, List) and not all(isinstance(obs, str) for obs in obs_ids))):
            raise ValueError("The 'obs_ids' argument must be a list of strings.")

        self._obs_ids = obs_ids

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
    def header(self) -> FITSHDR:
        """
        Property getter allowing access to the astropy fits header object of this event list.

        :return: The primary header of the event list header.
        :rtype: FITSHDR
        """
        # If the header attribute is None then we know we have to read the header in
        if self._header is None:
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

    def _read_header_on_demand(self):
        """
        This will read the event list header into memory, without loading the data from the event list main table. That
        way the user can get access to the summary information stored in the header without wasting a lot of memory.
        """

        # We could likely treat the remote and local file access identically, but we're doing it this way for
        #  now out of an abundance of caution - I don't know how local files would behave using fsspec
        if self._local_file:
            try:
                # Reads only the header information
                # self._header = read_header(self.path)
                with fits.open(self.path, lazy_load_hdus=True) as fitso:
                    self._header = fitso[0].header
            except OSError:
                raise FileNotFoundError("{f} header cannot be opened. This product (of type {t}) is associated "
                                        "with {s}.".format(f=self.path, s=self.src_name, t=self.type))
        else:
            try:
                with fits.open(self.path, lazy_load_hdus=True, use_fsspec=True,
                               fsspec_kwargs=self.fsspec_kwargs) as fitso:
                    self._header = fitso[0].header
            except OSError:
                raise FileNotFoundError("The remote file's ({f}) header cannot be opened. This product (of type {t}) "
                                        "is associated with {s}.".format(f=self.path, s=self.src_name, t=self.type))

    def _read_data_on_demand(self):
        """
        This will read the event list table into memory.
        """

        try:
            # reads the events table into a np.recarray
            arr = fitsio.read(self.path, ext=1)
            # nicer to return a df than an array
            self._data = pd.DataFrame.from_records(arr)

        except OSError:
            raise FileNotFoundError("FITSIO read method cannot open {f}, possibly because there is a problem with "
                                    "the file, it doesn't exist, or maybe an SFTP problem? This product is associated "
                                    "with {s}.".format(f=self.path, s=self.src_name))

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

    def get_columns_from_data(self, col_names: List[str]) -> pd.DataFrame:
        """
        This method allows you to retrieve specific columns from the event list table, without loading the whole table
        into memory.

        :param List[str] col_names: A list of column names to retrieve.
        """

        # There is no sense reading in the columns again, if the whole event list is already in memory
        if self._data is not None:
            return self.data.loc[:, col_names]

        try:
            # Reads the events table into a np.recarray
            arr = fitsio.read(self.path, columns=col_names, ext=1)

            # Makes sure that the byte order is correct
            if arr.dtype[0].byteorder != '<':
                arr = arr.view(arr.dtype.newbyteorder()).byteswap(inplace=False)

            # Much nicer to have a dataframe than a recarray
            return pd.DataFrame.from_records(arr)

        except ValueError as err:
            # The error message generated by fitsio is informative enough
            raise err

        except OSError:
            raise FileNotFoundError("FITSIO read method cannot open {f}, possibly because there is a problem with "
                                    "the file, it doesn't exist, or maybe an SFTP problem? This product is associated "
                                    "with {s}.".format(f=self.path, s=self.src_name))