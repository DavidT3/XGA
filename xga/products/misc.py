#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 14/02/2024, 12:26. Copyright (c) The Contributors

from typing import List

from fitsio import read_header, FITSHDR
import fitsio
import pandas as pd

from . import BaseProduct


class EventList(BaseProduct):
    """
    A product class for event lists, it stores information about the event list.

    :param str path: The path to where the event list file SHOULD be located.
    :param str obs_id: The ObsID related to the event list being declared.
    :param str instrument: The instrument related to the event list being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the event list.
    :param str telescope: The telescope that is the source of this event list. The default is None.
    :param List[str] obs_ids: The obs ids that were combined to make this event list. The default is None.
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, telescope: str = None, obs_ids: List[str] = None):
        """
        The init method of the EventList class, a product class for event lists, it stores information about
        the event list.

        :param str path: The path to where the event list file SHOULD be located.
        :param str obs_id: The ObsID related to the event list being declared.
        :param str instrument: The instrument related to the event list being declared.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the event list.
        :param str telescope: The telescope that is the source of this event list. The default is None.
        """
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, telescope=telescope)
        self._prod_type = "events"
        self._telescope = telescope
        # These store the header of the event list fits file (if read in), as well as the main table of event
        #  information (again if read in).
        self._header = None
        self._data = None

        if obs_ids is not None and not isinstance(obs_ids, List):
            raise ValueError("The 'obs_ids' argument nust be a list of strings.")

        if obs_ids is not None and not all(isinstance(obs, str) for obs in obs_ids):
            raise ValueError("The 'obs_ids' argument nust be a list of strings.")
        
        self._obs_ids = obs_ids
    
    @property
    def obs_ids(self) -> list:
        """
        Property getter for the ObsIDs that are involved in this image, if combined. Otherwise will return a list
        with one element, the single relevant ObsID.

        :return: List of ObsIDs involved in this image.
        :rtype: list
        """

        return self._obs_ids

    # This absolutely doesn't get a setter considering it's the header object
    @property
    def header(self) -> FITSHDR:
        """
        Property getter allowing access to the astropy fits header object of this event list.

        :return: The header of the primary data table of the event list.
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

    def _read_header_on_demand(self):
        """
        This will read the event list header into memory, without loading the data from the event list main table. That
        way the user can get access to the summary information stored in the header without wasting a lot of memory.
        """
        try:
            # Reads only the header information
            self._header = read_header(self.path)
        except OSError:
            raise FileNotFoundError("FITSIO read_header cannot open {f}, possibly because there is a problem with "
                                    "the file, it doesn't exist, or maybe an SFTP problem? This product is associated "
                                    "with {s}.".format(f=self.path, s=self.src_name))

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

    def _read_data_on_demand(self):
        """
        This will read the event list table into memory.
        """
        if self._telescope != 'erosita':
            raise NotImplementedError("Reading Eventlist tables is not yet implemented for telescopes that aren't eROSITA.")
        else:
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
    
    def get_columns_from_data(self, colnames: List[str]) -> pd.DataFrame:
        """
        This method allows you to retrieve specific columns from the event list table, without loading the whole table 
        into memory.

        :param List[str] colnames: A list of column names to retrieve.
        """

        if self._telescope != 'erosita':
            raise NotImplementedError("Reading Eventlist tables is not yet implemented for telescopes that aren't eROSITA.")
        
        else:
            try:
                # reads the events table into a np.recarray
                arr = fitsio.read(self.path, columns=colnames, ext=1)
                # nicer to have a dataframe than a recarray
                return pd.DataFrame.from_records(arr)
            
            except ValueError:
                # The error message generated by fitsio is informative enough
                raise

            except OSError:
                raise FileNotFoundError("FITSIO read method cannot open {f}, possibly because there is a problem with "
                                        "the file, it doesn't exist, or maybe an SFTP problem? This product is associated "
                                        "with {s}.".format(f=self.path, s=self.src_name))