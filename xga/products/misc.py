#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 12/10/2023, 15:47. Copyright (c) The Contributors


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
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, telescope: str = None):
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






