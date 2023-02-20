#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors


from . import BaseProduct


class EventList(BaseProduct):
    """
    A product class for event lists, largely just used as a means to store information about the event list.

    :param str path: The path to where the event list file SHOULD be located.
    :param str obs_id: The ObsID related to the event list being declared.
    :param str instrument: The instrument related to the event list being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the event list.
    """
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str):
        """
        The init method of the EventList class.
        """
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd)
        self._prod_type = "events"






