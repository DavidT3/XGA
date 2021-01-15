#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/01/2021, 16:30. Copyright (c) David J Turner


from . import BaseProduct


class EventList(BaseProduct):
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str):
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd)
        self._prod_type = "events"






