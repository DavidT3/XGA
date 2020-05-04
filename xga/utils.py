#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 03/05/2020, 12:14. Copyright (c) David J Turner

# TODO Possibly this is where the config file I'll have to come up with will be parsed/checked for
# TODO Might do a decorator here to try except all sas functions, like Tim suggested


class HeasoftError(Exception):
    def __init__(self, *args):
        """
        Exception raised for unexpected output from HEASOFT calls, currently all encompassing.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{0} '.format(self.message)
        else:
            return 'HeasoftError has been raised'




