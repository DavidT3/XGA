#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 04/05/2020, 16:57. Copyright (c) David J Turner


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


class XGAConfigError(Exception):
    def __init__(self, *args):
        """
        Exception raised for flawed XGA config files.
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
            return 'XGAConfig has been raised'


class SASNotFoundError(Exception):
    def __init__(self, *args):
        """
        Exception raised if the XMM Scientific Analysis System can not be found on the system.
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
            return 'XGAConfig has been raised'