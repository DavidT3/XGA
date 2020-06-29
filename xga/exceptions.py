#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 24/06/2020, 10:52. Copyright (c) David J Turner


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
            return 'SASNotFoundError has been raised'


# I do not know if I will keep this as is or expand out into different errors
# The trouble is there are many hundreds of possible SAS errors, and I don't know if I
# want a class for all of them
class SASGenerationError(Exception):
    def __init__(self, *args):
        """
        Exception raised if an error is found to have occured during a run of a part
        of the SAS software
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
            return 'A generic SASNotFoundError has been raised'


class UnknownCommandlineError(Exception):
    def __init__(self, *args):
        """
        Exception raised if an error is found to have occured during a run of a part
        of the SAS software, but it cannot be linked to a SAS function.
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
            return 'A generic UnknownCommandlineError has been raised'


class FailedProductError(Exception):
    def __init__(self, *args):
        """
        Exception raised when trying to access certain data/attributes from an object wrapping a
         product that failed to generate properly.
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
            return 'FailedProductError has been raised.'


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


class NoMatchFoundError(Exception):
    def __init__(self, *args):
        """
        Exception raised when source ra and dec coordinates can't be made to match to any XMM observation.
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
            return 'NoMatchFoundError has been raised'


class NotAssociatedError(Exception):
    def __init__(self, *args):
        """
        Error raised when a given ObsID is not associated with a source object.
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
            return 'NotAssociatedError has been raised'


class UnknownProductError(Exception):
    def __init__(self, *args):
        """
        Error raised when there is an attempt to write an unknown XMM product type to an XGA source.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'UnknownProductTypeError has been raised'


class NoValidObservationsError(Exception):
    def __init__(self, *args):
        """
        Error raised when there is an initial match to an XMM observation, but the
        necessary files cannot be found.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'NoValidObservationsError has been raised'


class MultipleMatchError(Exception):
    def __init__(self, *args):
        """
        Error raised when more than one match for a specific type of object is found in a single region file.
        Hopefully this error is never raised, it probably shouldn't!
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'MultipleMatchError has been raised'


class NoRegionsError(Exception):
    def __init__(self, *args):
        """
        Error raised when there are no appropriate regions available for an attempted analysis.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'NoRegionsError has been raised'


class NoProductAvailableError(Exception):
    def __init__(self, *args):
        """
        Error raised when requesting a product from an XGA source that hasn't yet been generated.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'NoProductAvailableError has been raised'


class ModelNotAssociatedError(Exception):
    def __init__(self, *args):
        """
        Error raised when values from a model fit that isn't associated with a particular product object.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'ModelNotAssociatedError has been raised'


class ParameterNotAssociatedError(Exception):
    def __init__(self, *args):
        """
        Error raised when a parameter is not associated with a particular model.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'ParameterNotAssociatedError has been raised'


class XSPECFitError(Exception):
    def __init__(self, *args):
        """
        This error is raised when there is a problem during an XSPEC fit instigated by XGA.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'XSPECFitError has been raised'


class RateMapPairError(Exception):
    def __init__(self, *args):
        """
        This error is raised when there is a problem with the pair of Image and Exposure map objects that
        are passed into the RateMap init.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'RateMapPairError has been raised'


class PeakConvergenceFailedError(Exception):
    def __init__(self, *args):
        """
        This error is raised when iterating peak finding fails to converge within the allowed number
        of iterations.
        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self):
        if self.message:
            return '{}'.format(self.message)
        else:
            return 'PeakConvergenceFailedError has been raised'




