#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 25/08/2025, 13:58. Copyright (c) The Contributors


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


class XSPECNotFoundError(Exception):
    def __init__(self, *args):
        """
        Exception raised if XSPEC can not be found on the system.

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
            return 'XSPECNotFoundError has been raised'


# I do not know if I will keep this as is or expand out into different errors
# The trouble is there are many hundreds of possible SAS errors, and I don't know if I
# want a class for all of them
class SASGenerationError(Exception):
    def __init__(self, *args):
        """
        Exception raised if an error is found to have occured during a run of a part of the SAS software.

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


class SASInputInvalid(Exception):
    def __init__(self, *args):
        """
        This error is raised when a user provides an invalid input to a SAS function.

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
            return 'SASInputInvalid has been raised'


class NotPSFCorrectedError(Exception):
    def __init__(self, *args):
        """
        Raised when the user tries to set PSF deconvolution properties of an Image product, but the
        psf correction flag indicates that the product is not deconvolved.

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
            return 'NotPSFCorrectedError has been raised'


class IncompatibleProductError(Exception):
    def __init__(self, *args):
        """
        Raised when products are used together that do not have matching ObsID and instrument
        values, or when they were not generated at the same coordinates. For instance when you try to
        re-sample a PSF with an Image object from a different observation and instrument.

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
            return 'IncompatibleProductError has been raised'


class XGAFitError(Exception):
    def __init__(self, *args):
        """
        Raised when there is an issue with a fit that XGA is trying to perform.

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
            return 'XGAFitError has been raised'


class XGAInvalidModelError(Exception):
    def __init__(self, *args):
        """
        Raised when the user is trying to fit a model that is not appropriate to the data.

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
            return 'XGAInvalidModelError has been raised'


class XGAFunctionConversionError(Exception):
    def __init__(self, *args):
        """
        Raised when an attempt to convert an XGA model function (or a custom function supplied by the user) to the
        standard required by scipy's orthogonal distance regression failed.

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
            return 'XGAFunctionConversionError has been raised'


class XGAOptionalDependencyError(Exception):
    def __init__(self, *args):
        """
        Raised when an optional XGA dependency hasn't been installed, but the feature that needs it has been
        activated.

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
            return 'XGAOptionalDependencyError has been raised'


class XGASetIDError(Exception):
    def __init__(self, *args):
        """
        Raised when something is attempting to use annular spectra (which are all part of spectrum sets) together,
        but they have different set IDs.
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
            return 'XGASetIDError has been raised'


class XGAPoorDataError(Exception):
    def __init__(self, *args):
        """
        Raised when the data aren't of sufficient quality to complete an operation.
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
            return 'XGAPoorDataError has been raised'


class InvalidProductError(Exception):
    def __init__(self, *args):
        """
        Raised when there is a signficiant problem with a data product.
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
            return 'InvalidProductError has been raised'


class NotSampleMemberError(Exception):
    def __init__(self, *args):
        """
        Raised when a feature reserved for sources that belong to samples is used, and the source is independent
        of a sample.
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
            return 'NotSampleMemberError has been raised'


class XGADeveloperError(Exception):
    def __init__(self, *args):
        """
        Raised when an error has occurred that needs the attention of developers.
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
            return 'XGADeveloperError has been raised'

class FitConfNotAssociatedError(Exception):
    def __init__(self, *args):
        """
        Raised when a supplied fit configuration isn't associated with the source, model, or spectrum.
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
            return 'FitConfNotAssociatedError has been raised'

class NoTelescopeDataError(Exception):
    def __init__(self, *args):
        """
        Raised when a part of XGA that directly accesses telescope data is used, but there are no relevant
        telescopes setup in the configuration file.
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
            return 'NoTelescopeDataError has been raised'


class InvalidTelescopeError(Exception):
    def __init__(self, *args):
        """
        Raised when the name of a telescope has been passed to a function, but it is not recognised or supported.
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
            return 'InvalidTelescopeError has been raised'