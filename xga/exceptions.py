#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/13/26, 5:20 PM. Copyright (c) The Contributors.
"""This module defines custom exceptions for XGA."""


class eSASSInputInvalid(Exception):
    """Exception raised when a user provides an invalid input to an eSASS function."""

    def __init__(self, *args: object) -> None:
        """
        This error is raised when a user provides an invalid input to an eSASS function.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "eSASSInputInvalid has been raised"


class HeasoftError(Exception):
    """Exception raised for unexpected output from HEASoft calls, currently all encompassing."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised for unexpected output from HEASoft calls, currently all encompassing.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "HeasoftError has been raised"


class SASNotFoundError(Exception):
    """Exception raised if the XMM Scientific Analysis System cannot be found on the system."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if the XMM Scientific Analysis System can not be found on the system.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "SASNotFoundError has been raised"


class eSASSNotFoundError(Exception):
    """Exception raised if the eROSITA Science Analysis Software System cannot be found on the system."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if the eROSITA Science Analysis Software
        System can not be found on the system.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "eSASSNotFoundError has been raised"


class CIAONotFoundError(Exception):
    """Exception raised if Chandra CIAO cannot be found on the system."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if the Chandra CIAO can not be found on the system.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "CIAONotFoundError has been raised"


class CALDBNotFoundError(Exception):
    """Exception raised if the calibration files CALDB cannot be found on the system."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if the Calibration files CALDB can not be found on the system.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "CALDBNotFoundError has been raised"


class XSPECNotFoundError(Exception):
    """Exception raised if XSPEC cannot be found on the system."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if XSPEC can not be found on the system.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "XSPECNotFoundError has been raised"


class ProductGenerationError(Exception):
    """Exception raised if an error occurs during a run of backend product-generation software."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if an error is found to have occurred during a run of backend software that generates
        products for a telescope supported by XGA - equivalent to the now-removed SASGenerationError, but more general.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "ProductGenerationError has been raised"


class UnknownCommandlineError(Exception):
    """Exception raised if an unlinked error occurs during a run of backend SAS software."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if an error is found to have occurred during a run of a part
        of the SAS software, but it cannot be linked to a SAS function.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "A generic UnknownCommandlineError has been raised"


class TelescopeNotAssociatedError(Exception):
    """Exception raised when a telescope-specific function is called on a source or sample lacking that telescope."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised if the user tries to run a telescope specific function (e.g. a SAS function for XMM, or an
        eSASS function for eROSITA) and the source or sample has no data for that telescope associated with it.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "TelescopeNotAssociatedError has been raised"


class FailedProductError(Exception):
    """Exception raised when accessing data from an object wrapping a product that failed to generate properly."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "FailedProductError has been raised."


class XGAConfigError(Exception):
    """Exception raised for flawed XGA configuration files."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised for flawed XGA config files.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "XGAConfig has been raised"


class NoMatchFoundError(Exception):
    """Exception raised when source coordinates cannot be matched to any observation."""

    def __init__(self, *args: object) -> None:
        """
        Exception raised when source ra and dec coordinates can't be made to match to any XMM observation.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "NoMatchFoundError has been raised"


class NotAssociatedError(Exception):
    """Exception raised when a given telescope or ObsID is not associated with a source object."""

    def __init__(self, *args: object) -> None:
        """
        Error raised when a given telescope or ObsID is not associated with a source object.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "NotAssociatedError has been raised"


class UnknownProductError(Exception):
    """Exception raised when attempting to write an unknown product type to an XGA source."""

    def __init__(self, *args: object) -> None:
        """
        Error raised when there is an attempt to write an unknown XMM product type to an XGA source.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "UnknownProductTypeError has been raised"


class NoValidObservationsError(Exception):
    """Exception raised when an observation match is found, but the necessary files cannot be located."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "NoValidObservationsError has been raised"


class MultipleMatchError(Exception):
    """Exception raised when more than one match for a specific type of object is found in a single region file."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "MultipleMatchError has been raised"


class NoRegionsError(Exception):
    """Exception raised when there are no appropriate regions available for an attempted analysis."""

    def __init__(self, *args: object) -> None:
        """
        Error raised when there are no appropriate regions available for an attempted analysis.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "NoRegionsError has been raised"


class NoProductAvailableError(Exception):
    """Exception raised when requesting a product from an XGA source that has not yet been generated."""

    def __init__(self, *args: object) -> None:
        """
        Error raised when requesting a product from an XGA source that hasn't yet been generated.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "NoProductAvailableError has been raised"


class ModelNotAssociatedError(Exception):
    """Exception raised when requesting values from a model fit that is not associated with a product object."""

    def __init__(self, *args: object) -> None:
        """
        Error raised when values from a model fit that isn't associated with a particular product object.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "ModelNotAssociatedError has been raised"


class ParameterNotAssociatedError(Exception):
    """Exception raised when a parameter is not associated with a particular model."""

    def __init__(self, *args: object) -> None:
        """
        Error raised when a parameter is not associated with a particular model.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "ParameterNotAssociatedError has been raised"


class XSPECFitError(Exception):
    """Exception raised when there is a problem during an XSPEC fit instigated by XGA."""

    def __init__(self, *args: object) -> None:
        """
        This error is raised when there is a problem during an XSPEC fit instigated by XGA.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "XSPECFitError has been raised"


class RateMapPairError(Exception):
    """Exception raised when there is a problem with the Image and ExpMap pair passed to RateMap."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "RateMapPairError has been raised"


class PeakConvergenceFailedError(Exception):
    """Exception raised when iterative peak finding fails to converge within the allowed iterations."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "PeakConvergenceFailedError has been raised"


class SASInputInvalid(Exception):
    """Exception raised when a user provides an invalid input to a SAS function."""

    def __init__(self, *args: object) -> None:
        """
        This error is raised when a user provides an invalid input to a SAS function.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "SASInputInvalid has been raised"


class NotPSFCorrectedError(Exception):
    """Exception raised when setting PSF deconvolution properties on an Image that is not PSF-corrected."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "NotPSFCorrectedError has been raised"


class IncompatibleProductError(Exception):
    """Exception raised when products used together do not have matching ObsID, instrument, or coordinates."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "IncompatibleProductError has been raised"


class XGAFitError(Exception):
    """Exception raised when there is an issue with a fit that XGA is trying to perform."""

    def __init__(self, *args: object) -> None:
        """
        Raised when there is an issue with a fit that XGA is trying to perform.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "XGAFitError has been raised"


class XGAInvalidModelError(Exception):
    """Exception raised when the user is trying to fit a model that is not appropriate to the data."""

    def __init__(self, *args: object) -> None:
        """
        Raised when the user is trying to fit a model that is not appropriate to the data.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "XGAInvalidModelError has been raised"


class XGAFunctionConversionError(Exception):
    """Exception raised when converting a model function to the format required by scipy ODR fails."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "XGAFunctionConversionError has been raised"


class XGAOptionalDependencyError(Exception):
    """Exception raised when a feature requiring an uninstalled optional XGA dependency is activated."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "XGAOptionalDependencyError has been raised"


class XGASetIDError(Exception):
    """Exception raised when attempting to use annular spectra together that have different set IDs."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "XGASetIDError has been raised"


class XGAPoorDataError(Exception):
    """Exception raised when the data are not of sufficient quality to complete an operation."""

    def __init__(self, *args: object) -> None:
        """
        Raised when the data aren't of sufficient quality to complete an operation.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "XGAPoorDataError has been raised"


class InvalidProductError(Exception):
    """Exception raised when there is a significant problem with a data product."""

    def __init__(self, *args: object) -> None:
        """
        Raised when there is a significant problem with a data product.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "InvalidProductError has been raised"


class NotSampleMemberError(Exception):
    """Exception raised when a sample-only feature is used on a source that does not belong to a sample."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "NotSampleMemberError has been raised"


class NoTelescopeDataError(Exception):
    """Exception raised when accessing telescope data when no relevant telescopes are configured."""

    def __init__(self, *args: object) -> None:
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

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "NoTelescopeDataError has been raised"


class InvalidTelescopeError(Exception):
    """Exception raised when an unrecognised or unsupported telescope name is passed to a function."""

    def __init__(self, *args: object) -> None:
        """
        Raised when the name of a telescope has been passed to a function, but it is not recognised or supported.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "InvalidTelescopeError has been raised"


class XGADeveloperError(Exception):
    """Exception raised when an error has occurred that needs the attention of developers."""

    def __init__(self, *args: object) -> None:
        """
        Raised when an error has occurred that needs the attention of developers.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "XGADeveloperError has been raised"


class ProductNotUsableError(Exception):
    """Exception raised when a product instance is unusable for reasons in not_usable_reasons."""

    def __init__(self, *args: object) -> None:
        """
        Raised when a product instance is unusable, for reasons specified in the
        not_usable_reasons property.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message} "
        else:
            return "ProductNotUsableError has been raised"


class FitConfNotAssociatedError(Exception):
    """Exception raised when a supplied fit configuration is not associated with the source, model, or spectrum."""

    def __init__(self, *args: object) -> None:
        """
        Raised when a supplied fit configuration isn't associated with the source, model, or spectrum.

        :param expression:
        :param message:
        """
        if args:
            self.message = args[0]
        else:
            self.message = None

    def __str__(self) -> str:
        """Return the string representation of this exception."""
        if self.message:
            return f"{self.message}"
        else:
            return "FitConfNotAssociatedError has been raised"
