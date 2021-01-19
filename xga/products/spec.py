#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 19/01/2021, 17:23. Copyright (c) David J Turner


import os
import warnings
from typing import Tuple, Union, List

import numpy as np
from astropy.units import Quantity
from fitsio import FITS, hdu
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from . import BaseProduct, BaseAggregateProduct
from ..exceptions import ModelNotAssociatedError, ParameterNotAssociatedError, XGASetIDError, NotAssociatedError
from ..utils import dict_search


class Spectrum(BaseProduct):
    """
    This class is designed to act as an interface with an X-ray spectrum, and includes various methods and attributes
    for storing and accessing fit information, as well as viewing the spectrum.
    """
    def __init__(self, path: str, rmf_path: str, arf_path: str, b_path: str, b_rmf_path: str, b_arf_path: str,
                 central_coord: Quantity, inn_rad: Quantity, out_rad: Quantity, obs_id: str, instrument: str,
                 grouped: bool, min_counts: int, min_sn: float, over_sample: int, stdout_str: str,
                 stderr_str: str, gen_cmd: str, region: bool = False):

        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd)
        self._prod_type = "spectrum"

        if os.path.exists(rmf_path):
            self._rmf = rmf_path
        else:
            self._rmf = ''
            self._usable = False
            self._why_unusable.append("RMFPathDoesNotExist")

        if os.path.exists(arf_path):
            self._arf = arf_path
        else:
            self._arf = ''
            self._usable = False
            self._why_unusable.append("ARFPathDoesNotExist")

        if os.path.exists(b_path):
            self._back_spec = b_path
        else:
            self._back_spec = ''
            self._usable = False
            self._why_unusable.append("BackSpecPathDoesNotExist")

        if os.path.exists(b_rmf_path):
            self._back_rmf = b_rmf_path
        else:
            self._back_rmf = ''
            self._usable = False
            self._why_unusable.append("BackRMFPathDoesNotExist")

        if os.path.exists(b_arf_path):
            self._back_arf = b_arf_path
        else:
            self._back_arf = ''
            self._usable = False
            self._why_unusable.append("BackARFPathDoesNotExist")

        # Storing the central coordinate of this spectrum
        self._central_coord = central_coord

        # Storing the region information
        self._inner_rad = inn_rad
        self._outer_rad = out_rad
        # And also the shape of the region
        if self._inner_rad.isscalar:
            self._shape = 'circular'
        else:
            self._shape = 'elliptical'

        try:
            self._update_spec_headers("main")
            self._update_spec_headers("back")
        except OSError:
            self._usable = False
            self._why_unusable.append("FITSIOOSError")

        self._exp = None
        self._plot_data = {}
        self._luminosities = {}
        self._count_rate = {}

        # This is specifically for fakeit runs (for cntrate - lum conversions) on the ARF/RMF
        #  associated with this Spectrum
        self._conv_factors = {}

        # This set of properties describe the configuration of evselect/specgroup during generation
        self._grouped = grouped
        self._min_counts = min_counts
        self._min_sn = min_sn
        if self._grouped and self._min_counts is not None:
            self._grouped_on = 'counts'
        elif self._grouped and self._min_sn is not None:
            self._grouped_on = 'signal to noise'
        else:
            self._grouped_on = None

        # Not to do with grouping, but this states the level of oversampling requested from evselect
        self._over_sample = over_sample

        # This describes whether this spectrum was generated directly from a region present in a region file
        self._region = region

        # Here we generate the storage key for this object, its just convenient to do it in here
        # Sets up the extra part of the storage key name depending on if grouping is enabled
        if grouped and min_counts is not None:
            extra_name = "_mincnt{}".format(min_counts)
        elif grouped and min_sn is not None:
            extra_name = "_minsn{}".format(min_sn)
        else:
            extra_name = ''

        # And if it was oversampled during generation then we need to include that as well
        if over_sample is not None:
            extra_name += "_ovsamp{ov}".format(ov=over_sample)

        spec_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}"
        if not self._region and self.inner_rad.isscalar:
            spec_storage_name = spec_storage_name.format(ra=self.central_coord[0].value,
                                                         dec=self.central_coord[1].value,
                                                         ri=self._inner_rad.value, ro=self._outer_rad.value,
                                                         gr=grouped)
        elif not self._region and not self._inner_rad.isscalar:
            inn_rad_str = 'and'.join(self._inner_rad.value.astype(str))
            out_rad_str = 'and'.join(self._outer_rad.value.astype(str))
            spec_storage_name = spec_storage_name.format(ra=self.central_coord[0].value,
                                                         dec=self.central_coord[1].value, ri=inn_rad_str,
                                                         ro=out_rad_str, gr=grouped)
        else:
            spec_storage_name = "region_grp{gr}".format(gr=grouped)

        spec_storage_name += extra_name
        # And we save the completed key to an attribute
        self._storage_key = spec_storage_name

        # This attribute is set via the property, ONLY if this spectrum is considered to be a member of a set
        #  of annular spectra. It describes which position in the set this spectrum has
        self._ann_ident = None
        # This holds a unique random identifier for the set itself, and again will only be set from outside
        self._set_ident = None

    def _update_spec_headers(self, which_spec: str):
        """
        An internal method that will 'push' the current class attributes that hold the paths to data products
        (like ARF and RMF) to the relevant spectrum file.
        :param str which_spec: A flag that tells the method whether to update the header of
         the main or background spectrum.
        """
        # This function is meant for internal use only, so I won't check that the passed-in file paths
        #  actually exist. This will have been checked already
        if which_spec == "main" and self.usable:
            with FITS(self._path, 'rw') as spec_fits:
                spec_fits[1].write_key("RESPFILE", self._rmf)
                spec_fits[1].write_key("ANCRFILE", self._arf)
                spec_fits[1].write_key("BACKFILE", self._back_spec)
                spec_fits[0].write_key("RESPFILE", self._rmf)
                spec_fits[0].write_key("ANCRFILE", self._arf)
                spec_fits[0].write_key("BACKFILE", self._back_spec)
        elif which_spec == "back" and self.usable:
            with FITS(self._back_spec, 'rw') as spec_fits:
                spec_fits[1].write_key("RESPFILE", self._back_rmf)
                spec_fits[1].write_key("ANCRFILE", self._back_arf)
                spec_fits[0].write_key("RESPFILE", self._back_rmf)
                spec_fits[0].write_key("ANCRFILE", self._back_arf)

    @property
    def path(self) -> str:
        """
        This method returns the path to the spectrum file of this object.
        :return: The path to the spectrum file associated with this object.
        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, new_path: str):
        """
        This setter updates the path to the spectrum file, and then updates that file with the current values of
        the RMF, ARF, and background spectrum paths. WARNING: This does permanently alter the file, so use your
        own spectrum file with caution.
        :param str new_path: The updated path to the spectrum file.
        """
        if os.path.exists(new_path):
            self._path = new_path
            # Call this here because it'll replace any existing arf and rmf file paths with the ones
            #  currently loaded in the instance of this object.
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new spectrum file does not exist")

    @property
    def rmf(self) -> str:
        """
        This method returns the path to the RMF file of the main spectrum of this object.
        :return: The path to the RMF file associated with the main spectrum of this object.
        :rtype: str
        """
        return self._rmf

    @rmf.setter
    def rmf(self, new_path: str):
        """
        This setter updates the path to the main RMF file, then writes that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!
        :param str new_path: The path to the new RMF file.
        """
        if os.path.exists(new_path):
            self._rmf = new_path
            # Push to the actual file
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new RMF file does not exist")

    @property
    def arf(self) -> str:
        """
        This method returns the path to the ARF file of the main spectrum of this object.
        :return: The path to the ARF file associated with the main spectrum of this object.
        :rtype: str
        """
        return self._arf

    @arf.setter
    def arf(self, new_path: str):
        """
        This setter updates the path to the main ARF file, then writes that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!
        :param str new_path: The path to the new ARF file.
        """
        if os.path.exists(new_path):
            self._arf = new_path
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new ARF file does not exist")

    @property
    def background(self) -> str:
        """
        This method returns the path to the background spectrum.
        :return: Path of the background spectrum.
        :rtype: str
        """
        return self._back_spec

    @background.setter
    def background(self, new_path: str):
        """
        This method is the setter for the background spectrum. It can be used to change the background
        spectrum file associated with this object, and will write that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!
        :param str new_path: The path to the new background spectrum.
        """
        if os.path.exists(new_path):
            self._back_spec = new_path
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new background spectrum file does not exist")

    @property
    def background_rmf(self) -> str:
        """
        This method returns the path to the background spectrum's RMF file.
        :return: The path the the background spectrum's RMF.
        :rtype: str
        """
        return self._back_rmf

    @background_rmf.setter
    def background_rmf(self, new_path: str):
        """
        This setter method will change the RMF associated with the background spectrum, then write
        that change to the background spectrum file.
        :param str new_path: The path to the background spectrum's new RMF.
        """
        if os.path.exists(new_path):
            self._back_rmf = new_path
            self._update_spec_headers("back")
        else:
            raise FileNotFoundError("That new background RMF file does not exist")

    @property
    def background_arf(self) -> str:
        """
        This method returns the path to the background spectrum's ARF file.
        :return: The path the the background spectrum's ARF.
        :rtype: str
        """
        return self._back_arf

    @background_arf.setter
    def background_arf(self, new_path: str):
        """
        This setter method will change the ARF associated with the background spectrum, then write
        that change to the background spectrum file.
        :param str new_path: The path to the background spectrum's new ARF.
        """
        if os.path.exists(new_path):
            self._back_arf = new_path
            self._update_spec_headers("back")
        else:
            raise FileNotFoundError("That new background ARF file does not exist")

    @property
    def storage_key(self) -> str:
        """
        This property returns the storage key which this object assembles to place the Spectrum in
        an XGA source's storage structure. The key is based on the properties of the spectrum, and
        some of the configuration options, and is basically human readable.

        :return: String storage key.
        :rtype: str
        """
        return self._storage_key

    @property
    def central_coord(self) -> Quantity:
        """
        This property provides the central coordinates (RA-Dec) of the region that this spectrum
        was generated from.

        :return: Astropy quantity object containing the central coordinate in degrees.
        :rtype: Quantity
        """
        return self._central_coord

    @property
    def shape(self) -> str:
        """
        Returns the shape of the outer edge of the region this spectrum was generated from.

        :return: The shape (either circular or elliptical).
        :rtype: str
        """
        return self._shape

    @property
    def inner_rad(self) -> Quantity:
        """
        Gives the inner radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this spectrum was generated.

        :return: The inner radius(ii) of the region.
        :rtype: Quantity
        """
        return self._inner_rad

    @property
    def outer_rad(self):
        """
        Gives the outer radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this spectrum was generated.

        :return: The outer radius(ii) of the region.
        :rtype: Quantity
        """
        return self._outer_rad

    @property
    def grouped(self) -> bool:
        """
        A property stating whether SAS was told to group this spectrum during generation or not.

        :return: Boolean variable describing whether the spectrum is grouped or not
        :rtype: bool
        """
        return self._grouped

    @property
    def grouped_on(self) -> str:
        """
        A property stating what metric this spectrum was grouped on.

        :return: String representation of the metric this spectrum was grouped on (None if not grouped).
        :rtype: str
        """
        return self._grouped_on

    @property
    def min_counts(self) -> int:
        """
        A property stating the minimum number of counts allowed in a grouped channel.

        :return: The integer minimum number of counts per grouped channel (if this spectrum was grouped on
            minimum numbers of counts).
        :rtype: int
        """
        return self._min_counts

    @property
    def min_sn(self) -> Union[float, int]:
        """
        A property stating the minimum signal to noise allowed in a grouped channel.

        :return: The minimum signal to noise per grouped channel (if this spectrum was grouped on
            minimum signal to noise).
        :rtype: Union[float, int]
        """
        return self._min_sn

    @property
    def over_sample(self) -> float:
        """
        A property string stating the amount of oversampling applied by evselect during the spectrum
        generation process.

        :return: Oversampling applied during generation
        :rtype: float
        """
        return self._over_sample

    @property
    def region(self) -> bool:
        """
        This property states whether this spectrum was generated directly from a region file
        region or not. If true then this isn't from any arbitrary radii or an overdensity radius, but
        instead directly from a source finder.

        :return: A boolean flag describing if this is a region spectrum or not.
        :rtype: bool
        """
        return self._region

    @property
    def annulus_ident(self) -> int:
        """
        This property returns the integer identifier of which annulus in a set this Spectrum is, if it
        is part of a set.

        :return: Integer annulus identifier, None if not part of a set.
        :rtype: object
        """
        return self._ann_ident

    @annulus_ident.setter
    def annulus_ident(self, new_ident: int):
        """
        This property sets the annulus identifier of this object.

        :param int new_ident: The annulus integer identifier of this spectrum.
        """
        if not isinstance(new_ident, int):
            raise TypeError("Spectrum annulus identifiers may ONLY be positive integers")
        self._ann_ident = new_ident

    @property
    def set_ident(self) -> int:
        """
        This property returns the random id of the spectrum set this is a part of.

        :return: Set identifier, None if not part of a set.
        :rtype: object
        """
        return self._set_ident

    @set_ident.setter
    def set_ident(self, new_ident: int):
        """
        This property sets the set identifier of this object.

        :param int new_ident: The set identifier of this spectrum.
        """
        if not isinstance(new_ident, int):
            raise TypeError("Spectrum set identifiers may ONLY be positive integers")
        self._set_ident = new_ident

    @property
    def exposure(self) -> Quantity:
        """
        Property that returns the spectrum exposure time used by XSPEC.
        :return: Spectrum exposure time.
        :rtype: Quantity
        """
        if self._exp is None:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this Spectrum")
        else:
            exp = Quantity(self._exp, 's')

        return exp

    def add_fit_data(self, model: str, tab_line, plot_data: hdu.table.TableHDU):
        """
        Method that adds information specific to a spectrum from an XSPEC fit to this object. This includes
        individual spectrum exposure and count rate, as well as calculated luminosities, and plotting
        information for data and model.
        :param str model: String representation of the XSPEC model fitted to the data.
        :param tab_line: The line of the SPEC_INFO table produced by xga_extract.tcl that is relevant to this
        spectrum object.
        :param hdu.table.TableHDU plot_data: The PLOT{N} table in the file produced by xga_extract.tcl that is
        relevant to this spectrum object.
        """
        # This stores the exposure time that XSPEC uses for this specific spectrum.
        if self._exp is None:
            self._exp = float(tab_line["EXPOSURE"])

        # This is the count rate and error for this spectrum.
        self._count_rate[model] = [float(tab_line["COUNT_RATE"]), float(tab_line["COUNT_RATE_ERR"])]

        # Searches for column headers with 'Lx' in them (this has to be dynamic as the user can calculate
        #  luminosity in as many bands as they like)
        lx_inds = np.where(np.char.find(tab_line.dtype.names, "Lx") == 0)[0]
        lx_cols = np.array(tab_line.dtype.names)[lx_inds]

        # Constructs a dictionary of luminosities and their errors for the different energy bands
        #  in this XSPEC fit.
        lx_dict = {}
        for col in lx_cols:
            lx_info = col.split("_")
            if lx_info[2][-1] == "-" or lx_info[2][-1] == "+":
                en_band = "bound_{l}-{u}".format(l=lx_info[1], u=lx_info[2][:-1])
                err_type = lx_info[-1][-1]
            else:
                en_band = "bound_{l}-{u}".format(l=lx_info[1], u=lx_info[2])
                err_type = ""

            if en_band not in lx_dict:
                lx_dict[en_band] = [0, 0, 0]

            if err_type == "":
                lx_dict[en_band][0] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")
            elif err_type == "-":
                lx_dict[en_band][1] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")
            elif err_type == "+":
                lx_dict[en_band][2] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")

        self._luminosities[model] = lx_dict

        self._plot_data[model] = {"x": plot_data["X"][:], "x_err": plot_data["XERR"][:],
                                  "y": plot_data["Y"][:], "y_err": plot_data["YERR"][:],
                                  "model": plot_data["YMODEL"][:]}

    def get_luminosities(self, model: str, lo_en: Quantity = None, hi_en: Quantity = None):
        """
        Returns the luminosities measured for this spectrum from a given model.
        :param model: Name of model to fetch luminosities for.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :return: Luminosity measurement, either for all energy bands, or the one requested with the energy
        limit parameters. Luminosity measurements are presented as three column numpy arrays, with column 0
        being the value, column 1 being err-, and column 2 being err+.
        """
        # Checking the input energy limits are valid, and assembles the key to look for lums in those energy
        #  bounds. If the limits are none then so is the energy key
        if lo_en is not None and hi_en is not None and lo_en > hi_en:
            raise ValueError("The low energy limit cannot be greater than the high energy limit")
        elif lo_en is not None and hi_en is not None:
            en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        else:
            en_key = None

        # Checks that the requested region, model and energy band actually exist
        if len(self._luminosities) == 0:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with {s}".format(s=self.src_name))
        elif model not in self._luminosities:
            av_mods = ", ".join(self._luminosities.keys())
            raise ModelNotAssociatedError("{0} has not been fitted to this spectrum; "
                                          "available models are {1}".format(model, av_mods))
        elif en_key is not None and en_key not in self._luminosities[model]:
            av_bands = ", ".join([en.split("_")[-1] + "keV" for en in self._luminosities[model].keys()])
            raise ParameterNotAssociatedError("{l}-{u}keV was not an energy band for the fit with {m}; available "
                                              "energy bands are {b}".format(l=lo_en.to("keV").value,
                                                                            u=hi_en.to("keV").value,
                                                                            m=model, b=av_bands))

        if en_key is None:
            return self._luminosities[model]
        else:
            return self._luminosities[model][en_key]

    def get_rate(self, model: str) -> Quantity:
        """
        Fetches the count rate for a particular model fitted to this spectrum.
        :param model: The model to fetch count rate for.
        :return: Count rate in counts per second.
        :rtype: Quantity
        """
        if model not in self._count_rate:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this Spectrum")
        else:
            rate = Quantity(self._count_rate[model], 'ct/s')

        return rate

    # TODO Should this take parameter values as arguments too? - It definitely should
    def add_conv_factors(self, lo_ens: np.ndarray, hi_ens: np.ndarray, rates: np.ndarray,
                         lums: np.ndarray, model: str):
        """
        Method used to store countrate to luminosity conversion factors derived from fakeit spectra, as well as
        the actual countrate and luminosity measured in case the user wants to create a combined factor for multiple
        observations
        :param np.ndarray lo_ens: A numpy array of string representations of the lower energy bounds for the cntrate
        and luminosity measurements.
        :param np.ndarray hi_ens: A numpy array of string representations of the upper energy bounds for the cntrate
        and luminosity measurements.
        :param np.ndarray rates: A numpy array of the rates measured for this arf/rmf combination for the energy
        ranges specified in lo_ens and hi_end.
        :param np.ndarray lums: A numpy array of the luminosities measured for this arf/rmf combination
        for the energy ranges specified in lo_ens and hi_end.
        :param str model: The name of the model used to calculate this factor.
        """
        for row_ind, lo_en in enumerate(lo_ens):
            # Define the key with energy information under which to store this information
            hi_en = hi_ens[row_ind]
            en_key = "bound_{l}-{u}".format(l=lo_en, u=hi_en)

            # Split out the rate and lum for this particular set of energy limits
            rate = Quantity(rates[row_ind], "ct/s")
            lum = Quantity(lums[row_ind], "10^44 erg/s")

            # Will be storing the individual components, but will also store the factor for this spectrum
            factor = lum / rate

            if model not in self._conv_factors:
                self._conv_factors[model] = {}

            self._conv_factors[model][en_key] = {"rate": rate, "lum": lum, "factor": factor}

    def get_conv_factor(self, lo_en: Quantity, hi_en: Quantity, model: str) -> Tuple[Quantity, Quantity, Quantity]:
        """
        Retrieves a conversion factor between count rate and luminosity for a given energy range, if one
        has been calculated.
        :param Quantity lo_en: The lower energy bound for the desired conversion factor.
        :param Quantity hi_en: The upper energy bound for the desired conversion factor.
        :param str model: The model used to generate the desired conversion factor.
        :return: The conversion factor, luminosity, and rate for the supplied model-energy combination.
        :rtype: Tuple[Quantity, Quantity, Quantity]
        """
        en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        if model not in self._conv_factors:
            mods = ", ".join(list(self._conv_factors.keys()))
            raise ModelNotAssociatedError("{0} is not associated with this spectrum, only {1} "
                                          "are available.".format(model, mods))
        elif en_key not in self._conv_factors[model]:
            raise ParameterNotAssociatedError("The conversion factor for {m} in {l}-{u}keV has not been "
                                              "calculated".format(m=model, l=lo_en.to("keV").value,
                                                                  u=hi_en.to("keV").value))

        rel_vals = self._conv_factors[model][en_key]
        return rel_vals["factor"], rel_vals["lum"], rel_vals["rate"]

    def view(self, lo_en: Quantity = Quantity(0.0, "keV"), hi_en: Quantity = Quantity(30.0, "keV"),
             figsize: Tuple = (8, 6)):
        """
        Very simple method to plot the data/models associated with this Spectrum object,
        between certain energy limits.

        :param Quantity lo_en: The lower energy limit from which to plot the spectrum.
        :param Quantity hi_en: The upper energy limit to plot the spectrum to.
        :param Tuple figsize: The desired size of the output figure.
        """
        if lo_en > hi_en:
            raise ValueError("hi_en cannot be greater than lo_en")
        else:
            lo_en = lo_en.to("keV").value
            hi_en = hi_en.to("keV").value

        if len(self._plot_data.keys()) != 0:
            # Create figure object
            plt.figure(figsize=figsize)

            # Set the plot up to look nice and professional.
            ax = plt.gca()
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

            # Set the title with all relevant information about the spectrum object in it
            plt.title("{n} - {o}{i} {r} Spectrum".format(n=self.src_name, o=self.obs_id, i=self.instrument.upper(),
                                                         r=self.reg_type))
            for mod_ind, mod in enumerate(self._plot_data):
                x = self._plot_data[mod]["x"]
                # If the defaults are left, just update them to the min and max of the dataset
                #  to avoid unsightly gaps at the sides of the plot
                if lo_en == 0.:
                    lo_en = x.min()
                if hi_en == 30.0:
                    hi_en = x.max()

                # Cut the x dataset to just the energy range we want
                plot_x = x[(x > lo_en) & (x < hi_en)]

                if mod_ind == 0:
                    # Read out the data just for line length reasons
                    # Make the cuts based on energy values supplied to the view method
                    plot_y = self._plot_data[mod]["y"][(x > lo_en) & (x < hi_en)]
                    plot_xerr = self._plot_data[mod]["x_err"][(x > lo_en) & (x < hi_en)]
                    plot_yerr = self._plot_data[mod]["y_err"][(x > lo_en) & (x < hi_en)]
                    plot_mod = self._plot_data[mod]["model"][(x > lo_en) & (x < hi_en)]

                    plt.errorbar(plot_x, plot_y, xerr=plot_xerr, yerr=plot_yerr, fmt="k+", label="data", zorder=1)
                else:
                    # Don't want to re-plot data points as they should be identical, so if there is another model
                    #  only it will be plotted
                    plot_mod = self._plot_data[mod]["model"][(x > lo_en) & (x < hi_en)]

                # The model line is put on
                plt.plot(plot_x, plot_mod, label=mod, linewidth=1.5)

            # Generate the legend for the data and model(s)
            plt.legend(loc="best")

            # Ensure axis is limited to the chosen energy range
            plt.xlim(lo_en, hi_en)

            plt.xlabel("Energy [keV]")
            plt.ylabel("Normalised Counts s$^{-1}$ keV$^{-1}$")

            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

            plt.tight_layout()
            # Display the spectrum
            plt.show()

            # Wipe the figure
            plt.close("all")

        else:
            warnings.warn("There are no XSPEC fits associated with this Spectrum, you can't view it.")


class AnnularSpectra(BaseAggregateProduct):
    """
    A class designed to hold a set of XGA spectra generated in concentric, circular annuli.
    """
    def __init__(self, spectra: List[Spectrum]):
        """
        The init method for the AnnularSpectrum class, performs checks and organises the spectra which
        have been passed in, for easy retrieval.

        :param List[Spectrum] spectra: A list of XGA spectrum objects which make up this set.
        """
        super().__init__([s.path for s in spectra], 'spectrum', "combined", "combined")

        # There shouldn't be any way this can happen, but it doesn't hurt to check that all of the spectra
        #  have the same set ID
        set_idents = set([s.set_ident for s in spectra])
        if len(set_idents) != 1:
            raise XGASetIDError("You have passed spectra that have set IDs that do not match")

        # Here I run through all the spectra and access their annulus_ident property, that way we can determine how
        #  many annuli there are and start storing spectra appropriately
        self._num_ann = len(set([s.annulus_ident for s in spectra]))

        # While the official ObsID and Instrument of this product are 'combined', I do still
        #  want to know which ObsIDs and instruments the spectra belong to
        inst_dict = {o: [] for o in [s.obs_id for s in spectra]}
        for s in spectra:
            if s.instrument not in inst_dict[s.obs_id]:
                inst_dict[s.obs_id].append(s.instrument)

        # The same idea as the source.instruments dictionary
        self._instruments = inst_dict

        # All the radii will be in degrees, but I'll grab it dynamically anyway
        self._rad_unit = spectra[0].inner_rad.unit

        # I want to grab the radii out of the spectra, then put them in order, just so I have them
        radii = sorted(list(set([s.inner_rad for s in spectra] + [s.outer_rad for s in spectra])))
        self._radii = Quantity([r.value for r in radii], self._rad_unit)

        # Finally storing the spectra inside the product, though with multiple layers of products
        # This sets up the component products dictionary, allowing for the separated storage of
        #  spectra from different ObsIDs
        self._component_products = {ai: {o: {i: None for i in self._instruments[o]} for o in self.obs_ids}
                                    for ai in range(self._num_ann)}
        self._component_products = {o: {i: {ai: None for ai in range(self._num_ann)}
                                        for i in self._instruments[o]} for o in self.obs_ids}
        # And putting the spectra in their place
        for s in spectra:
            self._component_products[s.obs_id][s.instrument][s.annulus_ident] = s

        self._all_usable = all(s.usable for s in self.all_spectra)

    @property
    def num_annuli(self) -> int:
        """
        A property getter for the number of annular spectra.
        :return: The number of annular spectra associated with this product.
        :rtype: int
        """
        return self._num_ann

    def background(self, obs_id: str, inst: str) -> str:
        """
        This method returns the path to the background spectrum for a particular ObsID and
        instrument. It is the background associated with the outermost annulus of this object.

        :param str obs_id: The ObsID to get the background spectrum for.
        :param str inst: The instrument to get the background spectrum for.
        :return: Path of the background spectrum.
        :rtype: str
        """
        return self.get_spectra(self._num_ann-1, obs_id, inst).background

    def background_rmf(self, obs_id: str, inst: str) -> str:
        """
        This method returns the path to the background spectrum's RMF for a particular ObsID and
        instrument. It is the RMF of the background associated with the outermost annulus of this object.

        :param str obs_id: The ObsID to get the background spectrum's RMF for.
        :param str inst: The instrument to get the background spectrum' RMF for.
        :return: Path of the background spectrum RMF.
        :rtype: str
        """
        return self.get_spectra(self._num_ann-1, obs_id, inst).background_rmf

    def background_arf(self, obs_id: str, inst: str) -> str:
        """
        This method returns the path to the background spectrum's ARF for a particular ObsID and
        instrument. It is the ARF of the background associated with the outermost annulus of this object.

        :param str obs_id: The ObsID to get the background spectrum's ARF for.
        :param str inst: The instrument to get the background spectrum' ARF for.
        :return: Path of the background spectrum ARF.
        :rtype: str
        """
        return self.get_spectra(self._num_ann - 1, obs_id, inst).background_arf

    @property
    def obs_ids(self) -> list:
        """
        A property of this spectrum set that details which ObsIDs have contributed spectra to this object.

        :return: A list of ObsIDs.
        containing instruments associated with those ObsIDs.
        :rtype: dict
        """
        return list(self._instruments.keys())

    @property
    def instruments(self) -> dict:
        """
        A property of this spectrum set that details which ObsIDs and instruments have contributed spectra
        to this object.

        :return: A dictionary of lists, with the top level keys being ObsIDs, and the lists
        containing instruments associated with those ObsIDs.
        :rtype: dict
        """
        return self._instruments

    def get_spectra(self, annulus_ident, obs_id: str = None, inst: str = None) -> Union[List[Spectrum], Spectrum]:
        """
        This is the getter for the spectra stored in the AnnularSpectra data storage structure. They can
        be retrieved based on annulus identifier, ObsID, and instrument.

        :param int annulus_ident: The annulus identifier to retrieve spectra for.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied.
        :param str inst: Optionally, a specific instrument to search for can be supplied.
        :return: List of matching spectra, or just a Spectrum object if one match is found.
        :rtype: Union[List[Spectrum], Spectrum]
        """
        def unpack_list(to_unpack: list):
            """
            A recursive function to go through every layer of a nested list and flatten it all out. It
            doesn't return anything because to make life easier the 'results' are appended to a variable
            in the namespace above this one.

            :param list to_unpack: The list that needs unpacking.
            """
            # Must iterate through the given list
            for entry in to_unpack:
                # If the current element is not a list then all is chill, this element is ready for appending
                # to the final list
                if not isinstance(entry, list):
                    out.append(entry)
                else:
                    # If the current element IS a list, then obviously we still have more unpacking to do,
                    # so we call this function recursively.
                    unpack_list(entry)

        if annulus_ident not in np.array(range(0, self._num_ann)):
            ann_str = ", ".join(np.array(range(0, self._num_ann)).astype(str))
            raise IndexError("{i} is not an annulus ID associated with this AnnularSpectra object. "
                             "Allowed annulus IDs are; ".format(i=annulus_ident, a=ann_str))
        elif obs_id not in self._component_products and obs_id is not None:
            raise NotAssociatedError("{0} is not associated with this AnnularSpectra.".format(obs_id))
        elif (obs_id is not None and obs_id in self._component_products) and \
                (inst is not None and inst not in self._component_products[obs_id]):
            raise NotAssociatedError("Instrument {1} is not associated with {0}".format(obs_id, inst))

        matches = []
        for match in dict_search(annulus_ident, self._component_products):
            out = []
            unpack_list(match)
            if (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None):
                matches.append(out[-1])

        # Here I only return the object if one match was found
        if len(matches) == 1:
            matches = matches[0]
        return matches

    @property
    def all_spectra(self) -> List[Spectrum]:
        """
        Simple extra wrapper for get_spectra that allows the user to retrieve every single spectrum associated
        with this AnnularSpectra instance, for all annulus IDs.

        :return: A list of every single spectrum associated with this object.
        :rtype: List[Spectrum]
        """
        all_spec = []
        for ann_i in range(self._num_ann):
            all_spec += self.get_spectra(ann_i)

        return all_spec








