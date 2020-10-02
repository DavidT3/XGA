#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 24/09/2020, 11:53. Copyright (c) David J Turner


import os
import warnings
from typing import Tuple

import numpy as np
from astropy.units import Quantity
from fitsio import FITS, hdu
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter

from . import BaseProduct, BaseAggregateProduct
from ..exceptions import ModelNotAssociatedError, ParameterNotAssociatedError


class Spectrum(BaseProduct):
    def __init__(self, path: str, rmf_path: str, arf_path: str, b_path: str, b_rmf_path: str, b_arf_path: str,
                 reg_type: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str, gen_cmd: str,
                 raise_properly: bool = True):

        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, raise_properly)
        self._prod_type = "spectrum"

        if os.path.exists(rmf_path):
            self._rmf = rmf_path
        else:
            self._rmf = None
            self._usable = False
            self._why_unusable.append("RMFPathDoesNotExist")

        if os.path.exists(arf_path):
            self._arf = arf_path
        else:
            self._arf = None
            self._usable = False
            self._why_unusable.append("ARFPathDoesNotExist")

        if os.path.exists(b_path):
            self._back_spec = b_path
        else:
            self._back_spec = None
            self._usable = False
            self._why_unusable.append("BackSpecPathDoesNotExist")

        if os.path.exists(b_rmf_path):
            self._back_rmf = b_rmf_path
        else:
            self._back_rmf = None
            self._usable = False
            self._why_unusable.append("BackRMFPathDoesNotExist")

        if os.path.exists(b_arf_path):
            self._back_arf = b_arf_path
        else:
            self._back_arf = None
            self._usable = False
            self._why_unusable.append("BackARFPathDoesNotExist")

        allowed_regs = ["region", "r2500", "r500", "r200", "custom"]
        if reg_type in allowed_regs:
            self._reg_type = reg_type
        else:
            self._usable = False
            self._why_unusable.append("InvalidRegionType")
            self._reg_type = None
            raise ValueError("{0} is not a supported region type, please use one of these; "
                             "{1}".format(reg_type, ", ".join(allowed_regs)))

        self._update_spec_headers("main")
        self._update_spec_headers("back")

        self._exp = None
        self._plot_data = {}
        self._luminosities = {}
        self._count_rate = {}

        # This is specifically for fakeit runs (for cntrate - lum conversions) on the ARF/RMF
        #  associated with this Spectrum
        self._conv_factors = {}

    def _update_spec_headers(self, which_spec: str):
        """
        An internal method that will 'push' the current class attributes that hold the paths to data products
        (like ARF and RMF) to the relevant spectrum file.
        :param str which_spec: A flag that tells the method whether to update the header of
         the main or background spectrum.
        """
        # This function is meant for internal use only, so I won't check that the passed-in file paths
        #  actually exist. This will have been checked already
        if which_spec == "main":
            with FITS(self._path, 'rw') as spec_fits:
                spec_fits[1].write_key("RESPFILE", self._rmf)
                spec_fits[1].write_key("ANCRFILE", self._arf)
                spec_fits[1].write_key("BACKFILE", self._back_spec)
                spec_fits[0].write_key("RESPFILE", self._rmf)
                spec_fits[0].write_key("ANCRFILE", self._arf)
                spec_fits[0].write_key("BACKFILE", self._back_spec)
        elif which_spec == "back":
            with FITS(self._back_spec, 'rw') as spec_fits:
                spec_fits[1].write_key("RESPFILE", self._back_rmf)
                spec_fits[1].write_key("ANCRFILE", self._back_arf)
                spec_fits[0].write_key("RESPFILE", self._back_rmf)
                spec_fits[0].write_key("ANCRFILE", self._back_arf)
        else:
            raise ValueError("Illegal value for which_spec, you shouldn't be using this internal function!")

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

    # This is an intrinsic property of the generated spectrum, so users will not be allowed to change this
    @property
    def reg_type(self) -> str:
        """
        Getter method for the type of region this spectrum was generated for. e.g. 'region' - which would
        mean it represents the spectrum inside a region specificied by region files, or 'r500' - which
        would mean the radius of a cluster where the mean density is 500 times critical density of the Universe.
        :return: The region type this spectrum was generated for
        :rtype: str
        """
        return self._reg_type

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
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this source")
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

    def view(self, lo_en: Quantity = Quantity(0.0, "keV"), hi_en: Quantity = Quantity(30.0, "keV")):
        """
        Very simple method to plot the data/models associated with this Spectrum object,
        between certain energy limits.
        :param Quantity lo_en: The lower energy limit from which to plot the spectrum.
        :param Quantity hi_en: The upper energy limit to plot the spectrum to.
        """
        if lo_en > hi_en:
            raise ValueError("hi_en cannot be greater than lo_en")
        else:
            lo_en = lo_en.to("keV").value
            hi_en = hi_en.to("keV").value

        if len(self._plot_data.keys()) != 0:
            # Create figure object
            plt.figure(figsize=(8, 5))

            # Set the plot up to look nice and professional.
            ax = plt.gca()
            ax.minorticks_on()
            ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

            # Set the title with all relevant information about the spectrum object in it
            plt.title("{n} - {o}{i} {r} Spectrum".format(n=self.obj_name, o=self.obs_id, i=self.instrument.upper(),
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
    def __init__(self, path: str, obs_id: str, instrument: str, stdout_str: str, stderr_str: str,
                 gen_cmd: str, raise_properly: bool = True):
        raise NotImplementedError("Annular Spectra aren't even started")
        # super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd, raise_properly)





