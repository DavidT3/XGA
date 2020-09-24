#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 24/09/2020, 10:33. Copyright (c) David J Turner

import warnings
from typing import Tuple, Union

from astropy import wcs
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitConversionError, pix, kpc
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter

from .general import ExtendedSource
from ..exceptions import ModelNotAssociatedError, ParameterNotAssociatedError, NoRegionsError
from ..imagetools import radial_brightness, pizza_brightness
from ..sourcetools import ang_to_rad, rad_to_ang

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class GalaxyCluster(ExtendedSource):
    def __init__(self, ra, dec, redshift, name=None, r200: Quantity = None, r500: Quantity = None,
                 r2500: Quantity = None, richness: float = None, richness_err: float = None,
                 wl_mass: Quantity = None, wl_mass_err: Quantity = None, custom_region_radius=None, use_peak=True,
                 peak_lo_en=Quantity(0.5, "keV"), peak_hi_en=Quantity(2.0, "keV"), back_inn_rad_factor=1.05,
                 back_out_rad_factor=1.5, cosmology=Planck15, load_products=True, load_fits=False,
                 clean_obs=True, clean_obs_reg="r200", clean_obs_threshold=0.3, regen_merged: bool = True):
        super().__init__(ra, dec, redshift, name, custom_region_radius, use_peak, peak_lo_en, peak_hi_en,
                         back_inn_rad_factor, back_out_rad_factor, cosmology, load_products, load_fits)

        if r200 is None and r500 is None and r2500 is None:
            raise ValueError("You must set at least one overdensity radius")

        # Here we don't need to check if a non-null redshift was supplied, a redshift is required for
        #  initialising a GalaxyCluster object. These chunks just convert the radii to kpc.
        # I know its ugly to have the same code three times, but I want these to be in attributes.
        if r200 is not None and r200.unit.is_equivalent("deg"):
            self._r200 = ang_to_rad(r200, self._redshift, self._cosmo).to("kpc")
            # Radii must be stored in degrees in the internal radii dictionary
            self._radii["r200"] = r200.to("deg")
        elif r200 is not None and r200.unit.is_equivalent("kpc"):
            self._r200 = r200.to("kpc")
            self._radii["r200"] = rad_to_ang(r200, self.redshift, self.cosmo)
        elif r200 is not None and not r200.unit.is_equivalent("kpc") and not r200.unit.is_equivalent("deg"):
            raise UnitConversionError("R200 radius must be in either angular or distance units.")

        if r500 is not None and r500.unit.is_equivalent("deg"):
            self._r500 = ang_to_rad(r500, self._redshift, self._cosmo).to("kpc")
            self._radii["r500"] = r500.to("deg")
        elif r500 is not None and r500.unit.is_equivalent("kpc"):
            self._r500 = r500.to("kpc")
            self._radii["r500"] = rad_to_ang(r500, self.redshift, self.cosmo)
        elif r500 is not None and not r500.unit.is_equivalent("kpc") and not r500.unit.is_equivalent("deg"):
            raise UnitConversionError("R500 radius must be in either angular or distance units.")

        if r2500 is not None and r2500.unit.is_equivalent("deg"):
            self._r2500 = ang_to_rad(r2500, self._redshift, self._cosmo).to("kpc")
            self._radii["r2500"] = r2500.to("deg")
        elif r2500 is not None and r2500.unit.is_equivalent("kpc"):
            self._r2500 = r2500.to("kpc")
            self._radii["r2500"] = rad_to_ang(r2500, self.redshift, self.cosmo)
        elif r2500 is not None and not r2500.unit.is_equivalent("kpc") and not r2500.unit.is_equivalent("deg"):
            raise UnitConversionError("R2500 radius must be in either angular or distance units.")

        # Reading observables into their attributes, if the user doesn't pass a value for a particular observable
        #  it will be None.
        self._richness = richness
        self._richness_err = richness_err

        # Mass has a unit, unlike richness, so need to check that as we're reading it in
        if wl_mass is not None and wl_mass.unit.is_equivalent("Msun"):
            self._wl_mass = wl_mass.to("Msun")
        elif wl_mass is not None and not wl_mass.unit.is_equivalent("Msun"):
            raise UnitConversionError("The weak lensing mass value cannot be converted to MSun.")

        if wl_mass_err is not None and wl_mass_err.unit.is_equivalent("Msun"):
            self._wl_mass_err = wl_mass_err.to("Msun")
        elif wl_mass_err is not None and not wl_mass_err.unit.is_equivalent("Msun"):
            raise UnitConversionError("The weak lensing mass error value cannot be converted to MSun.")

        if clean_obs and clean_obs_reg in self._radii:
            # Use this method to figure out what data to throw away
            reject_dict = self.obs_check(clean_obs_reg, clean_obs_threshold)
            if len(reject_dict) != 0:
                # Use the source method to remove data we've decided isn't worth keeping
                self.disassociate_obs(reject_dict)
                # I used to run these just so there is an up to date combined ratemap, but its quite
                #  inefficient to do it on an individual basis if dealing with a sample, so the user will have
                #  to run those commands themselves later
                # Now I will run them only if the regen_merged flag is True
                if regen_merged:
                    from ..sas import emosaic
                    emosaic(self, "image", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
                    emosaic(self, "expmap", self._peak_lo_en, self._peak_hi_en, disable_progress=True)

        # Throws an error if a poor choice of region has been made
        elif clean_obs and clean_obs_reg not in self._radii:
            raise NoRegionsError("{c} is not associated with this source".format(c=clean_obs_reg))

    # Property getters for the over density radii, they don't get setters as various things are defined on init
    #  that I don't want to call again.
    @property
    def r200(self) -> Quantity:
        """
        Getter for the radius at which the average density is 200 times the critical density.
        :return: The R200 in kpc.
        :rtype: Quantity
        """
        return self._r200

    @property
    def r500(self) -> Quantity:
        """
        Getter for the radius at which the average density is 500 times the critical density.
        :return: The R500 in kpc.
        :rtype: Quantity
        """
        return self._r500

    @property
    def r2500(self) -> Quantity:
        """
        Getter for the radius at which the average density is 2500 times the critical density.
        :return: The R2500 in kpc.
        :rtype: Quantity
        """
        return self._r2500

    # Property getters for other observables I've allowed to be passed in.
    @property
    def weak_lensing_mass(self) -> Tuple[Quantity, Quantity]:
        """
        Gets the weak lensing mass passed in at initialisation of the source.
        :return: Two quantities, the weak lensing mass, and the weak lensing mass error in Msun. If the
        values were not passed in at initialisation, the returned values will be None.
        :rtype: Tuple[Quantity, Quantity]
        """
        return self._wl_mass, self._wl_mass_err

    @property
    def richness(self) -> Tuple[Quantity, Quantity]:
        """
        Gets the richness passed in at initialisation of the source.
        :return: Two quantities, the richness, and the weak lensing mass error. If the
        values were not passed in at initialisation, the returned values will be None.
        :rtype: Tuple[Quantity, Quantity]
        """
        return self._richness, self._richness_err

    # This does duplicate some of the functionality of get_results, but in a more specific way. I think its
    #  justified considering how often the cluster temperature is used in X-ray cluster studies.
    def get_temperature(self, reg_type: str, model: str = None):
        """
        Convenience method that calls get_results to retrieve temperature measurements. All matching values
        from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+).
        :param str reg_type: The type of region that the fitted spectra were generated from.
        :param str model: The name of the fitted model that you're requesting the results from (e.g. tbabs*apec).
        :return: The temperature value, and uncertainties.
        """
        allowed_rtype = ["region", "custom", "r500", "r200", "r2500"]

        if reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))
        elif len(self._fit_results) == 0:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this source")
        elif reg_type not in self._fit_results:
            av_regs = ", ".join(self._fit_results.keys())
            raise ModelNotAssociatedError("{0} has no associated XSPEC fit to this source; available regions are "
                                          "{1}".format(reg_type, av_regs))

        # Find which available models have kT in them
        models_with_kt = [m for m, v in self._fit_results[reg_type].items() if "kT" in v]

        if model is not None and model not in self._fit_results[reg_type]:
            av_mods = ", ".join(self._fit_results[reg_type].keys())
            raise ModelNotAssociatedError("{0} has not been fitted to {1} spectra of this source; available "
                                          "models are  {2}".format(model, reg_type, av_mods))
        elif model is not None and "kT" not in self._fit_results[reg_type][model]:
            raise ParameterNotAssociatedError("kT was not a free parameter in the {} fit to this "
                                              "source.".format(model))
        elif model is not None and "kT" in self._fit_results[reg_type][model]:
            # Just going to call the get_results method with specific parameters, to get the result formatted
            #  the same way.
            return self.get_results(reg_type, model, "kT")
        elif model is None and len(models_with_kt) != 1:
            raise ValueError("The model parameter can only be None when there is only one model available"
                             " with a kT measurement.")
        # For convenience sake, if there is only one model with a kT measurement, I'll allow the model parameter
        #  to be None.
        elif model is None and len(models_with_kt) == 1:
            return self.get_results(reg_type, models_with_kt[0], "kT")

    def view_brightness_profile(self, reg_type: str, profile_type: str = "radial", num_slices: int = 4,
                                same_peak: bool = True, pix_step: int = 1, min_snr: Union[float, int] = 0.0):
        """
        A method that generates and displays brightness profiles for the current cluster. Brightness profiles
        exclude point sources and either measure the average counts per second within a circular annulus (radial),
        or an angular region of a circular annulus (pizza). All points correspond to an annulus of width 1 pixel,
        and this method does NOT do any rebinning to maximise signal to noise.
        :param str reg_type: The region in which to view the radial brightness profile.
        :param str profile_type: The type of brightness profile you wish to view, radial or pizza.
        :param int num_slices: The number of pizza slices to cut the cluster into. The size of each
        slice will be 360 / num_slices degrees.
        :param bool same_peak: If True then the radial profiles (including for PSF corrected ratemaps)
         will all be constructed centered on the peak found for the 'normal' combined ratemap. If False,
         peaks will be found for each individual combined ratemap and profiles will be constructed
         centered on them.
        :param int pix_step: The width (in pixels) of each annular bin, default is 1.
        :param Union[float, int] min_snr: The minimum signal to noise allowed for each radial bin. This is 0 by
        default, which disables any automatic rebinning.
        """
        allowed_rtype = ["custom", "r500", "r200", "r2500"]
        if reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))

        # Check that the passed profile type is valid
        allowed_ptype = ["radial", "pizza"]
        if profile_type not in allowed_ptype:
            raise ValueError("The only allowed profile types are {}".format(", ".join(allowed_ptype)))

        # Check that the valid region choice actually has an entry that is not None
        if reg_type == "custom" and self._custom_region_radius is None:
            raise NoRegionsError("No custom region has been setup for this cluster")
        elif reg_type == "r200" and self._r200 is None:
            raise NoRegionsError("No R200 region has been setup for this cluster")
        elif reg_type == "r500" and self._r500 is None:
            raise NoRegionsError("No R500 region has been setup for this cluster")
        elif reg_type == "r2500" and self._r2500 is None:
            raise NoRegionsError("No R2500 region has been setup for this cluster")

        en_key = "bound_{l}-{u}".format(l=self._peak_lo_en.value, u=self._peak_hi_en.value)
        comb_rt = [rt[-1] for rt in self.get_products("combined_ratemap", just_obj=False) if en_key in rt][0]
        # If there have been PSF deconvolutions of the above data, then we can grab them too
        psf_comb_rts = [rt for rt in self.get_products("combined_ratemap", just_obj=False)
                        if en_key + "_" in rt[-2]]

        # TODO Decide what to do here
        source_mask, background_mask = self.get_mask(reg_type)
        source_mask = self.get_interloper_mask()
        # Get combined peak - basically the only peak internal methods will use
        pix_peak = comb_rt.coord_conv(self.peak, pix)
        rad = Quantity(self.source_back_regions(reg_type)[0].to_pixel(comb_rt.radec_wcs).radius, pix)

        # Setup the figure
        plt.figure(figsize=(8, 5))
        ax = plt.gca()

        # The plotting will be slightly different based on the profile type, also have to call the methods
        #  to generate the profiles as I don't currently store the data.
        if profile_type == "radial":
            ax.set_title("{n} - {l}-{u}keV Radial Brightness Profile".format(n=self.name, l=self._peak_lo_en.value,
                                                                             u=self._peak_hi_en.value))
            brightness, radii, radii_bins, og_background, success = radial_brightness(comb_rt, source_mask,
                                                                                      background_mask, pix_peak, rad,
                                                                                      self._redshift, pix_step, kpc,
                                                                                      self.cosmo, min_snr=min_snr)
            prof = plt.errorbar(radii.value, brightness, xerr=radii_bins.value, fmt='x', capsize=2, label="Emission")
            plt.plot(radii.value, brightness, color=prof[0].get_color())

            for psf_comb_rt in psf_comb_rts:
                p_rt = psf_comb_rt[-1]
                # If the user wants to use individual peaks, we have to find them here.
                if not same_peak:
                    pix_peak = self.find_peak(p_rt)[0]
                brightness, radii, radii_bins, background, success = radial_brightness(psf_comb_rt[-1], source_mask,
                                                                                       background_mask, pix_peak, rad,
                                                                                       self._redshift, pix_step, kpc,
                                                                                       self.cosmo, min_snr=min_snr)
                prof = plt.errorbar(radii.value, brightness, xerr=radii_bins.value, capsize=2, fmt="x",
                                    label="{m} PSF Corrected".format(m=p_rt.psf_model))
                plt.plot(radii.value, brightness, color=prof[0].get_color())

                plt.axhline(background, color=prof[0].get_color(), linestyle="dashed",
                            label="{m} PSF Corrected Background".format(m=p_rt.psf_model))

        elif profile_type == "pizza":
            ax.set_title("{n} - {l}-{u}keV Pizza Brightness Profile".format(n=self.name, l=self._peak_lo_en.value,
                                                                            u=self._peak_hi_en.value))
            brightness, radii, angles, og_background, inn_rads, out_rads = pizza_brightness(comb_rt, source_mask,
                                                                                            background_mask, pix_peak,
                                                                                            rad, num_slices,
                                                                                            self._redshift, pix_step,
                                                                                            kpc, self.cosmo)
            for ang_ind in range(angles.shape[0]):
                # Setup labels with the angles covered by the profile
                lab_str = "{0}$^{{\circ}}$-{1}$^{{\circ}}$ Slice Emission".format(*angles[ang_ind, :].value)
                plt.plot(radii, brightness[:, ang_ind], label=lab_str)

        # Plot the background level
        plt.axhline(og_background, color="black", linestyle="dashed", label="Background")

        # This adds small ticks to the axis
        ax.minorticks_on()
        # Set the lower limit of the x-axis to 0
        ax.set_xlim(0,)
        # Adjusts how the ticks look
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)
        # Choose y-axis log scaling because otherwise you can't really make out the profiles very well
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlim(radii.min().value, )
        ax.xaxis.set_major_formatter(ScalarFormatter())

        # Labels and legends
        ax.set_ylabel("S$_{b}$ [count s$^{-1}$ pix$^{-2}$]")
        ax.set_xlabel("Radius [kpc]")
        plt.legend(loc="best")
        # Removes white space around the plot
        plt.tight_layout()
        # Plots the plot
        plt.show()
        plt.close('all')

    def combined_lum_conv_factor(self, reg_type: str, lo_en: Quantity, hi_en: Quantity) -> Quantity:
        """
        Combines conversion factors calculated for this source with individual instrument-observation
        spectra, into one overall conversion factor.
        :param str reg_type: The region type the conversion factor is associated with.
        :param Quantity lo_en: The lower energy limit of the conversion factors.
        :param Quantity hi_en: The upper energy limit of the conversion factors.
        :return: A combined conversion factor that can be applied to a combined ratemap to
        calculate luminosity.
        :rtype: Quantity
        """
        # Grabbing the relevant spectra
        spec = self.get_products("spectrum", extra_key=reg_type)
        # Setting up variables to be added into
        av_lum = Quantity(0, "erg/s")
        total_rate = Quantity(0, "ct/s")
        # Cycling through the relevant spectra
        for s in spec:
            # The luminosity is added to the average luminosity variable, will be divided by N
            #  spectra at the end.
            av_lum += s.get_conv_factor(lo_en, hi_en, "tbabs*apec")[1]
            # The count rate is just added into a total count rate
            total_rate += s.get_conv_factor(lo_en, hi_en, "tbabs*apec")[2]

        # Making av_lum actually an average
        av_lum /= len(spec)

        # Calculating and returning the combined factor.
        return av_lum / total_rate

    def combined_norm_conv_factor(self, reg_type: str, lo_en: Quantity, hi_en: Quantity) -> Quantity:
        """
        Combines count-rate to normalisation conversion factors associated with this source
        :param str reg_type: The region type the conversion factor is associated with.
        :param Quantity lo_en: The lower energy limit of the conversion factors.
        :param Quantity hi_en: The upper energy limit of the conversion factors.
        :return: A combined conversion factor that can be applied to a combined ratemap to
        calculate luminosity.
        :rtype: Quantity
        """
        spec = self.get_products("spectrum", extra_key=reg_type)
        total_rate = Quantity(0, "ct/s")
        for s in spec:
            total_rate += s.get_conv_factor(lo_en, hi_en, "tbabs*apec")[2]

        # Calculating and returning the combined factor.
        return 1 / total_rate



