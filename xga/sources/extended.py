#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 23/02/2021, 13:07. Copyright (c) David J Turner

import warnings
from typing import Union, List, Tuple, Dict

import numpy as np
from astropy import wcs
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitConversionError, kpc

from .general import ExtendedSource
from ..exceptions import NoRegionsError, NoProductAvailableError
from ..imagetools import radial_brightness
from ..products import Spectrum, BaseProfile1D
from ..products.profile import ProjectedGasTemperature1D, APECNormalisation1D
from ..sourcetools import ang_to_rad, rad_to_ang

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class GalaxyCluster(ExtendedSource):
    """
    This class is for the declaration and analysis of GalaxyCluster sources, and is a subclass of ExtendedSource.
    """
    def __init__(self, ra, dec, redshift, name=None, r200: Quantity = None, r500: Quantity = None,
                 r2500: Quantity = None, richness: float = None, richness_err: float = None,
                 wl_mass: Quantity = None, wl_mass_err: Quantity = None, custom_region_radius=None, use_peak=True,
                 peak_lo_en=Quantity(0.5, "keV"), peak_hi_en=Quantity(2.0, "keV"), back_inn_rad_factor=1.05,
                 back_out_rad_factor=1.5, cosmology=Planck15, load_products=True, load_fits=False,
                 clean_obs=True, clean_obs_reg="r200", clean_obs_threshold=0.3, regen_merged: bool = True):
        self._radii = {}
        if r200 is None and r500 is None and r2500 is None:
            raise ValueError("You must set at least one overdensity radius")

        # Here we don't need to check if a non-null redshift was supplied, a redshift is required for
        #  initialising a GalaxyCluster object. These chunks just convert the radii to kpc.
        # I know its ugly to have the same code three times, but I want these to be in attributes.
        if r200 is not None and r200.unit.is_equivalent("deg"):
            self._r200 = ang_to_rad(r200, redshift, cosmology).to("kpc")
            # Radii must be stored in degrees in the internal radii dictionary
            self._radii["r200"] = r200.to("deg")
        elif r200 is not None and r200.unit.is_equivalent("kpc"):
            self._r200 = r200.to("kpc")
            self._radii["r200"] = rad_to_ang(r200, redshift, cosmology)
        elif r200 is not None and not r200.unit.is_equivalent("kpc") and not r200.unit.is_equivalent("deg"):
            raise UnitConversionError("R200 radius must be in either angular or distance units.")
        elif r200 is None and clean_obs_reg == "r200":
            clean_obs_reg = "r500"

        if r500 is not None and r500.unit.is_equivalent("deg"):
            self._r500 = ang_to_rad(r500, redshift, cosmology).to("kpc")
            self._radii["r500"] = r500.to("deg")
        elif r500 is not None and r500.unit.is_equivalent("kpc"):
            self._r500 = r500.to("kpc")
            self._radii["r500"] = rad_to_ang(r500, redshift, cosmology)
        elif r500 is not None and not r500.unit.is_equivalent("kpc") and not r500.unit.is_equivalent("deg"):
            raise UnitConversionError("R500 radius must be in either angular or distance units.")

        if r2500 is not None and r2500.unit.is_equivalent("deg"):
            self._r2500 = ang_to_rad(r2500, redshift, cosmology).to("kpc")
            self._radii["r2500"] = r2500.to("deg")
        elif r2500 is not None and r2500.unit.is_equivalent("kpc"):
            self._r2500 = r2500.to("kpc")
            self._radii["r2500"] = rad_to_ang(r2500, redshift, cosmology)
        elif r2500 is not None and not r2500.unit.is_equivalent("kpc") and not r2500.unit.is_equivalent("deg"):
            raise UnitConversionError("R2500 radius must be in either angular or distance units.")

        super().__init__(ra, dec, redshift, name, custom_region_radius, use_peak, peak_lo_en, peak_hi_en,
                         back_inn_rad_factor, back_out_rad_factor, cosmology, load_products, load_fits)

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
            raise NoRegionsError("{c} is not associated with {s}".format(c=clean_obs_reg, s=self.name))

    def _source_type_match(self, source_type: str) -> Tuple[Dict, Dict, Dict]:
        """
        A function to override the _source_type_match method of the BaseSource class, containing slightly
        more complex matching criteria for galaxy clusters. Galaxy clusters having their own version of this
        method was driven by issue #407, the problems I was having with low redshift clusters particularly.

        Point sources within 0.15R500, 0.1R200, or 0.5R2500 (in order of descending priority, R200 will only be
        used if R500 isn't available etc.) will be allowed to remain in the analysis, as they may well be cool-cores.

        :param str source_type: Should either be ext or pnt, describes what type of source I
            should be looking for in the region files.
        :return: A dictionary containing the matched region for each ObsID + a combined region, another
            dictionary containing any sources that matched to the coordinates and weren't chosen,
            and a final dictionary with sources that aren't the target, or in the 2nd dictionary.
        :rtype: Tuple[Dict, Dict, Dict]
        """
        def dist_from_source(reg):
            """
            Calculates the euclidean distance between the centre of a supplied region, and the
            position of the source.

            :param reg: A region object.
            :return: Distance between region centre and source position.
            """
            ra = reg.center.ra.value
            dec = reg.center.dec.value
            return Quantity(np.sqrt(abs(ra - self._ra_dec[0]) ** 2 + abs(dec - self._ra_dec[1]) ** 2), 'deg')

        results_dict, alt_match_dict, anti_results_dict = super()._source_type_match('ext')

        # TODO ACTUALLY BASE THESE FACTORS ON REAL RADIUS - OBSERVABLE RELATIONS?
        if self._radii['r500'] is not None:
            check_rad = self.convert_radius(self._radii['r500'] * 0.15, 'deg')
        elif self._radii['r200'] is not None:
            check_rad = self.convert_radius(self._radii['r200'] * 0.1, 'deg')
        else:
            check_rad = self.convert_radius(self._radii['r2500'] * 0.5, 'deg')

        new_anti_results = {}
        for obs in self._obs:
            new_anti_results[obs] = []
            for reg_obj in anti_results_dict[obs]:
                dist = dist_from_source(reg_obj)
                if reg_obj.visual["color"] == 'red' and dist < check_rad:
                    warnings.warn("A point source has been detected very close to the user supplied coordinates of "
                                  "{} and will not be excluded from analysis due to the possibility of a "
                                  "mis-identified cool core".format(self.name))
                elif reg_obj.visual["color"] == "magenta" and dist < check_rad:
                    warnings.warn("A PSF sized extended source has been detected very close to the user supplied "
                                  "coordinates of {} and will not be excluded from analysis due to the possibility "
                                  "of a mis-identified cool core".format(self.name))
                else:
                    new_anti_results[obs].append(reg_obj)

        return results_dict, alt_match_dict, new_anti_results

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
    def weak_lensing_mass(self) -> Quantity:
        """
        Gets the weak lensing mass passed in at initialisation of the source.

        :return: Two quantities, the weak lensing mass, and the weak lensing mass error in Msun. If the
            values were not passed in at initialisation, the returned values will be None.
        :rtype: Quantity
        """
        if self._wl_mass is not None:
            wl_list = [self._wl_mass.value]
            wl_unit = self._wl_mass.unit
        else:
            wl_list = [np.NaN]
            wl_unit = ''

        if self._wl_mass_err is None:
            wl_list.append(np.NaN)
        elif isinstance(self._wl_mass_err, Quantity) and not self._wl_mass_err.isscalar:
            wl_list += list(self._wl_mass_err.value)
        elif isinstance(self._wl_mass_err, Quantity) and self._wl_mass_err.isscalar:
            wl_list.append(self._wl_mass_err.value)

        return Quantity(wl_list, wl_unit)

    @property
    def richness(self) -> Quantity:
        """
        Gets the richness passed in at initialisation of the source.

        :return: Two floats, the richness, and the richness error. If the values were not passed in at
            initialisation, the returned values will be None.
        :rtype: Quantity
        """
        if self._richness is not None:
            r_list = [self._richness]
        else:
            r_list = [np.NaN]

        if self._richness_err is None:
            r_list.append(np.NaN)
        elif isinstance(self._richness_err, (float, int)):
            r_list.append(self._richness_err)
        elif isinstance(self._richness_err, list):
            r_list += self._richness_err
        elif isinstance(self._richness_err, np.ndarray):
            r_list += list(self._richness_err)

        return Quantity(r_list)

    # This does duplicate some of the functionality of get_results, but in a more specific way. I think its
    #  justified considering how often the cluster temperature is used in X-ray cluster studies.
    def get_temperature(self, model: str, outer_radius: Union[str, Quantity],
                        inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                        min_counts: int = 5, min_sn: float = None, over_sample: float = None):
        """
        Convenience method that calls get_results to retrieve temperature measurements. All matching values
        from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+).

        :param str model: The name of the fitted model that you're requesting the results from (e.g. tbabs*apec).
        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in
            region files), then any inner radius will be ignored.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :return: The temperature value, and uncertainties.
        """
        res = self.get_results(model, outer_radius, inner_radius, "kT", group_spec, min_counts, min_sn, over_sample)

        return Quantity(res, 'keV')

    def _get_spec_based_profiles(self, search_key: str, radii: Quantity = None, group_spec: bool = True,
                                 min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                                 set_id: int = None) -> Union[BaseProfile1D, List[BaseProfile1D]]:
        """
        The generic get method for profiles which have been based on spectra, the only thing that tends to change
        about how we search for them is the specific search key. Largely copied from get_annular_spectra.

        :param str search_key: The profile search key, e.g. combined_1d_proj_temperature_profile.
        :param Quantity radii: The annulus boundary radii that were used to generate the annular spectra set
            from which the projected temperature profile was measured.
        :param bool group_spec: Was the spectrum set used to generate the profile grouped
        :param float min_counts: If the spectrum set used to generate the profile was grouped on minimum
            counts, what was the minimum number of counts?
        :param float min_sn: If the spectrum set used to generate the profile was grouped on minimum signal to
            noise, what was the minimum signal to noise.
        :param float over_sample: If the spectrum set used to generate the profile was over sampled, what was
            the level of over sampling used?
        :param int set_id: The unique identifier of the annular spectrum set used to generate the profile.
            Passing a value for this parameter will override any other information that you have given this method.
        :return: An XGA profile object if there is an exact match, a list of such objects if there are multiple matches.
        :rtype: Union[BaseProfile1D, List[BaseProfile1D]]
        """
        if group_spec and min_counts is not None:
            extra_name = "_mincnt{}".format(min_counts)
        elif group_spec and min_sn is not None:
            extra_name = "_minsn{}".format(min_sn)
        else:
            extra_name = ''

        # And if it was oversampled during generation then we need to include that as well
        if over_sample is not None:
            extra_name += "_ovsamp{ov}".format(ov=over_sample)

        # Combines the annular radii into a string, and makes sure the radii are in degrees, as radii are in
        #  degrees in the storage key
        if radii is not None:
            # We're dealing with the best case here, the user has passed radii, so we can generate an exact
            #  storage key and look for a single match
            ann_rad_str = "_".join(self.convert_radius(radii, 'deg').value.astype(str))
            spec_storage_name = "ra{ra}_dec{dec}_ar{ar}_grp{gr}"
            spec_storage_name = spec_storage_name.format(ra=self.default_coord[0].value,
                                                         dec=self.default_coord[1].value,
                                                         ar=ann_rad_str, gr=group_spec)
            spec_storage_name += extra_name
        else:
            # This is a worse case, we don't have radii, so we split the known parts of the key into a list
            #  and we'll look for partial matches
            pos_str = "ra{ra}_dec{dec}".format(ra=self.default_coord[0].value, dec=self.default_coord[1].value)
            grp_str = "grp{gr}".format(gr=group_spec) + extra_name
            spec_storage_name = [pos_str, grp_str]

        # If the user hasn't passed a set ID AND the user has passed radii then we'll go looking with out
        #  properly constructed storage key
        if set_id is None and radii is not None:
            matched_prods = self.get_products(search_key, extra_key=spec_storage_name)
        # But if the user hasn't passed an ID AND the radii are None then we look for partial matches
        elif set_id is None and radii is None:
            matched_prods = [p for p in self.get_products(search_key)
                             if spec_storage_name[0] in p.storage_key and spec_storage_name[1] in p.storage_key]
        # However if they have passed a setID then this over-rides everything else
        else:
            matched_prods = [p for p in self.get_products(search_key) if p.set_ident == set_id]

        return matched_prods

    def get_proj_temp_profiles(self, radii: Quantity = None, group_spec: bool = True, min_counts: int = 5,
                               min_sn: float = None, over_sample: float = None, set_id: int = None) \
            -> Union[ProjectedGasTemperature1D, List[ProjectedGasTemperature1D]]:
        """
        A get method for projected temperature profiles generated by XGA's XSPEC interface. This works identically
        to the get_annular_spectra method, because projected temperature profiles are generated from annular spectra,
        and as such can be described by the same parameters.

        :param Quantity radii: The annulus boundary radii that were used to generate the annular spectra set
            from which the projected temperature profile was measured.
        :param bool group_spec: Was the spectrum set used to generate the profile grouped
        :param float min_counts: If the spectrum set used to generate the profile was grouped on minimum
            counts, what was the minimum number of counts?
        :param float min_sn: If the spectrum set used to generate the profile was grouped on minimum signal to
            noise, what was the minimum signal to noise.
        :param float over_sample: If the spectrum set used to generate the profile was over sampled, what was
            the level of over sampling used?
        :param int set_id: The unique identifier of the annular spectrum set used to generate the profile.
            Passing a value for this parameter will override any other information that you have given this method.
        :return: An XGA ProjectedGasTemperature1D object if there is an exact match, a list of such objects
            if there are multiple matches.
        :rtype: Union[ProjectedGasTemperature1D, List[ProjectedGasTemperature1D]]
        """
        matched_prods = self._get_spec_based_profiles("combined_1d_proj_temperature_profile", radii, group_spec,
                                                      min_counts, min_sn, over_sample, set_id)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("No matching 1D projected temperature profiles can be found.")

        return matched_prods

    def get_apec_norm_profiles(self, radii: Quantity = None, link_norm: bool = False,
                               group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                               over_sample: float = None, set_id: int = None) \
            -> Union[APECNormalisation1D, List[APECNormalisation1D]]:
        """
        A get method for APEC normalisation profiles generated by XGA's XSPEC interface.

        :param Quantity radii: The annulus boundary radii that were used to generate the annular spectra set
            from which the normalisation profile was measured.
        :param bool link_norm: This is equivelant to the link_norm parameter in single_temp_apec_profile. If
            True then the fit was run with XSPEC normalisations linked across spectra within an annulus, if False
            then each spectrum in an annulus had an individual link_norm, and multiple profiles were generated.
        :param bool group_spec: Was the spectrum set used to generate the profile grouped
        :param float min_counts: If the spectrum set used to generate the profile was grouped on minimum
            counts, what was the minimum number of counts?
        :param float min_sn: If the spectrum set used to generate the profile was grouped on minimum signal to
            noise, what was the minimum signal to noise.
        :param float over_sample: If the spectrum set used to generate the profile was over sampled, what was
            the level of over sampling used?
        :param int set_id: The unique identifier of the annular spectrum set used to generate the profile.
            Passing a value for this parameter will override any other information that you have given this method.
        :return: An XGA APECNormalisation1D object if there is an exact match, a list of such objects
            if there are multiple matches.
        :rtype: Union[ProjectedGasTemperature1D, List[ProjectedGasTemperature1D]]
        """
        if link_norm:
            matched_prods = self._get_spec_based_profiles("combined_1d_apec_norm_profile", radii, group_spec,
                                                          min_counts, min_sn, over_sample, set_id)
        else:
            matched_prods = self._get_spec_based_profiles("1d_apec_norm_profile", radii, group_spec, min_counts,
                                                          min_sn, over_sample, set_id)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("No matching APEC normalisation profiles can be found.")

        return matched_prods

    def view_brightness_profile(self, reg_type: str, central_coord: Quantity = None, pix_step: int = 1,
                                min_snr: Union[float, int] = 0.0, figsize: tuple = (10, 7), xscale: str = 'log',
                                yscale: str = 'log', back_sub: bool = True, lo_en: Quantity = Quantity(0.5, 'keV'),
                                hi_en: Quantity = Quantity(2.0, 'keV')):
        """
        A method that generates and displays brightness profile objects for this galaxy cluster. Interloper
        sources are excluded, and any fits performed to pre-existing brightness profiles which are being
        viewed will also be displayed. The profile will be generated using a RateMap between the energy bounds
        specified by lo_en and hi_en.

        :param str reg_type: The region in which to view the radial brightness profile.
        :param Quantity central_coord: The central coordinate of the brightness profile.
        :param int pix_step: The width (in pixels) of each annular bin, default is 1.
        :param float/int min_snr: The minimum signal to noise allowed for each radial bin. This is 0 by
            default, which disables any automatic rebinning.
        :param tuple figsize: The desired size of the figure, the default is (10, 7)
        :param str xscale: The scaling to be applied to the x axis, default is log.
        :param str yscale: The scaling to be applied to the y axis, default is log.
        :param bool back_sub: Should the plotted data be background subtracted, default is True.
        :param Quantity lo_en: The lower energy bound of the RateMap to generate the profile from.
        :param Quantity hi_en: The upper energy bound of the RateMap to generate the profile from.
        """
        allowed_rtype = ["custom", "r500", "r200", "r2500"]
        if reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))

        # Check that the valid region choice actually has an entry that is not None
        if reg_type == "custom" and self._custom_region_radius is None:
            raise NoRegionsError("No custom region has been setup for this cluster")
        elif reg_type == "r200" and self._r200 is None:
            raise NoRegionsError("No R200 region has been setup for this cluster")
        elif reg_type == "r500" and self._r500 is None:
            raise NoRegionsError("No R500 region has been setup for this cluster")
        elif reg_type == "r2500" and self._r2500 is None:
            raise NoRegionsError("No R2500 region has been setup for this cluster")

        comb_rt = self.get_combined_ratemaps(lo_en, hi_en)
        # If there have been PSF deconvolutions of the above data, then we can grab them too
        # I still do it this way rather than with get_combined_ratemaps because I want ALL PSF corrected ratemaps
        en_key = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)
        psf_comb_rts = [rt for rt in self.get_products("combined_ratemap", just_obj=False)
                        if en_key + "_" in rt[-2]]

        # Fetch the mask that will remove all interloper sources from the combined ratemap
        int_mask = self.get_interloper_mask()

        if central_coord is None:
            central_coord = self.default_coord

        # Read out the radii
        rad = self.get_radius(reg_type)

        # This fetches any profiles that might have already been generated to our required specifications
        try:
            sb_profile = self.get_1d_brightness_profile(rad, combined=True, pix_step=pix_step, min_snr=min_snr,
                                                        lo_en=lo_en, hi_en=hi_en)
            if isinstance(sb_profile, list):
                raise ValueError("There are multiple matches for this brightness profile, and its the developers "
                                 "fault not yours.")
        except NoProductAvailableError:
            sb_profile, success = radial_brightness(comb_rt, central_coord, rad, self._back_inn_factor,
                                                    self._back_out_factor, int_mask, self.redshift, pix_step, kpc,
                                                    self.cosmo, min_snr)
            self.update_products(sb_profile)

        for psf_comb_rt in psf_comb_rts:
            p_rt = psf_comb_rt[-1]

            try:
                psf_sb_profile = self.get_1d_brightness_profile(rad, combined=True, pix_step=pix_step, min_snr=min_snr,
                                                                psf_corr=True, psf_model=p_rt.psf_model,
                                                                psf_bins=p_rt.psf_bins, psf_algo=p_rt.psf_algorithm,
                                                                psf_iter=p_rt.psf_iterations, lo_en=lo_en, hi_en=hi_en)
                if isinstance(psf_sb_profile, list):
                    raise ValueError("There are multiple matches for this brightness profile, and its the developers "
                                     "fault not yours.")
            except NoProductAvailableError:
                psf_sb_profile, success = radial_brightness(psf_comb_rt[-1], central_coord, rad,
                                                            self._back_inn_factor, self._back_out_factor, int_mask,
                                                            self.redshift, pix_step, kpc, self.cosmo, min_snr)
                self.update_products(psf_sb_profile)

            sb_profile += psf_sb_profile

        draw_rads = {}
        for r_name in self._radii:
            if r_name not in ['search', 'custom']:
                new_key = "R$_{" + r_name[1:] + "}$"
                draw_rads[new_key] = self.get_radius(r_name, sb_profile.radii_unit)
            elif r_name == "custom":
                draw_rads["Custom"] = self.get_radius(r_name, sb_profile.radii_unit)

        sb_profile.view(xscale=xscale, yscale=yscale, figsize=figsize, draw_rads=draw_rads, back_sub=back_sub)

    def combined_lum_conv_factor(self, outer_radius: Union[str, Quantity], lo_en: Quantity, hi_en: Quantity,
                                 inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                                 min_counts: int = 5, min_sn: float = None, over_sample: float = None) -> Quantity:
        """
        Combines conversion factors calculated for this source with individual instrument-observation
        spectra, into one overall conversion factor.

        :param str/Quantity outer_radius: The name or value of the outer radius of the spectra that should be used
            to calculate conversion factors (for instance 'r200' would be acceptable for a GalaxyCluster, or
            Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any
            inner radius will be ignored.
        :param str/Quantity inner_radius: The name or value of the inner radius of the spectra that should be used
            to calculate conversion factors (for instance 'r500' would be acceptable for a GalaxyCluster, or
            Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a circular spectrum.
        :param Quantity lo_en: The lower energy limit of the conversion factors.
        :param Quantity hi_en: The upper energy limit of the conversion factors.
        :param bool group_spec: Whether the spectra that were used for fakeit were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were used for fakeit
            were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were used for fakeit
            were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were used for fakeit.
        :return: A combined conversion factor that can be applied to a combined ratemap to calculate luminosity.
        :rtype: Quantity
        """
        # Grabbing the relevant spectra
        spec = self.get_spectra(outer_radius, inner_radius=inner_radius, group_spec=group_spec, min_counts=min_counts,
                                min_sn=min_sn, over_sample=over_sample)
        # Setting up variables to be added into
        av_lum = Quantity(0, "erg/s")
        total_phot = 0

        if isinstance(spec, Spectrum):
            spec = [spec]

        # Cycling through the relevant spectra
        for s in spec:
            # The luminosity is added to the average luminosity variable, will be divided by N
            #  spectra at the end.
            av_lum += s.get_conv_factor(lo_en, hi_en, "tbabs*apec")[1]
            # Multiplying by 1e+4 because that is the length of the simulated exposure in seconds
            total_phot += s.get_conv_factor(lo_en, hi_en, "tbabs*apec")[2] * 1e+4

        # Then the total combined rate is the total number of photons / the total summed exposure (which
        #  is just 10000 seconds multiplied by the number of spectra).
        total_rate = Quantity(total_phot / (1e+4 * len(spec)), 'ct/s')

        # Making av_lum actually an average
        av_lum /= len(spec)

        # Calculating and returning the combined factor.
        return av_lum / total_rate

    def combined_norm_conv_factor(self, outer_radius: Union[str, Quantity], lo_en: Quantity, hi_en: Quantity,
                                  inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                                  min_counts: int = 5, min_sn: float = None, over_sample: float = None) -> Quantity:
        """
        Combines count-rate to normalisation conversion factors associated with this source

        :param str/Quantity outer_radius: The name or value of the outer radius of the spectra that should be used
            to calculate conversion factors (for instance 'r200' would be acceptable for a GalaxyCluster, or
            Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in region files), then any
            inner radius will be ignored.
        :param str/Quantity inner_radius: The name or value of the inner radius of the spectra that should be used
            to calculate conversion factors (for instance 'r500' would be acceptable for a GalaxyCluster, or
            Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a circular spectrum.
        :param Quantity lo_en: The lower energy limit of the conversion factors.
        :param Quantity hi_en: The upper energy limit of the conversion factors.
        :param bool group_spec: Whether the spectra that were used for fakeit were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were used for fakeit
            were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were used for fakeit
            were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were used for fakeit.
        :return: A combined conversion factor that can be applied to a combined ratemap to
            calculate luminosity.
        :rtype: Quantity
        """
        # Grabbing the relevant spectra
        spec = self.get_spectra(outer_radius, inner_radius=inner_radius, group_spec=group_spec, min_counts=min_counts,
                                min_sn=min_sn, over_sample=over_sample)

        if isinstance(spec, Spectrum):
            spec = [spec]

        total_phot = 0
        for s in spec:
            s: Spectrum
            # Multiplying the rate by 10000 because that is the exposure of the simulated spectra
            #  Set in xspec_scripts/cr_conv_calc.xcm
            total_phot += s.get_conv_factor(lo_en, hi_en, "tbabs*apec")[2].value * 1e+4

        # Then the total combined rate is the total number of photons / the total summed exposure (which
        #  is just 10000 seconds multiplied by the number of spectra).
        total_rate = Quantity(total_phot / (1e+4 * len(spec)), 'ct/s')

        # Then we return 1/the rate because this method calculates the conversion factor from count rate to
        #  normalisation (which is always 1 for these simulated spectra).
        return 1 / total_rate



