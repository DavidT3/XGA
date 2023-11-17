#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 28/04/2023, 10:18. Copyright (c) The Contributors

from typing import Union, List, Tuple, Dict
from warnings import warn, simplefilter

import numpy as np
from astropy import wcs
from astropy.cosmology import Cosmology
from astropy.units import Quantity, UnitConversionError, kpc

from .general import ExtendedSource
from .. import DEFAULT_COSMO
from ..exceptions import NoRegionsError, NoProductAvailableError
from ..imagetools import radial_brightness
from ..products import Spectrum, BaseProfile1D
from ..products.profile import ProjectedGasTemperature1D, APECNormalisation1D, GasDensity3D, GasTemperature3D, \
    HydrostaticMass
from ..sourcetools import ang_to_rad, rad_to_ang

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
simplefilter('ignore', wcs.FITSFixedWarning)


class GalaxyCluster(ExtendedSource):
    """
    This class is for the declaration and analysis of GalaxyCluster sources, and is a subclass of ExtendedSource.
    Using this source for cluster analysis gives you access to many more analyses than an ExtendedSource, and due
    to the passing of overdensity radii it has a more physically based idea of the size of the object. There are
    also extra source matching steps to deal with the possible detection of cool cores as point sources.

    :param float ra: The right-ascension of the cluster, in degrees.
    :param float dec: The declination of the cluster, in degrees.
    :param float redshift: The redshift of the cluster, required for cluster analysis.
    :param str name: The name of the cluster, optional. Name will be constructed from position if None.
    :param Quantity r200: A value for the R200 of the source. At least one overdensity radius must be passed.
    :param Quantity r500: A value for the R500 of the source. At least one overdensity radius must be passed.
    :param Quantity r2500: A value for the R2500 of the source. At least one overdensity radius must be passed.
    :param richness: An optical richness of the cluster, optional.
    :param richness_err: An uncertainty on the optical richness of the cluster, optional.
    :param Quantity wl_mass: A weak lensing mass of the cluster, optional.
    :param Quantity wl_mass_err: An uncertainty on the weak lensing mass of the cluster, optional.
    :param Quantity custom_region_radius: A custom analysis region radius for this cluster, optional.
    :param bool use_peak: Whether peak position should be found and used.
    :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak position
        from. Default is 0.5keV
    :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak position
        from. Default is 2.0keV.
    :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
        radius for the background region. Default is 1.05.
    :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
        radius for the background region. Default is 1.5.
    :param cosmology: An astropy cosmology object for use throughout analysis of the source.
    :param bool load_products: Whether existing products should be loaded from disk.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param str peak_find_method: Which peak finding method should be used (if use_peak is True). Default
        is hierarchical, simple may also be passed.
    :param bool clean_obs: Should the observations be subjected to a minimum coverage check, i.e. whether a
        certain fraction of a certain region is covered by an ObsID. Default is True.
    :param str clean_obs_reg: The region to use for the cleaning step, default is R200.
    :param float clean_obs_threshold: The minimum coverage fraction for an observation to be kept for analysis.
    :param bool regen_merged: Should merged images/exposure maps be regenerated after cleaning. Default is True.
    :param str peak_find_method: Which peak finding method should be used (if use_peak is True). Default
        is hierarchical, simple may also be passed.
    :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
        to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
        is False. User should only set to True to remove warnings.
    """
    def __init__(self, ra, dec, redshift, name=None, r200: Quantity = None, r500: Quantity = None,
                 r2500: Quantity = None, richness: float = None, richness_err: float = None,
                 wl_mass: Quantity = None, wl_mass_err: Quantity = None, custom_region_radius=None, use_peak=True,
                 peak_lo_en=Quantity(0.5, "keV"), peak_hi_en=Quantity(2.0, "keV"), back_inn_rad_factor=1.05,
                 back_out_rad_factor=1.5, cosmology: Cosmology = DEFAULT_COSMO, load_products=True, load_fits=False,
                 clean_obs=True, clean_obs_reg="r200", clean_obs_threshold=0.3, regen_merged: bool = True,
                 peak_find_method: str = "hierarchical", in_sample: bool = False):
        """
        The init of the GalaxyCluster specific XGA class, takes information on the cluster to enable analyses.

        :param float ra: The right-ascension of the cluster, in degrees.
        :param float dec: The declination of the cluster, in degrees.
        :param float redshift: The redshift of the cluster, required for cluster analysis.
        :param str name: The name of the cluster, optional. Name will be constructed from position if None.
        :param Quantity r200: A value for the R200 of the source. At least one overdensity radius must be passed.
        :param Quantity r500: A value for the R500 of the source. At least one overdensity radius must be passed.
        :param Quantity r2500: A value for the R2500 of the source. At least one overdensity radius must be passed.
        :param richness: An optical richness of the cluster, optional.
        :param richness_err: An uncertainty on the optical richness of the cluster, optional.
        :param Quantity wl_mass: A weak lensing mass of the cluster, optional.
        :param Quantity wl_mass_err: An uncertainty on the weak lensing mass of the cluster, optional.
        :param Quantity custom_region_radius: A custom analysis region radius for this cluster, optional.
        :param bool use_peak: Whether peak position should be found and used.
        :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak position
            from. Default is 0.5keV
        :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak position
            from. Default is 2.0keV.
        :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
            radius for the background region. Default is 1.05.
        :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
            radius for the background region. Default is 1.5.
        :param cosmology: An astropy cosmology object for use throughout analysis of the source.
        :param bool load_products: Whether existing products should be loaded from disk.
        :param bool load_fits: Whether existing fits should be loaded from disk.
        :param str peak_find_method: Which peak finding method should be used (if use_peak is True). Default
            is hierarchical, simple may also be passed.
        :param bool clean_obs: Should the observations be subjected to a minimum coverage check, i.e. whether a
            certain fraction of a certain region is covered by an ObsID. Default is True.
        :param str clean_obs_reg: The region to use for the cleaning step, default is R200.
        :param float clean_obs_threshold: The minimum coverage fraction for an observation to be kept for analysis.
        :param bool regen_merged: Should merged images/exposure maps be regenerated after cleaning. Default is True.
        :param str peak_find_method: Which peak finding method should be used (if use_peak is True). Default
            is hierarchical, simple may also be passed.
        :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
            to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
            is False. User should only set to True to remove warnings.
        """
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
                         back_inn_rad_factor, back_out_rad_factor, cosmology, load_products, load_fits,
                         peak_find_method, in_sample)

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
                    self._all_peaks(peak_find_method)

                    # And finally this sets the default coordinate to the peak if use peak is True
                    if self._use_peak:
                        self._default_coord = self.peak

        # Throws an error if a poor choice of region has been made
        elif clean_obs and clean_obs_reg not in self._radii:
            raise NoRegionsError("{c} is not associated with {s}".format(c=clean_obs_reg, s=self.name))

    def _source_type_match(self, source_type: str) -> Tuple[Dict, Dict, Dict]:
        """
        A function to override the _source_type_match method of the BaseSource class, containing slightly
        more complex matching criteria for galaxy clusters. Galaxy clusters having their own version of this
        method was driven by issue #407, the problems I was having with low redshift clusters particularly.

        Point sources within 0.15R500, 0.15*0.66*R200, or 0.15*2.25*R2500 (in order of descending priority, R200
        will only be used if R500 isn't available etc. - the extra factors for R200 and R2500 are meant to convert
        the radius to ~R500, and were arrived at using the Arnaud et al. 2005 R-T scaling relations) will be allowed
        to remain in the analysis, as they may well be cool-cores.

        This method also attempts to check for fragmentation of clusters by the source finder, which can cause
        issues where low redshift clusters are split up into multiple extended sources. Any interloper sources which
        are extended and intersect with a source region that has been designated the actual source will NOT be
        removed from the analysis - in practise this may be more use in making sure regions which are not consistent
        across observations do not remove chunks of the cluster (see issue #407).

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

        # The 0.66 and 2.25 factors are intended to shift the r200 and r2500 values to approximately r500, and were
        #  decided on by dividing the Arnaud et al. 2005 R-T relations by one another and finding the mean factor
        if 'r500' in self._radii:
            check_rad = self.convert_radius(self._radii['r500'] * 0.15, 'deg')
        elif 'r200' in self._radii:
            check_rad = self.convert_radius(self._radii['r200'] * 0.66 * 0.15, 'deg')
        else:
            check_rad = self.convert_radius(self._radii['r2500'] * 2.25 * 0.15, 'deg')

        # Here we scrub the anti-results dictionary (I don't know why I called it that...) to make sure cool cores
        #  aren't accidentally removed, and that chunks of cluster emission aren't removed
        new_anti_results = {}
        for obs in self._obs:
            # This is where the cleaned interlopers will be stored
            new_anti_results[obs] = []
            # Cycling through the current interloper regions for the current ObsID
            for reg_obj in anti_results_dict[obs]:
                # Calculating the distance (in degrees) of the centre of the current interloper region from
                #  the user supplied coordinates of the cluster
                dist = dist_from_source(reg_obj)

                # If the current interloper source is a point source/a PSF sized extended source and is within the
                #  fraction of the chosen characteristic radius of the cluster then we assume it is a poorly handled
                #  cool core and allow it to stay in the analysis
                if reg_obj.visual["color"] == 'red' and dist < check_rad:
                    warn_text = "A point source has been detected in {o} and is very close to the user supplied " \
                                "coordinates of {s}. It will not be excluded from analysis due to the possibility " \
                                "of a mis-identified cool core".format(s=self.name, o=obs)
                    if not self._samp_member:
                        # We do print a warning though
                        warn(warn_text, stacklevel=2)
                    else:
                        self._supp_warn.append(warn_text)

                elif reg_obj.visual["color"] == "magenta" and dist < check_rad:
                    warn_text = "A PSF sized extended source has been detected in {o} and is very close to the " \
                                "user supplied coordinates of {s}. It will not be excluded from analysis due " \
                                "to the possibility of a mis-identified cool core".format(s=self.name, o=obs)
                    if not self._samp_member:
                        warn(warn_text, stacklevel=2)
                    else:
                        self._supp_warn.append(warn_text)
                else:
                    new_anti_results[obs].append(reg_obj)

            # Here we run through the 'chosen' region for each observation (so the region that we think is the
            #  cluster) and check if any of the current set of interloper regions intersects with it. If they do
            #  then they are allowed to stay in the analysis under assumption that they're actually part of the
            #  cluster
            for res_obs in results_dict:
                if results_dict[res_obs] is not None:
                    # Reads out the chosen region for res_obs
                    src_reg_obj = results_dict[res_obs]
                    # Stores its central coordinates in an astropy quantity
                    centre = Quantity([src_reg_obj.center.ra.value, src_reg_obj.center.dec.value], 'deg')

                    # At first I set the checking radius to the semimajor axis
                    rad = Quantity(src_reg_obj.width.to('deg').value/2, 'deg')
                    # And use my handy method to find which regions intersect with a circle with the semimajor length
                    #  as radius, centred on the centre of the current chosen region
                    within_width = self.regions_within_radii(Quantity(0, 'deg'), rad, centre, new_anti_results[obs])
                    # Make sure to only select extended (green) sources
                    within_width = [reg for reg in within_width if reg.visual['color'] == 'green']

                    # Then I repeat that process with the semiminor axis, and if a interloper intersects with both
                    #  then it would intersect with the ellipse of the current chosen region.
                    rad = Quantity(src_reg_obj.height.to('deg').value/2, 'deg')
                    within_height = self.regions_within_radii(Quantity(0, 'deg'), rad, centre, new_anti_results[obs])
                    within_height = [reg for reg in within_height if reg.visual['color'] == 'green']

                    # This finds which regions are present in both lists and makes sure if they are in both
                    #  then they are NOT removed from the analysis
                    intersect_regions = list(set(within_width) & set(within_height))
                    for inter_reg in intersect_regions:
                        inter_reg_ind = new_anti_results[obs].index(inter_reg)
                        new_anti_results[obs].pop(inter_reg_ind)

        return results_dict, alt_match_dict, new_anti_results

    def _new_rad_checks(self, new_rad: Quantity) -> Tuple[Quantity, Quantity]:
        """
        An internal method to check and convert new values of overdensity radii passed to the setters
        of those overdensity radii properties. Purely to avoid repeating the same code multiple times.

        :param Quantity new_rad: The new radius that the user has passed to the property setter.
        :return: The radius converted to kpc, and to degrees.
        :rtype: Tuple[Quantity, Quantity]
        """
        if not isinstance(new_rad, Quantity):
            raise TypeError("New overdensity radii must be an astropy Quantity")

        # This will make sure that the radius is converted into kpc, and will throw errors if the new_rad is in
        #  stupid units, so I don't need to do those checks here.
        converted_kpc_rad = self.convert_radius(new_rad, 'kpc')
        # For some reason I setup the _radii internal dictionary to have units of degrees, so I convert to that as well
        converted_deg_rad = self.convert_radius(new_rad, 'deg')

        return converted_kpc_rad, converted_deg_rad

    # Property getters for the over density radii. I've added property setters to allow overdensity radii to be
    #  set by external processes that might have measured a new value - for instance an iterative mass pipeline
    @property
    def r200(self) -> Quantity:
        """
        Getter for the radius at which the average density is 200 times the critical density.

        :return: The R200 in kpc.
        :rtype: Quantity
        """
        return self._r200

    @r200.setter
    def r200(self, new_value: Quantity):
        """
        The getter for the R200 property of the GalaxyCluster source class. This checks to make sure that the
        new value is an astropy Quantity, converts it to kpc, then updates all relevant attributes of this class.

        :param Quantity new_value:
        """
        # This checks that the input is a Quantity, then converts to kpc and to degrees
        new_value_kpc, new_value_deg = self._new_rad_checks(new_value)
        # For some reason these have to be set separately, stupid design on my part, but they are in different units
        #  so I guess I must have had some plan
        self._r200 = new_value_kpc
        self._radii['r200'] = new_value_deg

    @property
    def r500(self) -> Quantity:
        """
        Getter for the radius at which the average density is 500 times the critical density.

        :return: The R500 in kpc.
        :rtype: Quantity
        """
        return self._r500

    @r500.setter
    def r500(self, new_value: Quantity):
        """
        The getter for the R500 property of the GalaxyCluster source class. This checks to make sure that the
        new value is an astropy Quantity, converts it to kpc, then updates all relevant attributes of this class.

        :param Quantity new_value:
        """
        # This checks that the input is a Quantity, then converts to kpc and to degrees
        new_value_kpc, new_value_deg = self._new_rad_checks(new_value)
        # For some reason these have to be set separately, stupid design on my part, but they are in different units
        #  so I guess I must have had some plan
        self._r500 = new_value_kpc
        self._radii['r500'] = new_value_deg

    @property
    def r2500(self) -> Quantity:
        """
        Getter for the radius at which the average density is 2500 times the critical density.

        :return: The R2500 in kpc.
        :rtype: Quantity
        """
        return self._r2500

    @r2500.setter
    def r2500(self, new_value: Quantity):
        """
        The getter for the R2500 property of the GalaxyCluster source class. This checks to make sure that the
        new value is an astropy Quantity, converts it to kpc, then updates all relevant attributes of this class.

        :param Quantity new_value:
        """
        # This checks that the input is a Quantity, then converts to kpc and to degrees
        new_value_kpc, new_value_deg = self._new_rad_checks(new_value)
        # For some reason these have to be set separately, stupid design on my part, but they are in different units
        #  so I guess I must have had some plan
        self._r2500 = new_value_kpc
        self._radii['r2500'] = new_value_deg

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

    def get_results(self, outer_radius: Union[str, Quantity], model: str = 'constant*tbabs*apec',
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), par: str = None,
                    group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None):
        """
        Important method that will retrieve fit results from the source object. Either for a specific
        parameter of a given region-model combination, or for all of them. If a specific parameter is requested,
        all matching values from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+). If no parameter is specified, the return will be a dictionary
        of such numpy arrays, with the keys corresponding to parameter names.

        This overrides the BaseSource method, but the only difference is that this has a default model, which
        is what single_temp_apec fits (constant*tbabs*apec).

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in
            region files), then any inner radius will be ignored.
        :param str model: The name of the fitted model that you're requesting the results
            from (e.g. constant*tbabs*apec).
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum.
        :param str par: The name of the parameter you want a result for.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :return: The requested result value, and uncertainties.
        """
        return super().get_results(outer_radius, model, inner_radius, par, group_spec, min_counts, min_sn, over_sample)

    def get_luminosities(self, outer_radius: Union[str, Quantity], model: str = 'constant*tbabs*apec',
                         inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = None,
                         hi_en: Quantity = None, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                         over_sample: float = None):
        """
        Get method for luminosities calculated from model fits to spectra associated with this source.
        Either for given energy limits (that must have been specified when the fit was first performed), or
        for all luminosities associated with that model. Luminosities are returned as a 3 column numpy array;
        the 0th column is the value, the 1st column is the err-, and the 2nd is err+.

        This overrides the BaseSource method, but the only difference is that this has a default model, which
        is what single_temp_apec fits (constant*tbabs*apec).

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in
            region files), then any inner radius will be ignored.
        :param str model: The name of the fitted model that you're requesting the luminosities
            from (e.g. constant*tbabs*apec).
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :return: The requested luminosity value, and uncertainties.
        """
        return super().get_luminosities(outer_radius, model, inner_radius, lo_en, hi_en, group_spec, min_counts,
                                        min_sn, over_sample)

    # This does duplicate some of the functionality of get_results, but in a more specific way. I think its
    #  justified considering how often the cluster temperature is used in X-ray cluster studies.
    def get_temperature(self, outer_radius: Union[str, Quantity], model: str = 'constant*tbabs*apec',
                        inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                        min_counts: int = 5, min_sn: float = None, over_sample: float = None):
        """
        Convenience method that calls get_results to retrieve temperature measurements. All matching values
        from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+).

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in
            region files), then any inner radius will be ignored.
        :param str model: The name of the fitted model that you're requesting the results
            from (e.g. constant*tbabs*apec).
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
        res = self.get_results(outer_radius, model, inner_radius, "kT", group_spec, min_counts, min_sn, over_sample)

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

    def get_3d_temp_profiles(self, radii: Quantity = None, group_spec: bool = True, min_counts: int = 5,
                             min_sn: float = None, over_sample: float = None, set_id: int = None) \
            -> Union[GasTemperature3D, List[GasTemperature3D]]:
        """
        A get method for 3D temperature profiles generated by XGA's de-projection routines.

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
        matched_prods = self._get_spec_based_profiles("combined_gas_temperature_profile", radii, group_spec,
                                                      min_counts, min_sn, over_sample, set_id)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("No matching 3D temperature profiles can be found.")

        return matched_prods

    def get_apec_norm_profiles(self, radii: Quantity = None, group_spec: bool = True, min_counts: int = 5,
                               min_sn: float = None, over_sample: float = None, set_id: int = None) \
            -> Union[APECNormalisation1D, List[APECNormalisation1D]]:
        """
        A get method for APEC normalisation profiles generated by XGA's XSPEC interface.

        :param Quantity radii: The annulus boundary radii that were used to generate the annular spectra set
            from which the normalisation profile was measured.
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
        matched_prods = self._get_spec_based_profiles("combined_1d_apec_norm_profile", radii, group_spec,
                                                      min_counts, min_sn, over_sample, set_id)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("No matching APEC normalisation profiles can be found.")

        return matched_prods

    def get_density_profiles(self, outer_rad: Union[Quantity, str] = None, method: str = None, obs_id: str = None,
                             inst: str = None, central_coord: Quantity = None, radii: Quantity = None,
                             pix_step: int = 1, min_snr: Union[float, int] = 0.0, psf_corr: bool = True,
                             psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15,
                             group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                             over_sample: float = None, set_id: int = None) -> Union[GasDensity3D, List[GasDensity3D]]:
        """
        This is a get method for density profiles generated by XGA, both using surface brightness profiles and spectra.
        Having to account for two different methods is why this get method has so many arguments that can be passed. If
        multiple matches for the passed variables are found, then a list of density profiles will be returned,
        otherwise only a single profile will be returned.

        :param Quantity/str outer_rad: The outer radius of the density profile, either as a name ('r500' for instance)
            or an astropy Quantity.
        :param str method: The method used to generate the density profile. For a profile created by fitting a model
            to a surface brightness profile this should be the name of the model, for a profile from annular spectra
            this should be 'spec', and for a profile generated directly from the data of a surface brightness profile
            this should be 'onion'.
        :param str obs_id: The ObsID used to generate the profile in question, default is None (which will search for
            profiles generated from combined data).
        :param str inst: The instrument used to generate the profile in question, default is None (which will
            search for profiles generated from combined data).
        :param Quantity central_coord: The central coordinate of the density profile. Default is None, which means
            we shall use the default coordinate of this source.
        :param Quantity radii: If known, the radii that were used to measure the density profile.
        :param int pix_step: The width of each annulus in pixels used to generate the profile, for profiles based on
            surface brightness.
        :param float min_snr: The minimum signal to noise imposed upon the profile, for profiles based on
            surface brightness.
        :param bool psf_corr: Is the brightness profile corrected for PSF effects, for profiles based on
            surface brightness.
        :param str psf_model: If PSF corrected, the PSF model used, for profiles based on surface brightness.
        :param int psf_bins: If PSF corrected, the number of bins per side, for profiles based on surface brightness.
        :param str psf_algo: If PSF corrected, the algorithm used, for profiles based on surface brightness.
        :param int psf_iter: If PSF corrected, the number of algorithm iterations, for profiles based on
            surface brightness.
        :param bool group_spec: Was the spectrum set used to generate the profile grouped.
        :param float min_counts: If the spectrum set used to generate the profile was grouped on minimum
            counts, what was the minimum number of counts?
        :param float min_sn: If the spectrum set used to generate the profile was grouped on minimum signal to
            noise, what was the minimum signal to noise.
        :param float over_sample: If the spectrum set used to generate the profile was over sampled, what was
            the level of over sampling used?
        :param int set_id: The unique identifier of the annular spectrum set used to generate the profile.
            Passing a value for this parameter will override any other information that you have given this method.
        :return:
        :rtype: Union[GasDensity3D, List[GasDensity3D]]
        """
        if outer_rad is not None and isinstance(outer_rad, str):
            outer_rad = self.get_radius(outer_rad, 'deg')
        elif outer_rad is not None and isinstance(outer_rad, Quantity):
            outer_rad = self.convert_radius(outer_rad, 'deg')
        elif outer_rad is not None:
            raise ValueError("Outer radius may only be a string or an astropy quantity")

        if (obs_id == "combined" or inst == "combined" or obs_id is None or inst is None) and method != 'spec':
            interim_prods = self.get_combined_profiles("gas_density", central_coord, radii)
        elif method != 'spec':
            interim_prods = self.get_profiles("gas_density", obs_id, inst, central_coord, radii)
        elif (obs_id == "combined" or inst == "combined" or obs_id is None or inst is None) and method == 'spec':
            interim_prods = self._get_spec_based_profiles('combined_gas_density_profile', radii, group_spec, min_counts,
                                                          min_sn, over_sample, set_id)
        else:
            interim_prods = self._get_spec_based_profiles('gas_density_profile', radii, group_spec, min_counts, min_sn,
                                                          over_sample, set_id)

        # The methods I used to get this far will already have gotten upset if there are no matches, so I don't need
        #  to check they exist, but I do need to check if I have a list or a single object
        if not isinstance(interim_prods, list):
            interim_prods = [interim_prods]

        matched_prods = []
        if method != 'spec' and any([p.density_method == method for p in interim_prods]):
            interim_prods = [p for p in interim_prods if p.density_method == method]
            for dens_prof in interim_prods:
                p = dens_prof.generation_profile
                if not psf_corr and p.pix_step == pix_step and p.min_snr == min_snr and p.psf_corrected == psf_corr:
                    matched_prods.append(dens_prof)
                elif psf_corr and p.pix_step == pix_step and p.min_snr == min_snr and p.psf_corrected == psf_corr and \
                        p.psf_model == psf_model and p.psf_bins == psf_bins and p.psf_algorithm == psf_algo \
                        and p.psf_iterations == psf_iter:
                    matched_prods.append(dens_prof)

        elif method == 'spec' and any([p.density_method == 'spec' for p in interim_prods]):
            matched_prods = interim_prods
        else:
            matched_prods = interim_prods

        if outer_rad is not None:
            matched_prods = [im for im in matched_prods if self.convert_radius(im.outer_radius, 'deg') == outer_rad]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any density profiles matching your input.")

        return matched_prods

    def get_hydrostatic_mass_profiles(self, temp_prof: GasTemperature3D = None, temp_model_name: str = None,
                                      dens_prof: GasDensity3D = None, dens_model_name: str = None,
                                      radii: Quantity = None) -> Union[HydrostaticMass, List[HydrostaticMass]]:
        """
        A get method for hydrostatic mass profiles associated with this galaxy cluster. This works in a slightly
        different way to the temperature and density profile get methods, as you can pass the gas temperature and
        density profiles used to generate a hydrostatic mass profile to find it. If none of the optional
        arguments are passed then all hydrostatic mass profiles associated with this source will be returned, if
        only some are passed then mass profiles which match the limited information will be found.

        :param GasTemperature3D temp_prof: The temperature profile used to generate the required hydrostatic mass
            profile, default is None.
        :param str temp_model_name: The name of the model used to fit the temperature profile used to generate the
            required hydrostatic mass profile, default is None.
        :param GasDensity3D dens_prof: The density profile used to generate the required hydrostatic mass
            profile, default is None.
        :param str dens_model_name: The name of the model used to fit the density profile used to generate the
            required hydrostatic mass profile, default is None.
        :param Quantity radii: The radii at which the hydrostatic mass profile was measured, default is None.
        :return: Either a single hydrostatic mass profile, when there is a unique match, or a list of hydrostatic
            mass profiles if there is not.
        :rtype: Union[HydrostaticMass, List[HydrostaticMass]]
        """
        # Get all the hydrostatic mass profiles associated with this source
        matched_prods = self.get_profiles('combined_hydrostatic_mass')

        # Convert the radii to degrees for comparison with deg radii later
        if radii is not None:
            radii = self.convert_radius(radii, 'deg')

        # Checking steps, looking for matches with the information passed by the user.
        if temp_prof is not None:
            matched_prods = [p for p in matched_prods if p.temperature_profile != temp_prof]

        if dens_prof is not None:
            matched_prods = [p for p in matched_prods if p.density_profile != dens_prof]

        if temp_model_name is not None:
            matched_prods = [p for p in matched_prods if p.temperature_model.name != temp_model_name]

        if dens_model_name is not None:
            matched_prods = [p for p in matched_prods if p.density_model.name != dens_model_name]

        if radii is not None:
            matched_prods = [p for p in matched_prods if p.deg_radii != radii]

        if isinstance(matched_prods, list) and len(matched_prods) == 0:
            raise NoProductAvailableError("No matching hydrostatic mass profiles can be found.")
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
            default, which disables any automatic re-binning.
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
            sb_profile = self.get_1d_brightness_profile(rad, pix_step=pix_step, min_snr=min_snr, lo_en=lo_en,
                                                        hi_en=hi_en)
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
                psf_sb_profile = self.get_1d_brightness_profile(rad, pix_step=pix_step, min_snr=min_snr,
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

    def norm_conv_factor(self, outer_radius: Union[str, Quantity], lo_en: Quantity, hi_en: Quantity,
                         inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                         min_counts: int = 5, min_sn: float = None, over_sample: float = None, obs_id: str = None,
                         inst: str = None) -> Quantity:
        """
        Combines count-rate to normalisation conversion factors associated with this source.

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
        :param str obs_id: The ObsID to fetch a conversion factor for, default is None which means the combined
            conversion factor will be returned.
        :param str inst: The instrument to fetch a conversion factor for, default is None which means the combined
            conversion factor will be returned.
        :return: A combined conversion factor that can be applied to a combined ratemap to calculate luminosity.
        :rtype: Quantity
        """
        # Check the ObsID and instrument inputs
        if all([obs_id is None, inst is None]):
            pass
        elif all([obs_id is not None, inst is not None]):
            pass
        else:
            raise ValueError("If a value is supplied for obs_id, then a value must be supplied for inst as well, and "
                             "vice versa.")

        # Grabbing the relevant spectra
        spec = self.get_spectra(outer_radius, inner_radius=inner_radius, group_spec=group_spec, min_counts=min_counts,
                                min_sn=min_sn, over_sample=over_sample, obs_id=obs_id, inst=inst)

        # Its just easier if we know that the spectra are in a list
        if isinstance(spec, Spectrum):
            spec = [spec]

        mean_areas = []
        rates = []
        for s in spec:
            s: Spectrum

            # For the current spectrum we retrieve the ARF information so that we can use it to weight things with
            #  later
            ens = (s.eff_area_hi_en + s.eff_area_lo_en) / 2
            ars = s.eff_area

            # This finds only the areas in the current energy range we're considering
            rel_ars = ars[np.argwhere((ens <= hi_en) & (ens >= lo_en)).T[0]]
            # And finds the mean effective area in that range
            mean_areas.append(rel_ars.mean().value)

            # Then we fetch the count rate for the fakeit run of the current spectrum
            rates.append(s.get_conv_factor(lo_en, hi_en, "tbabs*apec")[2].value)

        # Just putting rates as an array for convenience
        rates = np.array(rates)
        # These normalisation factors put all the conversion factors on the same footing, by weighting by
        #  effective area of the spectra
        norm_factors = np.array(mean_areas)/mean_areas[0]
        # The total photons is the sum of the rates multiplied by the simulation exposure (always 1e+4 seconds)
        #  weighted by the effective area.
        total_phot = (rates * (1e+4/norm_factors)).sum()

        # Then the total combined rate is the total number of photons / the total summed exposure, which again is a
        #  sum of the simulation exposure weighted by the effective areas of all the spectra.
        total_rate = Quantity(total_phot / (1e+4/norm_factors).sum(), 'ct/s')

        # Then we return 1/the rate because this method calculates the conversion factor from count rate to
        #  normalisation (which is always 1 for these simulated spectra). The normalisation has units of cmm^-5,
        #  as is easily derived from the normalisation expression here
        #  (https://heasarc.gsfc.nasa.gov/xanadu/xspec/manual/XSmodelApec.html)
        return Quantity(1, 'cm^-5') / total_rate



