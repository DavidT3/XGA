#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 12/03/2025, 22:16. Copyright (c) The Contributors

from typing import Tuple, List, Union
from warnings import warn, simplefilter

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Cosmology
from astropy.units import Quantity, UnitBase, deg, UnitConversionError
from numpy import ndarray

from .base import BaseSource
from .. import DEFAULT_COSMO
from ..exceptions import NotAssociatedError, PeakConvergenceFailedError, NoRegionsError, NoValidObservationsError, \
    NoProductAvailableError
from ..products import RateMap
from ..sourcetools import rad_to_ang, ang_to_rad, nh_lookup

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
simplefilter('ignore', wcs.FITSFixedWarning)


class ExtendedSource(BaseSource):
    """
    The general extended source XGA class, for extended X-ray sources that you do not yet want to analyse as
    a specific type of astrophysical object. This class is subclassed by GalaxyCluster, which then adds more
    specific analyses, for instance. This general class is useful for when you're doing exploratory analyses.

    :param float ra: The right-ascension of the source, in degrees.
    :param float dec: The declination of the source, in degrees.
    :param float redshift: The redshift of the source, optional. Default is None.
    :param str name: The name of the source, optional. Name will be constructed from position if None.
    :param Quantity custom_region_radius: A custom analysis region radius for this source, optional.
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
    :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
        to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
        is False. User should only set to True to remove warnings.
    """
    def __init__(self, ra: float, dec: float, redshift: float = None, name: str = None,
                 custom_region_radius: Quantity = None, use_peak: bool = True,
                 peak_lo_en: Quantity = Quantity(0.5, "keV"), peak_hi_en: Quantity = Quantity(2.0, "keV"),
                 back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5,
                 cosmology: Cosmology = DEFAULT_COSMO,  load_products: bool = True, load_fits: bool = False,
                 peak_find_method: str = "hierarchical", in_sample: bool = False):
        """
        The init for the general extended source XGA class, takes information on the position (and optionally
        redshift) of source of interest, matches to extended regions, and optionally performs peak finding.

        :param float ra: The right-ascension of the source, in degrees.
        :param float dec: The declination of the source, in degrees.
        :param float redshift: The redshift of the source, optional. Default is None.
        :param str name: The name of the source, optional. Name will be constructed from position if None.
        :param Quantity custom_region_radius: A custom analysis region radius for this source, optional.
        :param bool use_peak: Whether peak position should be found and used.
        :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak position
            from. Default is 0.5keV
        :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak position
            from. Default is 2.0keV.
        :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
            radius for the background region. Default is 1.05.
        :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
            radius for the background region. Default is 1.5.
        :param Cosmology cosmology: An astropy cosmology object for use throughout analysis of the source.
        :param bool load_products: Whether existing products should be loaded from disk.
        :param bool load_fits: Whether existing fits should be loaded from disk.
        :param str peak_find_method: Which peak finding method should be used (if use_peak is True). Default
            is hierarchical, simple may also be passed.
        :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
            to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
            is False. User should only set to True to remove warnings.
        """
        # Calling the BaseSource init method
        super().__init__(ra, dec, redshift, name, cosmology, load_products, load_fits, in_sample)

        self._custom_region_radius = None
        # Setting up the custom region radius attributes
        if custom_region_radius is not None and custom_region_radius.unit.is_equivalent("kpc"):
            rad = rad_to_ang(custom_region_radius, self._redshift, self._cosmo).to("deg")
            self._custom_region_radius = rad
            self._radii["custom"] = self._custom_region_radius
            self._rad_info = True
        elif custom_region_radius is not None and not custom_region_radius.unit.is_equivalent("kpc"):
            self._custom_region_radius = custom_region_radius.to("deg")
            self._radii["custom"] = self._custom_region_radius
            self._rad_info = True

        # Adding a custom radius to act as a search aperture for peak finding
        # 500kpc in degrees, for the current redshift and cosmology
        #  Or 5 arcminutes if no redshift information is present (that is allowed for the ExtendedSource class)
        if self._redshift is not None:
            search_aperture = rad_to_ang(Quantity(500, "kpc"), self._redshift, cosmo=self._cosmo)
        else:
            search_aperture = Quantity(5, 'arcmin').to('deg')
        self._radii["search"] = search_aperture

        self._use_peak = use_peak
        self._back_inn_factor = back_inn_rad_factor
        self._back_out_factor = back_out_rad_factor
        # Make sure the peak energy boundaries are in keV
        self._peak_lo_en = peak_lo_en.to('keV')
        self._peak_hi_en = peak_hi_en.to('keV')
        self._peaks = {o: {} for o in self.obs_ids}
        self._peaks.update({"combined": None})
        self._peaks_near_edge = {o: {} for o in self.obs_ids}
        self._peaks_near_edge.update({"combined": None})
        self._chosen_peak_cluster = None
        self._other_peak_clusters = None
        self._snr = {}

        # This uses the added context of the type of source to find (or not find) matches in region files
        self._regions, self._alt_match_regions, self._other_regions = self._source_type_match("ext")

        # Making a combined list of interloper regions
        self._interloper_regions = []
        for o in self._other_regions:
            self._interloper_regions += self._other_regions[o]

        # Run through any alternative matches and raise warnings
        for o in self._alt_match_regions:
            if len(self._alt_match_regions[o]) > 0:
                warn_text = "There are {0} alternative matches for observation {1}, associated with " \
                            "source {2}".format(len(self._alt_match_regions[o]), o, self.name)
                if not self._samp_member:
                    warn(warn_text, stacklevel=2)
                else:
                    self._supp_warn.append(warn_text)

        if not in_sample:
            # We make sure there are some images in existence, using the peak finding energy bounds
            from xga.sas import evselect_image
            evselect_image(self, self.peak_lo_en, self.peak_hi_en)

        self._interloper_masks = {}
        for obs_id in self.obs_ids:
            # Generating and storing these because they should only
            cur_im = self.get_products("image", obs_id)[0]
            self._interloper_masks[obs_id] = self._generate_interloper_mask(cur_im)

        # Constructs the detected dictionary, detailing whether the source has been detected IN REGION FILES
        #  in each observation.
        self._detected = {o: self._regions[o] is not None for o in self._regions}

        # If in some of the observations the source has not been detected, a warning will be raised
        if True in self._detected.values() and False in self._detected.values():
            warn_text = "{n} has not been detected in all region files, so generating and fitting products" \
                        " with the 'region' reg_type will not use all available data".format(n=self.name)
            if not self._samp_member:
                warn(warn_text, stacklevel=2)
            else:
                self._supp_warn.append(warn_text)

        # If the source wasn't detected in ALL of the observations, then we have to rely on a custom region,
        #  and if no custom region options are passed by the user then an error is raised.
        elif all([det is False for det in self._detected.values()]) and self._custom_region_radius is not None:
            warn_text = "{n} has not been detected in ANY region files, so generating and fitting products" \
                        " with the 'region' reg_type will not work".format(n=self.name)
            if not self._samp_member:
                warn(warn_text, stacklevel=2)
            else:
                self._supp_warn.append(warn_text)
        elif all([det is False for det in self._detected.values()]) and self._custom_region_radius is None \
                and "GalaxyCluster" not in repr(self):
            raise NoRegionsError("{n} has not been detected in ANY region files, and no custom region or "
                                 " overdensity radius has been passed. No analysis is possible.".format(n=self.name))

        # Call to a method that goes through all the observations and finds the X-ray peak. Also at the same
        #  time finds the X-ray peak of the combined ratemap (an essential piece of information).
        self._all_peaks(peak_find_method)
        if self._use_peak:
            self._default_coord = self.peak

    def find_peak(self, rt: RateMap, method: str = "hierarchical", num_iter: int = 20, peak_unit: UnitBase = deg) \
            -> Tuple[Quantity, bool, bool, ndarray, List]:
        """
        A method that will find the X-ray peak for the RateMap that has been passed in. It takes
        the user supplied coordinates from source initialisation as a starting point, finds the peak within a 500kpc
        radius, re-centres the region, and iterates until the peak converges to within 15kpc, or until 20
        20 iterations has been reached.

        :param RateMap rt: The ratemap which we want to find the peak (local to our user supplied coordinates) of.
        :param str method: Which peak finding method to use. Currently either hierarchical or simple can be chosen.
        :param int num_iter: How many iterations should be allowed before the peak is declared as not converged.
        :param UnitBase peak_unit: The unit the peak coordinate is returned in.
        :return: The peak coordinate, a boolean flag as to whether the returned coordinates are near
            a chip gap/edge, and a boolean flag as to whether the peak converged. It also returns the coordinates
            of the points within the chosen point cluster, and a list of all point clusters that were not chosen.
        :rtype: Tuple[Quantity, bool, bool, ndarray, List]
        """
        all_meth = ["hierarchical", "simple"]
        if method not in all_meth:
            raise ValueError("{0} is not a recognised, use one of the following methods: "
                             "{1}".format(method, ", ".join(all_meth)))
        central_coords = SkyCoord(*self.ra_dec.to("deg"))

        # Iteration counter just to kill it if it doesn't converge
        count = 0
        # Allow 20 iterations by default before we kill this - alternatively loop will exit when centre converges
        #  to within 15kpc (or 0.15arcmin).
        while count < num_iter:
            aperture_mask = self.get_mask("search", rt.obs_id, central_coords)[0]

            # Find the peak using the experimental clustering_peak method
            if method == "hierarchical":
                try:
                    peak, near_edge, chosen_coords, other_coords = rt.clustering_peak(aperture_mask, peak_unit)
                except ValueError:
                    raise PeakConvergenceFailedError("The hierarchical clustering peak finder does not "
                                                     "have enough points to work with.")
            elif method == "simple":
                peak, near_edge = rt.simple_peak(aperture_mask, peak_unit)
                chosen_coords = []
                other_coords = []

            peak_deg = rt.coord_conv(peak, deg)
            # Calculate the distance between new peak and old central coordinates
            separation = Quantity(np.sqrt(abs(peak_deg[0].value - central_coords.ra.value) ** 2 +
                                          abs(peak_deg[1].value - central_coords.dec.value) ** 2), deg)

            central_coords = SkyCoord(*peak_deg.copy())
            if self._redshift is not None:
                separation = ang_to_rad(separation, self._redshift, self._cosmo)

            if count != 0 and self._redshift is not None and separation <= Quantity(15, "kpc"):
                break
            elif count != 0 and self._redshift is None and separation <= Quantity(0.15, 'arcmin'):
                break

            count += 1

        if count == num_iter:
            converged = False
            # To do the least amount of damage, if the peak doesn't converge then we just return the
            #  user supplied coordinates
            peak = self.ra_dec
            near_edge = rt.near_edge(peak)
        else:
            converged = True

        return peak, near_edge, converged, chosen_coords, other_coords

    def _all_peaks(self, method: str):
        """
        An internal method that finds the X-ray peaks for all of the available observations and instruments,
        as well as the combined ratemap. Peak positions for individual ratemap products are allowed to not
        converge, and will just write None to the peak dictionary, but if the peak of the combined ratemap fails
        to converge an error will be thrown. The combined ratemap peak will also be stored by itself in an
        attribute, to allow a property getter easy access.

        :param str method: The method to be used for peak finding.
        """
        en_key = "bound_{l}-{u}".format(l=self._peak_lo_en.value, u=self._peak_hi_en.value)
        comb_rt = [rt[-1] for rt in self.get_products("combined_ratemap", just_obj=False) if en_key in rt]

        if len(comb_rt) != 0:
            comb_rt = comb_rt[0]
        else:
            # I didn't want to import this here, but otherwise circular imports become a problem
            from xga.sas import emosaic
            emosaic(self, "image", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
            emosaic(self, "expmap", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
            comb_rt = [rt[-1] for rt in self.get_products("combined_ratemap", just_obj=False) if en_key in rt][0]

        if self._use_peak:
            coord, near_edge, converged, cluster_coords, other_coords = self.find_peak(comb_rt, method=method)
            # Updating nH for new coord, probably won't make a difference most of the time
            self._nH = nh_lookup(coord)[0]
        else:
            # If we don't care about peak finding then this is the one to go for
            coord = self.ra_dec
            near_edge = comb_rt.near_edge(coord)
            converged = True
            cluster_coords = np.ndarray([])
            other_coords = []

        # Unfortunately if the peak convergence fails for the combined ratemap I have to raise an error
        if converged:
            self._peaks["combined"] = coord
            self._peaks_near_edge["combined"] = near_edge
            # I'm only going to save the point cluster positions for the combined ratemap
            self._chosen_peak_cluster = cluster_coords
            self._other_peak_clusters = other_coords
        else:
            raise PeakConvergenceFailedError("Peak finding on the combined ratemap failed to converge within "
                                             "15kpc for {n} in the {l}-{u} energy "
                                             "band.".format(n=self.name, l=self._peak_lo_en, u=self._peak_hi_en))

        # for obs in self.obs_ids:
        #     for rt in self.get_products("ratemap", obs_id=obs, extra_key=en_key, just_obj=True):
        #         if self._use_peak:
        #             coord, near_edge, converged, cluster_coords, other_coords = self.find_peak(rt)
        #             if converged:
        #                 self._peaks[obs][rt.instrument] = coord
        #                 self._peaks_near_edge[obs][rt.instrument] = near_edge
        #             else:
        #                 self._peaks[obs][rt.instrument] = None
        #                 self._peaks_near_edge[obs][rt.instrument] = None
        #         else:
        #             self._peaks[obs][rt.instrument] = self.ra_dec
        #             self._peaks_near_edge[obs][rt.instrument] = rt.near_edge(self.ra_dec)

    def get_peaks(self, obs_id: str = None, inst: str = None) -> Quantity:
        """
        A get method to return the peak of the X-ray emission of this GalaxyCluster.

        :param str obs_id: The ObsID to return the X-ray peak coordinates for.
        :param str inst: The instrument to return the X-ray peak coordinates for.
        :return: The X-ray peak coordinates for the input parameters.
        :rtype: Quantity
        """
        # Common sense checks, are the obsids/instruments associated with this source etc.
        if obs_id is not None and obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {o} is not associated with {s}.".format(o=obs_id, s=self.name))
        elif obs_id is None and inst is not None:
            raise ValueError("If obs_id is None, inst cannot be None as well.")
        elif obs_id is not None and inst is not None and inst not in self._peaks[obs_id]:
            raise NotAssociatedError("The instrument {i} is not associated with observation {o} of this "
                                     "source.".format(i=inst, o=obs_id))
        elif obs_id is None and inst is None:
            chosen = self._peaks
        elif obs_id is not None and inst is None:
            chosen = self._peaks[obs_id]
        else:
            chosen = self._peaks[obs_id][inst]

        return chosen

    def get_1d_brightness_profile(self, outer_rad: Union[Quantity, str], obs_id: str = 'combined',
                                  inst: str = 'combined', central_coord: Quantity = None, radii: Quantity = None,
                                  lo_en: Quantity = None, hi_en: Quantity = None, pix_step: int = 1,
                                  min_snr: Union[float, int] = 0.0, psf_corr: bool = False, psf_model: str = "ELLBETA",
                                  psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15):
        """
        A specific get method for 1D brightness profiles. Should provide a relatively simple way of retrieving
        specific brightness profiles from XGA's storage system. Please note that there is not a separate get method
        for brightness profiles made from combined data, instead this method will search for combined profiles if
        either obs_id or inst is set to 'combined'. - Retrieving combined profiles is the default behaviour.

        :param Quantity/str outer_rad: The outermost radius of the profile, either as a Quantity or a name (e.g. r500).
        :param str obs_id: The ObsID used to generate the profile in question, default is None. If this is set to
            combined then this method will search for profiles based on combined data.
        :param str inst: The instrument used to generate the profile in question, default is None. If this is set to
            combined then this method will search for profiles based on combined data.
        :param Quantity central_coord: The central coordinate from which the profile was generated. Default is
            None, which means we shall use the default coordinate of this source.
        :param Quantity radii: Specific radii to check for in the profiles.
        :param Quantity lo_en: The lower energy bound of the RateMap used to generate the profile.
        :param Quantity hi_en: The upper energy bound of the RateMap used to generate the profile.
        :param int pix_step: The width of each annulus in pixels used to generate the profile.
        :param float min_snr: The minimum signal to noise imposed upon the profile.
        :param bool psf_corr: Is the brightness profile corrected for PSF effects?
        :param str psf_model: If PSF corrected, the PSF model used.
        :param int psf_bins: If PSF corrected, the number of bins per side.
        :param str psf_algo: If PSF corrected, the algorithm used.
        :param int psf_iter: If PSF corrected, the number of algorithm iterations.
        :return:
        """
        # Makes sure its in our standard unit
        if isinstance(outer_rad, str):
            outer_rad = self.get_radius(outer_rad, 'deg')
        elif isinstance(outer_rad, Quantity):
            outer_rad = self.convert_radius(outer_rad, 'deg')
        else:
            raise ValueError("Outer radius may only be a string or an astropy quantity")

        if obs_id == "combined" or inst == "combined":
            interim_prods = self.get_combined_profiles("brightness", central_coord, radii, lo_en=lo_en, hi_en=hi_en)
        else:
            interim_prods = self.get_profiles("brightness", obs_id, inst, central_coord, radii, lo_en=lo_en,
                                              hi_en=hi_en)

        # The methods I used to get this far will already have gotten upset if there are no matches, so I don't need
        #  to check they exist, but I do need to check if I have a list or a single object
        if not isinstance(interim_prods, list):
            interim_prods = [interim_prods]

        matched_prods = []
        for p in interim_prods:
            if not psf_corr and p.outer_radius == outer_rad and p.pix_step == pix_step and p.min_snr == min_snr \
                    and p.psf_corrected == psf_corr:
                matched_prods.append(p)
            elif psf_corr and p.outer_radius == outer_rad and p.pix_step == pix_step and p.min_snr == min_snr \
                    and p.psf_corrected == psf_corr and p.psf_model == psf_model and p.psf_bins == psf_bins \
                    and p.psf_algorithm == psf_algo and p.psf_iterations == psf_iter:
                matched_prods.append(p)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any brightness profiles matching your input.")

        return matched_prods

    # Property SPECIFICALLY FOR THE COMBINED PEAK - as this is the peak we should be using mostly.
    @property
    def peak(self) -> Quantity:
        """
        A property getter for the combined X-ray peak coordinates. Most analysis will be centered
        on these coordinates.

        :return: The X-ray peak coordinates for the combined ratemap.
        :rtype: Quantity
        """
        return self._peaks["combined"]

    # I'm allowing this a setter, as some users may want to update the peak from outside (as is
    #  done in the ClusterSample init).
    @peak.setter
    def peak(self, new_peak: Quantity):
        """
        Allows the user to update the peak value used during analyses manually.

        :param Quantity new_peak: A new RA-DEC peak coordinate, in degrees.
        """
        if not new_peak.unit.is_equivalent("deg"):
            raise UnitConversionError("The new peak value must be in RA and DEC coordinates")
        elif len(new_peak) != 2:
            raise ValueError("Please pass an astropy Quantity, in units of degrees, with two entries - "
                             "one for RA and one for DEC.")
        self._peaks["combined"] = new_peak.to("deg")

    @property
    def custom_radius(self) -> Quantity:
        """
        A getter for the custom region that can be defined on initialisation.

        :return: The radius (in kpc) of the user defined custom region.
        :rtype: Quantity
        """
        return self._custom_region_radius

    @property
    def point_clusters(self) -> Tuple[ndarray, List[ndarray]]:
        """
        This allows you to retrieve the point cluster positions from the hierarchical clustering
        peak finding method run on the combined ratemap. This includes both the chosen cluster and
        all others that were found.

        :return: A numpy array of the positions of points of the chosen cluster (not galaxy cluster,
            a cluster of points). A list of numpy arrays with the same information for all the other clusters
            that were found
        :rtype: Tuple[ndarray, List[ndarray]]
        """
        return self._chosen_peak_cluster, self._other_peak_clusters


class PointSource(BaseSource):
    """
    The general point source XGA class, for point X-ray sources that you do not yet want to analyse as
    a specific type of astrophysical object. This general class is useful for when you're doing exploratory analyses.

    :param float ra: The right-ascension of the point source, in degrees.
    :param float dec: The declination of the point source, in degrees.
    :param float redshift: The redshift of the point source, optional. Default is None.
    :param str name: The name of the point source, optional. If no names are supplied
        then they will be constructed from the supplied coordinates.
    :param Quantity point_radius: The point source analysis region radius for this sample. An astropy quantity
        containing the radius should be passed; remember that units like kpc will also need redshift
        information. Default is 30 arcsecond radius.
    :param bool use_peak: Whether peak position should be found and used. For PointSource the 'simple' peak
        finding method is the only one available, other methods are allowed for extended sources.
    :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak
        position from. Default is 0.5keV.
    :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak
        position from. Default is 2.0keV.
    :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
        radius for the background region. Default is 1.05.
    :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
        radius for the background region. Default is 1.5.
    :param Cosmology cosmology: An astropy cosmology object for use throughout analysis of the source.
    :param bool load_products: Whether existing products should be loaded from disk.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param bool regen_merged: Should merged images/exposure maps be regenerated after cleaning. Default is
        True. This option is here so that sample objects can regenerate all merged products at once, which is
        more efficient as it can exploit parallelisation more fully - user probably doesn't need to touch this.
    :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
        to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
        is False. User should only set to True to remove warnings.
    """
    def __init__(self, ra, dec, redshift=None, name=None, point_radius=Quantity(30, 'arcsec'), use_peak=False,
                 peak_lo_en=Quantity(0.5, "keV"), peak_hi_en=Quantity(2.0, "keV"), back_inn_rad_factor=1.05,
                 back_out_rad_factor=1.5, cosmology: Cosmology = DEFAULT_COSMO, load_products=True, load_fits=False,
                 regen_merged: bool = True, in_sample: bool = False):
        """
        The init of the general XGA point source class.

        :param float ra: The right-ascension of the point source, in degrees.
        :param float dec: The declination of the point source, in degrees.
        :param float redshift: The redshift of the point source, optional. Default is None.
        :param str name: The name of the point source, optional. If no names are supplied
            then they will be constructed from the supplied coordinates.
        :param Quantity point_radius: The point source analysis region radius for this sample. An astropy quantity
            containing the radius should be passed; remember that units like kpc will also need redshift
            information. Default is 30 arcsecond radius.
        :param bool use_peak: Whether peak position should be found and used. For PointSource the 'simple' peak
            finding method is the only one available, other methods are allowed for extended sources.
        :param Quantity peak_lo_en: The lower energy bound for the RateMap to calculate peak
            position from. Default is 0.5keV.
        :param Quantity peak_hi_en: The upper energy bound for the RateMap to calculate peak
            position from. Default is 2.0keV.
        :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
            radius for the background region. Default is 1.05.
        :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
            radius for the background region. Default is 1.5.
        :param cosmology: An astropy cosmology object for use throughout analysis of the source.
        :param bool load_products: Whether existing products should be loaded from disk.
        :param bool load_fits: Whether existing fits should be loaded from disk.
        :param bool regen_merged: Should merged images/exposure maps be regenerated after cleaning. Default is
            True. This option is here so that sample objects can regenerate all merged products at once, which is
            more efficient as it can exploit parallelisation more fully - user probably doesn't need to touch this.
        :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
            to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
            is False. User should only set to True to remove warnings.
        """

        super().__init__(ra, dec, redshift, name, cosmology, load_products, load_fits, in_sample)
        # This uses the added context of the type of source to find (or not find) matches in region files
        # This is the internal dictionary where all regions, defined by reg-files or by users, will be stored
        self._regions, self._alt_match_regions, self._other_regions = self._source_type_match("pnt")
        self._detected = {o: self._regions[o] is not None for o in self._regions}

        # Making a combined list of interloper regions
        self._interloper_regions = []
        for o in self._other_regions:
            self._interloper_regions += self._other_regions[o]

        if point_radius is not None and point_radius.unit.is_equivalent("kpc"):
            rad = rad_to_ang(point_radius, self._redshift, self._cosmo).to("deg")
            self._custom_region_radius = rad
            self._radii["point"] = self._custom_region_radius
            self._rad_info = True
        elif point_radius is not None and not point_radius.unit.is_equivalent("kpc"):
            self._custom_region_radius = point_radius.to("deg")
            self._radii["point"] = self._custom_region_radius
            self._rad_info = True

        if self._redshift is not None and point_radius.unit.is_equivalent("kpc"):
            search_aperture = rad_to_ang(point_radius.to("kpc"), self._redshift, cosmo=self._cosmo)
        elif point_radius.unit.is_equivalent("deg"):
            search_aperture = point_radius.to("deg")
        else:
            raise UnitConversionError("Can't convert {u} to a XGA supported length unit".format(u=point_radius.unit))
        self._radii["search"] = search_aperture

        if not in_sample:
            # We make sure there are some images in existence, using the peak finding energy bounds
            from xga.sas import evselect_image
            evselect_image(self, self.peak_lo_en, self.peak_hi_en)

        # This generates masks to remove interloper regions
        self._interloper_masks = {}
        for obs_id in self.obs_ids:
            # Generating and storing these because they should only
            cur_im = self.get_products("image", obs_id)[0]
            self._interloper_masks[obs_id] = self._generate_interloper_mask(cur_im)

        # Here we automatically clean the observations, to make sure the point source does actually lie
        #  on the detector and not just near it
        # Use a pretty harsh acceptance fraction
        reject_dict = self.obs_check("point", 0.9)
        if len(reject_dict) != 0:
            # Use the source method to remove data we've decided isn't worth keeping
            self.disassociate_obs(reject_dict)
            if len(self._obs) == 0:
                raise NoValidObservationsError("Observation cleaning has been run and there are no remaining"
                                               " observations. ")

            if regen_merged:
                from ..sas import emosaic
                emosaic(self, "image", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
                emosaic(self, "expmap", self._peak_lo_en, self._peak_hi_en, disable_progress=True)

        self._use_peak = use_peak
        self._back_inn_factor = back_inn_rad_factor
        self._back_out_factor = back_out_rad_factor
        # Make sure the peak energy boundaries are in keV
        self._peak_lo_en = peak_lo_en.to('keV')
        self._peak_hi_en = peak_hi_en.to('keV')
        self._peaks = {o: {} for o in self.obs_ids}
        self._peaks.update({"combined": None})
        self._peaks_near_edge = {o: {} for o in self.obs_ids}

        self._all_peaks()

        if self._use_peak:
            self._default_coord = self.peak

    @property
    def point_radius(self) -> Quantity:
        """
        Property getter to access the point_radius declared on initialisation of the source, the radius
        of the region that is used for point source analysis.

        :return: The radius of the point source analysis region.
        :rtype: Quantity
        """
        return self._custom_region_radius

    def _all_peaks(self):
        en_key = "bound_{l}-{u}".format(l=self._peak_lo_en.value, u=self._peak_hi_en.value)
        comb_rt = self.get_products("combined_ratemap", extra_key=en_key)

        if len(comb_rt) != 0:
            comb_rt = comb_rt[0]
        else:
            from xga.sas import emosaic
            emosaic(self, "image", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
            emosaic(self, "expmap", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
            comb_rt = self.get_products("combined_ratemap", extra_key=en_key)[0]

        if self._use_peak:
            coord, near_edge = self.find_peak(comb_rt)
            # Updating nH for new coord, probably won't make a difference most of the time
            self._nH = nh_lookup(coord)[0]
        else:
            # If we don't care about peak finding then this is the boi to go for
            coord = self.ra_dec
            near_edge = comb_rt.near_edge(coord)

        self._peaks["combined"] = coord
        self._peaks_near_edge["combined"] = near_edge

    def find_peak(self, rt: RateMap, peak_unit: UnitBase = deg) -> Tuple[Quantity, bool]:
        """
        Uses a simple 'brightest pixel' method to measure a peak coordinate for the point source.

        :param RateMap rt: The RateMap to measure the peak from.
        :param UnitBase peak_unit: The desired output unit of the peak.
        :return:  The peak, and a boolean flag as to whether the peak is near an edge.
        :rtype: Tuple[Quantity, bool]
        """
        central_coords = SkyCoord(*self.ra_dec.to("deg"))
        aperture_mask = self.get_mask("search", rt.obs_id, central_coords)[0]
        peak, near_edge = rt.simple_peak(aperture_mask, peak_unit)

        return peak, near_edge

    @property
    def peak(self) -> Quantity:
        """
        A property getter for the combined X-ray peak coordinates.

        :return: The X-ray peak coordinates for the combined ratemap.
        :rtype: Quantity
        """
        return self._peaks["combined"]







