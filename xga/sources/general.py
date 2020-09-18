#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 18/09/2020, 16:21. Copyright (c) David J Turner

import warnings
from typing import Tuple, List, Dict

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitBase, deg
from numpy import ndarray

from xga.exceptions import NotAssociatedError, PeakConvergenceFailedError, NoRegionsError, NoMatchFoundError
from xga.products import RateMap
from xga.sourcetools import rad_to_ang, ang_to_rad, nh_lookup
from .base import BaseSource

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class ExtendedSource(BaseSource):
    # TODO Make a view method for this class that plots the measured peaks on the combined ratemap.
    def __init__(self, ra, dec, redshift=None, name=None, custom_region_radius=None, use_peak=True,
                 peak_lo_en=Quantity(0.5, "keV"), peak_hi_en=Quantity(2.0, "keV"),
                 back_inn_rad_factor=1.05, back_out_rad_factor=1.5, cosmology=Planck15,
                 load_products=True, load_fits=False):
        # Calling the BaseSource init method
        super().__init__(ra, dec, redshift, name, cosmology, load_products, load_fits)

        # Setting up the custom region radius attributes
        if custom_region_radius is not None and custom_region_radius.unit.is_equivalent("kpc"):
            rad = rad_to_ang(custom_region_radius, self._redshift, self._cosmo).to("deg")
            self._custom_region_radius = rad
            self._radii["custom"] = self._custom_region_radius
        elif custom_region_radius is not None and not custom_region_radius.unit.is_equivalent("kpc"):
            self._custom_region_radius = custom_region_radius.to("deg")
            self._radii["custom"] = self._custom_region_radius

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
                warnings.warn("There are {0} alternative matches for observation {1}, associated with "
                              "source {2}".format(len(self._alt_match_regions[o]), o, self.name))

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
            warnings.warn("{n} has not been detected in all region files, so generating and fitting products"
                          " with the 'region' reg_type will not use all available data".format(n=self.name))
        # If the source wasn't detected in ALL of the observations, then we have to rely on a custom region,
        #  and if no custom region options are passed by the user then an error is raised.
        elif all([det is False for det in self._detected.values()]) and self._custom_region_radius is not None:
            warnings.warn("{n} has not been detected in ANY region files, so generating and fitting products"
                          " with the 'region' reg_type will not work".format(n=self.name))
        elif all([det is False for det in self._detected.values()]) and self._custom_region_radius is None:
            raise NoRegionsError("{n} has not been detected in ANY region files, and no custom region radius"
                                 "has been passed. No analysis is possible.".format(n=self.name))

        # Call to a method that goes through all the observations and finds the X-ray centroid. Also at the same
        #  time finds the X-ray centroid of the combined ratemap (an essential piece of information).
        self._all_peaks()
        if self._use_peak:
            self._default_coord = self.peak

    def find_peak(self, rt: RateMap, method: str = "hierarchical", num_iter: int = 20, peak_unit: UnitBase = deg) \
            -> Tuple[Quantity, bool, bool, ndarray, List]:
        """
        A method that will find the X-ray centroid for the RateMap that has been passed in. It takes
        the user supplied coordinates from source initialisation as a starting point, finds the peak within a 500kpc
        radius, re-centres the region, and iterates until the centroid converges to within 15kpc, or until 20
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
                peak, near_edge, chosen_coords, other_coords = rt.clustering_peak(aperture_mask, peak_unit)
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

    def _all_peaks(self):
        """
        An internal method that finds the X-ray peaks for all of the available observations and instruments,
        as well as the combined ratemap. Peak positions for individual ratemap products are allowed to not
        converge, and will just write None to the peak dictionary, but if the peak of the combined ratemap fails
        to converge an error will be thrown. The combined ratemap peak will also be stored by itself in an
        attribute, to allow a property getter easy access.
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
            coord, near_edge, converged, cluster_coords, other_coords = self.find_peak(comb_rt)
            # Updating nH for new coord, probably won't make a difference most of the time
            self._nH = nh_lookup(coord)[0]
        else:
            # If we don't care about peak finding then this is the boi to go for
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
        :param str obs_id: The ObsID to return the X-ray peak coordinates for.
        :param str inst: The instrument to return the X-ray peak coordinates for.
        :return: The X-ray peak coordinates for the input parameters.
        :rtype: Quantity
        """
        # Common sense checks, are the obsids/instruments associated with this source etc.
        if obs_id is not None and obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))
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

    # Property SPECIFICALLY FOR THE COMBINED PEAK - as this is the peak we should be using mostly.
    @property
    def peak(self) -> Quantity:
        """
        A property getter for the combined X-ray peak coordinates. Most analysis will be centered
        on these coordinates.
        :return: The X-ray peak coordinates for the combine ratemap.
        :rtype: Quantity
        """
        return self._peaks["combined"]

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

    def obs_check(self, reg_type: str, threshold_fraction: float = 0.5) -> Dict:
        """
        This method uses exposure maps and region masks to determine which ObsID/instrument combinations
        are not contributing to the analysis. It calculates the area intersection of the mask and exposure
        map, and if (for a given ObsID-Instrument) the ratio of that area to the maximum area calculated
        is less than the threshold fraction, that ObsID-instrument will be included in the returned
        rejection dictionary.
        :param str reg_type: The region type for which to calculate the area intersection.
        :param float threshold_fraction: Area to max area ratios below this value will mean the
        ObsID-Instrument is rejected.
        :return: A dictionary of ObsID keys on the top level, then instruments a level down, that
        should be rejected according to the criteria supplied to this method.
        :rtype: Dict
        """
        # Again don't particularly want to do this local import, but its just easier
        from xga.sas import eexpmap

        # Going to ensure that individual exposure maps exist for each of the ObsID/instrument combinations
        #  first, then checking where the source lies on the exposure map
        eexpmap(self, self._peak_lo_en, self._peak_hi_en)

        extra_key = "bound_{l}-{u}".format(l=self._peak_lo_en.to("keV").value, u=self._peak_hi_en.to("keV").value)

        max_area = 0
        area = {o: {} for o in self.obs_ids}
        for o in self.obs_ids:
            # Exposure maps of the peak finding energy range for this ObsID
            exp_maps = self.get_products("expmap", o, extra_key=extra_key)
            m = self.get_source_mask(reg_type, o, central_coord=self._default_coord)
            for ex in exp_maps:
                # Grabs exposure map data, then alters it so anything that isn't zero is a one
                ex_data = ex.data
                ex_data[ex_data > 0] = 1
                # We do this because it then becomes very easy to calculate the intersection area of the mask
                #  with the XMM chips. Just mask the modified expmap, then sum.
                area[o][ex.instrument] = (ex_data*m).sum()
                # Stores the maximum area intersection, this is used in the threshold calculation
                if area[o][ex.instrument] > max_area:
                    max_area = area[o][ex.instrument]

        # Just in case the maximum area hasn't changed at all...
        if max_area == 0:
            raise NoMatchFoundError("There doesn't appear to be any intersection between any {r} mask and "
                                    "the data from the simple match".format(r=reg_type))

        # Now we know the max intersection area for all data, we can accept or reject particular data
        reject_dict = {}
        for o in area:
            for i in area[o]:
                frac = (area[o][i] / max_area)
                if frac <= threshold_fraction and o not in reject_dict:
                    reject_dict[o] = [i]
                elif frac <= threshold_fraction and o in reject_dict:
                    reject_dict[o].append(i)

        return reject_dict


class PointSource(BaseSource):
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15, load_products=True, load_fits=False):
        raise NotImplementedError("Unfortunately as I specialise in clusters, I haven't put any effort into "
                                  "point sources yet.")
        super().__init__(ra, dec, redshift, name, cosmology, load_products, load_fits)
        # This uses the added context of the type of source to find (or not find) matches in region files
        # This is the internal dictionary where all regions, defined by regfiles or by users, will be stored
        self._regions, self._alt_match_regions, self._other_sources = self._source_type_match("pnt")
