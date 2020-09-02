#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 02/09/2020, 14:05. Copyright (c) David J Turner

import warnings
from typing import Tuple, List, Dict

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.units import Quantity, UnitBase, deg, UnitConversionError
from numpy import ndarray
from regions import SkyRegion, EllipseSkyRegion, CircleSkyRegion, \
    EllipsePixelRegion, CirclePixelRegion

from xga.exceptions import NotAssociatedError, PeakConvergenceFailedError, NoRegionsError, NoMatchFoundError
from xga.products import Image, RateMap
from xga.sourcetools import rad_to_ang, ang_to_rad
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

        # Setting up a bunch of attributes
        self._custom_region_radius = custom_region_radius
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

        # Run through any alternative matches and raise warnings if there are alternative matches
        for o in self._alt_match_regions:
            if len(self._alt_match_regions[o]) > 0:
                warnings.warn("There are {0} alternative matches for observation "
                              "{1}".format(len(self._alt_match_regions[o]), o))

        # Here we figure out what other sources are within the chosen extended source region
        self._within_source_regions = {}
        self._back_regions = {}
        self._within_back_regions = {}
        self._reg_masks = {obs: {inst: {} for inst in self._products[obs]} for obs in self.obs_ids}
        self._back_masks = {obs: {inst: {} for inst in self._products[obs]} for obs in self.obs_ids}
        # Iterating through obs_ids rather than _region keys because the _region dictionary will contain
        #  a combined region that cannot be used yet - the user cannot have generated any merged images yet.
        for obs_id in self.obs_ids:
            match_reg = self._regions[obs_id]
            # If the entry here is None, it means the source wasn't detected in the region files
            if match_reg is not None:
                other_regs = self._other_regions[obs_id]
                im = list(self.get_products("image", obs_id, just_obj=True))[0]

                m = match_reg.to_pixel(im.radec_wcs)
                crossover = np.array([match_reg.intersection(r).to_pixel(im.radec_wcs).to_mask().data.sum() != 0
                                      for r in other_regs])
                self._within_source_regions[obs_id] = np.array(other_regs)[crossover]

                # Here is where we initialise the background regions, first in pixel coords, then converting
                #  to ra-dec and adding to a dictionary of regions.
                if isinstance(match_reg, EllipseSkyRegion):
                    # Here we multiply the inner width/height by 1.05 (to just slightly clear the source region),
                    #  and the outer width/height by 1.5 (standard for XCS) - default values
                    # Ideally this would be an annulus region, but they are bugged in regions v0.4, so we must bodge
                    in_reg = EllipsePixelRegion(m.center, m.width * self._back_inn_factor,
                                                m.height * self._back_inn_factor, m.angle)
                    b_reg = EllipsePixelRegion(m.center, m.width * self._back_out_factor,
                                               m.height * self._back_out_factor,
                                               m.angle).symmetric_difference(in_reg)
                elif isinstance(match_reg, CircleSkyRegion):
                    in_reg = CirclePixelRegion(m.center, m.radius * self._back_inn_factor)
                    b_reg = CirclePixelRegion(m.center, m.radius *
                                              self._back_out_factor).symmetric_difference(in_reg)

                self._back_regions[obs_id] = b_reg.to_sky(im.radec_wcs)
                # This part is dealing with the region in sky coordinates,
                b_reg = self._back_regions[obs_id]
                crossover = np.array([b_reg.intersection(r).to_pixel(im.radec_wcs).to_mask().data.sum() != 0
                                      for r in other_regs])
                self._within_back_regions[obs_id] = np.array(other_regs)[crossover]
                # Ensures we only do regions for instruments that do have at least an events list.
                for inst in self._products[obs_id]:
                    cur_im = self.get_products("image", obs_id, inst)[0]
                    src_reg, bck_reg = self.get_source_region("region", obs_id)
                    self._reg_masks[obs_id][inst], self._back_masks[obs_id][inst] \
                        = self._generate_mask(cur_im, src_reg, bck_reg)

            else:
                # Fill out all the various region dictionaries with Nones for when a source isn't detected
                self._within_source_regions[obs_id] = np.array([])
                self._back_regions[obs_id] = None
                self._within_back_regions[obs_id] = np.array([])
                for inst in self._products[obs_id]:
                    self._reg_masks[obs_id][inst] = None
                    self._back_masks[obs_id][inst] = None

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

        # Constructs the custom region and adds to existing storage structure, if the user wants a custom region
        if self._custom_region_radius is not None:
            self._setup_new_region(self._custom_region_radius, "custom")
            # Doesn't really matter where this conversion happens, because setup_new_region checks the unit
            #  and converts anyway, but I want the internal unit of the custom radii to be degrees.
            # Originally was meant to be kpc, but then I realised that ExtendedSources are allowed to not have
            #  redshift information
            if self._custom_region_radius.unit.is_equivalent("kpc"):
                rad = rad_to_ang(self._custom_region_radius, self._redshift, self._cosmo).to("deg")
                self._custom_region_radius = rad
            else:
                self._custom_region_radius = self._custom_region_radius.to("deg")

    def _generate_mask(self, mask_image: Image, source_region: SkyRegion, back_reg: SkyRegion = None,
                       reg_type: str = None) -> Tuple[ndarray, ndarray]:
        """
        This uses available region files to generate a mask for the source region in the form of a
        numpy array. It takes into account any sources that were detected within the target source,
        by drilling them out.
        :param Image mask_image: An XGA image object that donates its WCS to convert SkyRegions to pixels.
        :param SkyRegion source_region: The SkyRegion containing the source to generate a mask for.
        :param SkyRegion back_reg: The SkyRegion containing the background emission to
        generate a mask for.
        :param bool reg_type: By default this is None, but if supplied by the user then this method
        will look for interloper regions under reg_type key rather than the ObsID key.
        :return: A boolean numpy array that can be used to mask images loaded in as numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        obs_id = mask_image.obs_id
        mask = source_region.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)

        if reg_type is None:
            # Now need to drill out any interloping sources, make a mask for that
            interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                               for reg in self._within_source_regions[obs_id]])
        else:
            interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                               for reg in self._within_source_regions[reg_type]])

        # Wherever the interloper src_mask is not 0, the global src_mask must become 0 because there is an
        # interloper source there - circular sentences ftw
        mask[interlopers != 0] = 0

        if back_reg is not None:
            back_mask = back_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
            if reg_type is None:
                # Now need to drill out any interloping sources, make a mask for that
                interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                                   for reg in self._within_back_regions[obs_id]])
            else:
                interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                                   for reg in self._within_back_regions[reg_type]])

            # Wherever the interloper src_mask is not 0, the global src_mask must become 0 because there is an
            # interloper source there - circular sentences ftw
            back_mask[interlopers != 0] = 0

            return mask, back_mask
        return mask

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

        # 500kpc in degrees, for the current redshift and cosmology
        #  Or 5 arcminutes if no redshift information is present (that is allowed for the ExtendedSource class)
        if self._redshift is not None:
            search_aperture = rad_to_ang(Quantity(500, "kpc"), self._redshift, cosmo=self._cosmo)
        else:
            search_aperture = Quantity(5, 'arcmin').to('deg')

        # Iteration counter just to kill it if it doesn't converge
        count = 0
        # Allow 20 iterations by default before we kill this - alternatively loop will exit when centre converges
        #  to within 15kpc (or 0.15arcmin).
        while count < num_iter:
            # Define a 500kpc radius region centered on the current central_coords
            cust_reg = CircleSkyRegion(central_coords, search_aperture)
            pix_cust_reg = cust_reg.to_pixel(rt.radec_wcs)

            # Setting up useful lists for adding regions to
            reg_crossover = []
            # I check through all available region lists to find regions that are within the custom region
            for obs_id in self._other_regions:
                other_regs = self._other_regions[obs_id]

                cross = np.array([cust_reg.intersection(r).to_pixel(rt.radec_wcs).to_mask().data.sum()
                                  != 0 and r.height == r.width for r in other_regs])

                if len(cross) != 0:
                    reg_crossover += list(np.array(other_regs)[cross])

            reg_crossover = np.array(reg_crossover)
            self._within_source_regions["search_aperture"] = reg_crossover
            # Generate the source mask for the peak finding method
            aperture_mask = self._generate_mask(rt, cust_reg, reg_type="search_aperture")

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

        del self._within_source_regions["search_aperture"]
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
            emosaic(self, "image", self._peak_lo_en, self._peak_hi_en)
            emosaic(self, "expmap", self._peak_lo_en, self._peak_hi_en)
            comb_rt = [rt[-1] for rt in self.get_products("combined_ratemap", just_obj=False) if en_key in rt][0]

        if self._use_peak:
            coord, near_edge, converged, cluster_coords, other_coords = self.find_peak(comb_rt)
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

        # TODO Decide what to do with this - see issue #85 for a description of why I'm not currently measuring
        #  the individual peaks.
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

    def _setup_new_region(self, radius: Quantity, reg_type: str):
        """
        This method is used to construct a new region (for instance 'custom' or 'r500'), using the a
        radius passed in by the user. If the user also decided to use the X-ray peak as the centre of the
        custom region, it will do iterative peak finding and re-centre the region. It then adds the region
        objects and peripheral information into the existing storage structures.
        :param Quantity radius: The radius of the new region being created.
        :param str reg_type: The type of new region to be created.
        """
        # Start off with the central coordinates of the custom region as the user's passed RA and DEC
        central_coords = SkyCoord(*self.ra_dec.to("deg"))

        # If a custom region radius is passed, then we define one, though we also need to convert
        #  whatever the input units are to degrees
        if radius.unit.is_equivalent('deg'):
            cust_reg = CircleSkyRegion(central_coords, radius.to('deg'))
        # As we need radius in degrees, and we need an angular diameter distance to convert to degrees from
        #  other units, we throw an error if there is no redshift.
        elif radius.unit.is_equivalent('kpc') and self.redshift is None:
            raise UnitConversionError("As you have not supplied a redshift, custom_region_radius can "
                                      "only be in degrees")
        elif radius.unit.is_equivalent('kpc') and self.redshift is not None:
            # Use a handy function I prepared earlier to go to degrees
            region_radius = rad_to_ang(radius, self._redshift, cosmo=self._cosmo)
            radius = region_radius.copy()
            cust_reg = CircleSkyRegion(central_coords, region_radius)
        else:
            raise UnitConversionError("Custom region radius must be in either angular or distance units.")

        # Find a suitable combined ratemap - I've decided this custom region (global region if you will)
        #  will be based around the use of complete products.
        en_key = "bound_{l}-{u}".format(l=self._peak_lo_en.value, u=self._peak_hi_en.value)
        # This should be guaranteed to exist by now, the _all_peaks method requires this product too
        comb_rt = [rt[-1] for rt in self.get_products("combined_ratemap", just_obj=False) if en_key in rt][0]

        # Determine if the initial coordinates are near an edge
        near_edge = comb_rt.near_edge(self.ra_dec)

        if self._use_peak:
            # Uses the peak of the combined ratemap as the centre, guaranteed to be there and converged,
            #  because if it hadn't converged an error would have been thrown earlier
            peak = self._peaks["combined"]
            central_coords = SkyCoord(*peak.to("deg"))
            cust_reg = CircleSkyRegion(central_coords, radius)

        # Define a background region
        # Annoyingly I can't remember why I had to do the regions as pixel first, but I promise there was
        #  a good reason at the time.
        pix_src_reg = cust_reg.to_pixel(comb_rt.radec_wcs)
        in_reg = CirclePixelRegion(pix_src_reg.center, pix_src_reg.radius * self._back_inn_factor)
        pix_bck_reg = CirclePixelRegion(pix_src_reg.center, pix_src_reg.radius
                                        * self._back_out_factor).symmetric_difference(in_reg)
        cust_back_reg = pix_bck_reg.to_sky(comb_rt.radec_wcs)

        # Setting up useful lists for adding regions to
        reg_crossover = []
        bck_crossover = []
        # I check through all available region lists to find regions that are within the custom region
        for obs_id in self._other_regions:
            other_regs = self._other_regions[obs_id]

            # Which regions are within the custom source region
            # Also are any regions that intersect with the custom region (could be an overdensity region) extended?
            #  If so they should be removed - far more likely that the cluster has been
            #  fragmented by the source finder.
            cross = np.array([cust_reg.intersection(r).to_pixel(comb_rt.radec_wcs).to_mask().data.sum()
                              != 0 and r.height == r.width for r in other_regs])

            if len(cross) != 0:
                reg_crossover += list(np.array(other_regs)[cross])

            # Which regions are within the custom background region
            bck_cross = np.array([cust_back_reg.intersection(r).to_pixel(comb_rt.radec_wcs).to_mask().data.sum()
                                  != 0 for r in other_regs])
            if len(bck_cross) != 0:
                bck_crossover += list(np.array(other_regs)[bck_cross])

        # Just quickly convert the lists to numpy arrays
        reg_crossover = np.array(reg_crossover)
        bck_crossover = np.array(bck_crossover)
        # And save them
        self._within_source_regions[reg_type] = reg_crossover
        self._within_back_regions[reg_type] = bck_crossover

        # Make the final masks for source and background regions.
        src_mask, bck_mask = self._generate_mask(comb_rt, cust_reg, cust_back_reg, reg_type=reg_type)

        # TODO Check that this isn't bollocks
        src_area = src_mask.sum()
        bck_area = bck_mask.sum()
        rate_ratio = ((comb_rt.data * src_mask).sum() / (comb_rt.data * bck_mask).sum()) * (bck_area / src_area)

        self._regions[reg_type] = cust_reg
        self._back_regions[reg_type] = cust_back_reg
        self._reg_masks[reg_type] = src_mask
        self._back_masks[reg_type] = bck_mask
        self._snr[reg_type] = rate_ratio

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
            for ex in exp_maps:
                # In an ideal world I'd only need to generate one mask per image, which would be fine if
                #  I can guarantee that the images are in sky coordinates, I can't though
                m = self._generate_mask(ex, self.get_source_region(reg_type)[0])

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
        raise NotImplementedError("Unfortunately as I specialise in clusters, I haven't put any effort into"
                                  "point sources yet.")
        super().__init__(ra, dec, redshift, name, cosmology, load_products, load_fits)
        # This uses the added context of the type of source to find (or not find) matches in region files
        # This is the internal dictionary where all regions, defined by regfiles or by users, will be stored
        self._regions, self._alt_match_regions, self._other_sources = self._source_type_match("pnt")
