#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/04/2020, 21:44. Copyright (c) David J Turner
import os
import warnings
from itertools import product
from typing import Tuple

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.io import fits
from regions import read_ds9, PixelRegion
from xga import xga_conf
from xga.exceptions import NotAssociatedError, UnknownProductTypeError
from xga.sourcetools import simple_xmm_match, nhlookup
from xga.utils import ENERGY_BOUND_PRODUCTS, ALLOWED_PRODUCTS

warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class BaseSource:
    def __init__(self, ra, dec, redshift=None, name='', cosmology=Planck15):
        self.source_name = name
        self.ra_dec = np.array([ra, dec])
        # Only want ObsIDs, not pointing coordinates as well
        # Don't know if I'll always use the simple method
        self.obs = simple_xmm_match(ra, dec)["ObsID"].values
        # Check in a box of half-side 5 arcminutes, should give an idea of which are on-axis
        on_axis_match = simple_xmm_match(ra, dec, 5)["ObsID"].values
        self.onaxis = np.isin(self.obs, on_axis_match)
        # nhlookup returns average and weighted average values, so just take the first
        self.nH = nhlookup(ra, dec)[0]
        self.redshift = redshift
        self._products = self._initial_products()
        if redshift is not None:
            self.lum_dist = cosmology.luminosity_distance(self.redshift)
        self.regions, self.matches = self._load_regions()
        # Don't know if I'll keep this in, but this sums the live times of all instruments of all observations
        # TODO Perhaps get rid of this, perhaps put in function
        self.livetime = 0
        for oi in product(self._products, ["pn", "mos1", "mos2"]):
            obs_id = oi[0]
            inst = oi[1]
            try:
                with fits.open(self._products[obs_id][inst]["events"]) as evt:
                    self.livetime += evt[1].header["LIVETIME"]
            except KeyError:
                pass

        # TODO Read in images maybe, or at least generate the 'master' image and expmap

    def _initial_products(self) -> dict:
        """
        Assembles the initial dictionary structure of existing XMM data products associated with this source.
        :return: A dictionary structure detailing the data products available at initialisation
        :rtype: dict
        """
        def gen_file_names(en_lims: tuple) -> Tuple[str, dict]:
            """
            This nested function takes pairs of energy limits defined in the config file and runs
            through the XMM products (also defined in the config file), filling in the energy limits and
            checking if the file paths exist. Those that do exist are returned in a dictionary.
            :param tuple en_lims: A tuple containing a lower and upper energy limit to generate file names for,
            the first entry should be the lower limit, the second the upper limit.
            :return: A dictionary key based on the energy limits for the file paths to be stored under, and the
            dictionary of file paths.
            :rtype: tuple[str, dict]
            """
            not_these = ["root_xmm_dir", "lo_en", "hi_en", "region_file", evt_key]
            files = {k.split('_')[1]: v.format(lo_en=en_lims[0], hi_en=en_lims[1], obs_id=obs_id)
                     for k, v in xga_conf["XMM_FILES"].items() if k not in not_these and inst in k}

            existing_files = {key: file for key, file in files.items() if os.path.exists(file)}

            bound_key = "{l}-{u}".format(l=float(en_lims[0]), u=float(en_lims[1]))
            return bound_key, existing_files

        # This dictionary structure will contain paths to all available data products associated with this
        # source instance, both pre-generated and made with XGA.
        obs_dict = {obs: {} for obs in self.obs}

        # Use itertools to create iterable and avoid messy nested for loop
        # product makes iterable of tuples, with all combinations of the events files and ObsIDs
        for oi in product(obs_dict, ["pn", "mos1", "mos2"]):
            # Produces a list of the combinations of upper and lower energy bounds from the config file.
            en_comb = zip(xga_conf["XMM_FILES"]["lo_en"], xga_conf["XMM_FILES"]["hi_en"])

            # This is purely to make the code easier to read
            obs_id = oi[0]
            inst = oi[1]
            evt_key = "clean_{}_evts".format(inst)
            evt_file = xga_conf["XMM_FILES"][evt_key].format(obs_id=obs_id)
            reg_file = xga_conf["XMM_FILES"]["region_file"].format(obs_id=obs_id)

            if os.path.exists(evt_file):
                # An instrument subsection of an observation will ONLY be populated if the events file exists
                # Otherwise nothing can be done with it.
                obs_dict[obs_id][inst] = {"events": evt_file}
                # Dictionary updated with derived product names
                map_ret = map(gen_file_names, en_comb)
                obs_dict[obs_id][inst].update({gen_return[0]: gen_return[1] for gen_return in map_ret})
                if os.path.exists(reg_file):
                    # Dictionary updated with path to region file, if it exists
                    # Want regions top level, as not associated with any one instrument
                    obs_dict[obs_id]["regions"] = reg_file

        # Cleans any observations that don't have at least one instrument associated with them
        obs_dict = {o: v for o, v in obs_dict.items() if len(v) != 0}
        return obs_dict

    def _update_products(self, obs_id: str, inst: str, p_type: str, p_path: str,
                         lo: float = None, hi: float = None):
        """
        Setter method for the products attribute of source objects. Cannot delete existing products,
        but will overwrite existing products with a warning. Raises errors if the ObsID is not associated
        with this source or the instrument is not associated with the ObsID.
        :param str obs_id: A valid XMM ObsID.
        :param str inst: XMM Instrument identifier. Allowed values are pn, mos1, or mos2.
        :param str p_type: Product type identifier. e.g. image or expmap.
        :param str p_path: Product file path.
        :param float lo: The lower energy bound used to create this product. This only needs to
        be supplied for image based products such as images or exposure maps.
        :param float hi: The upper energy bound used to create this product. This only needs to
        be supplied for image based products images or exposure maps.
        """
        # Allowed products are defined in the utils code, lay down the law about what types of products
        # are supported by XGA
        if p_type not in ALLOWED_PRODUCTS:
            raise UnknownProductTypeError()
        elif lo is not None and hi is not None:
            en_key = "{l}-{u}".format(l=float(lo), u=float(hi))
        elif lo is None or hi is None and p_type in ENERGY_BOUND_PRODUCTS:
            raise TypeError("Energy limits cannot be None when writing an energy bound product.")
        else:
            en_key = None

        # Just to be sure
        inst = inst.lower()

        # Don't want to point a source object at a file that doesn't actually exist.
        if not os.path.exists(p_path):
            raise FileNotFoundError("The path to the XMM product does not exist.")

        # Double check that something is trying to add spectra from another source to the current one.
        if obs_id not in self._products:
            raise NotAssociatedError("{o} is not associated with this X-ray source.".format(o=obs_id))
        elif inst not in self._products[obs_id]:
            raise NotAssociatedError("{i} is not associated with XMM observation {o}".format(i=inst, o=obs_id))
        elif en_key is not None and en_key in self._products[obs_id][inst] \
                and p_type in self._products[obs_id][inst][en_key]:
            warnings.warn("You are replacing an existing product associated with this X-ray source.")

        if p_type in ENERGY_BOUND_PRODUCTS:
            self._products[obs_id][inst][en_key][p_type] = p_path
        else:
            # TODO This implementation will have to change when I figure out how to implement
            #  writing groups of spectra.
            self._products[obs_id][inst][p_type] = p_path

    def _load_regions(self) -> Tuple[dict, dict]:
        """
        An internal method that reads and parses region files found for observations
        associated with this source. Also computes simple matches to find regions likely
        to be related to the source.
        :return: Tuple[dict, dict]
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
            return np.sqrt(abs(ra-self.ra_dec[0])**2 + abs(dec-self.ra_dec[1])**2)

        reg_dict = {}
        match_dict = {}
        # As we only allow one set of regions per observation, we shall assume that we can use the
        # WCS transform from ANY of the images to convert pixels to degrees

        for obs_id in self._products:
            if "regions" in self._products[obs_id]:
                ds9_regs = read_ds9(self._products[obs_id]["regions"])
                inst = [k for k in self._products[obs_id] if k in ["pn", "mos1", "mos2"]][0]
                en = [k for k in self._products[obs_id][inst] if "-" in k][0]
                # Making an assumption here, that if there are regions there will be images
                w = wcs.WCS(self._products[obs_id][inst][en]["image"])
                if isinstance(ds9_regs[0], PixelRegion):
                    sky_regs = [reg.to_sky(w) for reg in ds9_regs]
                    reg_dict[obs_id] = np.array(sky_regs)
                else:
                    reg_dict[obs_id] = np.array(ds9_regs)

                # Quickly calculating distance between source and center of regions, then sorting
                # and getting indices. Thus I only match to the closest 5 regions.
                diff_sort = np.array([dist_from_source(r) for r in reg_dict[obs_id]]).argsort()
                within = np.array([reg.contains(SkyCoord(*self.ra_dec, unit='deg'), w)
                                   for reg in reg_dict[obs_id][diff_sort[0:5]]])

                # Expands it so it can be used as a mask on the whole set of regions for this observation
                within = np.pad(within, [0, len(diff_sort)-len(within)])
                match_dict[obs_id] = within
        return reg_dict, match_dict

    def merged_im(self):
        # TODO Remove this experimental crap
        things = []
        hdus = []
        for obs in self._products:
            for ins in self._products[obs]:
                if "0.5-2.0" in self._products[obs][ins] and "image" in self._products[obs][ins]["0.5-2.0"]:
                    things.append(self._products[obs][ins]["0.5-2.0"]["image"])
                    hdus.append(fits.open(things[-1])[0])

        print('emosaic imagesets="'+' '.join(things) + '" mosaicedset="testsmoosh.fits"')

    def info(self):
        """
        Very simple function that just prints a summary of the BaseSource object.
        """
        print("\n-----------------------------------------------------")
        print("Source Name - {}".format(self.source_name))
        print("User Coordinates - ({0}, {1}) degrees".format(*self.ra_dec))
        print("X-ray Peak Coordinates - ({0}, {1}) degrees".format("N/A", "N/A"))
        print("XMM Observations - {}".format(self.__len__()))
        print("On-Axis - {}".format(self.onaxis.sum()))
        print("With regions - {}".format(len(self.regions)))
        print("Total regions - {}".format(sum([len(self.regions[o]) for o in self.regions])))
        print("Obs with one match - {}".format(sum([1 for o in self.matches if self.matches[o].sum() == 1])))
        print("Obs with >1 matches - {}".format(sum([1 for o in self.matches if self.matches[o].sum() > 1])))
        print("Total livetime - {} [seconds]".format(round(self.livetime, 2)))
        print("-----------------------------------------------------\n")

    def __len__(self) -> int:
        """
        Method to return the length of the products dictionary (which means the number of
        individual ObsIDs associated with this source), when len() is called on an instance of this class.
        :return: The integer length of the top level of the _products nested dictionary.
        :rtype: int
        """
        return len(self._products)


class ExtendedSource(BaseSource):
    def __init__(self, ra, dec, redshift=None):
        super().__init__(ra, dec, redshift)


class PointSource(BaseSource):
    def __init__(self, ra, dec, redshift=None):
        super().__init__(ra, dec, redshift)


class GalaxyCluster(ExtendedSource):
    def __init__(self, ra, dec, redshift=None):
        super().__init__(ra, dec, redshift)





