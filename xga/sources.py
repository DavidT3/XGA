#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/04/2020, 21:44. Copyright (c) David J Turner
import os
import warnings
from itertools import product
from typing import Tuple, List

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.units import Quantity
from regions import read_ds9, PixelRegion
from xga import xga_conf
from xga.exceptions import NotAssociatedError, UnknownProductError, NoValidObservationsError
from xga.sourcetools import simple_xmm_match, nhlookup
from xga.utils import ENERGY_BOUND_PRODUCTS, ALLOWED_PRODUCTS, XMM_INST
from xga.products import PROD_MAP, EventList, BaseProduct

import sys
# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class BaseSource:
    def __init__(self, ra, dec, redshift=None, name='', cosmology=Planck15):
        self.source_name = name
        self.ra_dec = np.array([ra, dec])
        # Only want ObsIDs, not pointing coordinates as well
        # Don't know if I'll always use the simple method
        self._obs = simple_xmm_match(ra, dec)["ObsID"].values
        # Check in a box of half-side 5 arcminutes, should give an idea of which are on-axis
        on_axis_match = simple_xmm_match(ra, dec, 5)["ObsID"].values
        self.onaxis = np.isin(self._obs, on_axis_match)
        # nhlookup returns average and weighted average values, so just take the first
        self.nH = nhlookup(ra, dec)[0]
        self.redshift = redshift
        self._products, region_dict, self._att_files, self._odf_paths = self._initial_products()
        # Want to update the ObsIDs associated with this source after seeing if all files are present
        self._obs = list(self._products.keys())
        # This is an important dictionary, mosaiced images and exposure maps will live here, which is what most
        # users should be using for analyses
        self._merged_products = {}

        if redshift is not None:
            self.lum_dist = cosmology.luminosity_distance(self.redshift)
        self.regions, self.matches = self._load_regions(region_dict)

        # This is a queue for products to be generated for this source, will be a numpy array in practise.
        # Items in the same row will all be generated in parallel, whereas items in the same column will
        # be combined into a command stack and run in order.
        self.queue = None
        # Another attribute destined to be an array, will contain the output type of each command submitted to
        # the queue array.
        self.queue_type = None
        # This contains an array of the paths of the final output of each command in the queue
        self.queue_path = None
        # This contains an array of the extra information needed to instantiate class
        # after the SAS command has run
        self.queue_extra_info = None

    # TODO Check for XGA generated products and load them in perhaps.
    def _initial_products(self) -> Tuple[dict, dict, dict, dict]:
        """
        Assembles the initial dictionary structure of existing XMM data products associated with this source.
        :return: A dictionary structure detailing the data products available at initialisation, another
        dictionary containing paths to region files, and another dictionary containing paths to attitude files.
        :rtype: Tuple[dict, dict, dict]
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
            not_these = ["root_xmm_dir", "lo_en", "hi_en", evt_key, "attitude_file", "odf_path"]
            # Formats the generic paths given in the config file for this particular obs and energy range
            files = {k.split('_')[1]: v.format(lo_en=en_lims[0], hi_en=en_lims[1], obs_id=obs_id)
                     for k, v in xga_conf["XMM_FILES"].items() if k not in not_these and inst in k}

            # It is not necessary to check that the files exist, as this happens when the product classes
            # are instantiated. So whether the file exists or not, an object WILL exist, and you can check if
            # you should use it for analysis using the .usable attribute

            # This looks up the class which corresponds to the key (which is the product
            # ID in this case e.g. image), then instantiates an object of that class
            lo = Quantity(float(en_lims[0]), 'keV')
            hi = Quantity(float(en_lims[1]), 'keV')
            prod_objs = {key: PROD_MAP[key](file, obs_id=obs_id, instrument=inst, stdout_str="", stderr_str="",
                                            gen_cmd="", lo_en=lo, hi_en=hi)
                         for key, file in files.items() if os.path.exists(file)}
            # As these files existed already, I don't have any stdout/err strings to pass, also no
            # command string.

            bound_key = "bound_{l}-{u}".format(l=float(en_lims[0]), u=float(en_lims[1]))
            return bound_key, prod_objs

        # This dictionary structure will contain paths to all available data products associated with this
        # source instance, both pre-generated and made with XGA.
        obs_dict = {obs: {} for obs in self._obs}
        # Regions will get their own dictionary, I don't care about keeping the reg_file paths as
        # an attribute because they get read into memory in the init of this class
        reg_dict = {}
        # Attitude files also get their own dictionary, they won't be read into memory by XGA
        att_dict = {}
        # ODF paths also also get their own dict, they will just be used to point cifbuild to the right place
        odf_dict = {}

        # Use itertools to create iterable and avoid messy nested for loop
        # product makes iterable of tuples, with all combinations of the events files and ObsIDs
        for oi in product(obs_dict, XMM_INST):
            # Produces a list of the combinations of upper and lower energy bounds from the config file.
            en_comb = zip(xga_conf["XMM_FILES"]["lo_en"], xga_conf["XMM_FILES"]["hi_en"])

            # This is purely to make the code easier to read
            obs_id = oi[0]
            inst = oi[1]
            evt_key = "clean_{}_evts".format(inst)
            evt_file = xga_conf["XMM_FILES"][evt_key].format(obs_id=obs_id)
            reg_file = xga_conf["XMM_FILES"]["region_file"].format(obs_id=obs_id)

            # Attitude file is a special case of data product, only SAS should ever need it, so it doesn't
            # have a product object
            att_file = xga_conf["XMM_FILES"]["attitude_file"].format(obs_id=obs_id)
            # ODF path isn't a data product, but is necessary for cifbuild
            odf_path = xga_conf["XMM_FILES"]["odf_path"].format(obs_id=obs_id)

            if os.path.exists(evt_file) and os.path.exists(att_file) and os.path.exists(odf_path):
                # An instrument subsection of an observation will ONLY be populated if the events file exists
                # Otherwise nothing can be done with it.
                obs_dict[obs_id][inst] = {"events": EventList(evt_file, obs_id=obs_id, instrument=inst,
                                                              stdout_str="", stderr_str="", gen_cmd="")}
                att_dict[obs_id] = att_file
                odf_dict[obs_id] = odf_path
                # Dictionary updated with derived product names
                map_ret = map(gen_file_names, en_comb)
                obs_dict[obs_id][inst].update({gen_return[0]: gen_return[1] for gen_return in map_ret})
                if os.path.exists(reg_file):
                    # Regions dictionary updated with path to region file, if it exists
                    reg_dict[obs_id] = reg_file

        # Cleans any observations that don't have at least one instrument associated with them
        obs_dict = {o: v for o, v in obs_dict.items() if len(v) != 0}
        if len(obs_dict) == 0:
            raise NoValidObservationsError("No matching observations have the necessary files.")
        return obs_dict, reg_dict, att_dict, odf_dict

    def update_products(self, prod_obj: BaseProduct):
        """
        Setter method for the products attribute of source objects. Cannot delete existing products,
        but will overwrite existing products with a warning. Raises errors if the ObsID is not associated
        with this source or the instrument is not associated with the ObsID.
        :param BaseProduct prod_obj: The new product object to be added to the source object.
        """
        if not isinstance(prod_obj, BaseProduct):
            raise TypeError("Only product objects can be assigned to sources.")

        en_bnds = prod_obj.energy_bounds
        if en_bnds[0] is not None and en_bnds[1] is not None:
            en_key = "bound_{l}-{u}".format(l=float(en_bnds[0].value), u=float(en_bnds[1].value))
        else:
            en_key = None

        # All information about where to place it in our storage hierarchy can be pulled from the product
        # object itself
        obs_id = prod_obj.obs_id
        inst = prod_obj.instrument
        p_type = prod_obj.type

        # Double check that something is trying to add products from another source to the current one.
        if obs_id != 'all' and obs_id not in self._products:
            raise NotAssociatedError("{o} is not associated with this X-ray source.".format(o=obs_id))
        elif inst != 'all' and inst not in self._products[obs_id]:
            raise NotAssociatedError("{i} is not associated with XMM observation {o}".format(i=inst, o=obs_id))

        if en_key is not None and obs_id != 'all':
            # If there is no entry for this energy band already, we must make one
            if en_key not in self._products[obs_id][inst]:
                self._products[obs_id][inst][en_key] = {}
            self._products[obs_id][inst][en_key][p_type] = prod_obj
        elif en_key is None and obs_id != 'all':
            self._products[obs_id][inst][p_type] = prod_obj
        # Here we deal with merged products
        elif en_key is not None and obs_id == 'all':
            # If there is no entry for this energy band already, we must make one
            if en_key not in self._merged_products:
                self._merged_products[en_key] = {}
            self._merged_products[en_key][p_type] = prod_obj
        elif en_key is None and obs_id == 'all':
            self._merged_products[p_type] = prod_obj

    def get_products(self, p_type: str, obs_id: str = None, inst: str = None) -> List[list]:
        """
        This is the getter for the products data structure of Source objects. Passing a 'product type'
        such as 'events' or 'images' will return every matching entry in the products data structure.
        :param str p_type: Product type identifier. e.g. image or expmap.
        :param str obs_id: Optionally, a specific obs_id to search can be supplied.
        :param str inst: Optionally, a specific instrument to search can be supplied.
        :return: List of matching products.
        :rtype: List[list]
        """
        def dict_search(key: str, var: dict) -> list:
            """
            This simple function was very lightly modified from a stackoverflow answer, and is an
            efficient method of searching through a nested dictionary structure for specfic keys
            (and yielding the values associated with them). In this case will extract all of a
            specific product type for a given source.
            :param key: The key in the dictionary to search for and extract values.
            :param var: The variable to search, likely to be either a dictionary or a string.
            :return list[list]: Returns information on keys and values
            """

            # Check that the input is actually a dictionary
            if hasattr(var, 'items'):
                for k, v in var.items():
                    if k == key:
                        yield v
                    # Here is where we dive deeper, recursively searching lower dictionary levels.
                    if isinstance(v, dict):
                        for result in dict_search(key, v):
                            # We yield a string of the result and the key, as we'll need to return the
                            # ObsID and Instrument information from these product searches as well.
                            # This will mean the output is an unpleasantly nested list, but we can solve that.
                            yield [str(k), result]

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

        # Only certain product identifier are allowed
        if p_type not in ALLOWED_PRODUCTS:
            raise UnknownProductError("Requested product type not recognised.")
        elif obs_id not in self._products and obs_id is not None:
            raise NotAssociatedError("{} is not associated with this source.".format(obs_id))
        elif inst not in XMM_INST and inst is not None:
            raise ValueError("{} is not an allowed instrument".format(inst))

        matches = []
        # Iterates through the dict search return, but each match is likely to be a very nested list,
        # with the degree of nesting dependant on product type (as event lists live a level up from
        # images for instance
        for match in dict_search(p_type, self._products):
            out = []
            unpack_list(match)
            # Only appends if this particular match is for the obs_id and instrument passed to this method
            # Though all matches will be returned if no obs_id/inst is passed
            if (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None):
                matches.append(out)
        return matches

    def _load_regions(self, reg_paths) -> Tuple[dict, dict]:
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

        for obs_id in reg_paths:
            ds9_regs = read_ds9(reg_paths[obs_id])
            inst = [k for k in self._products[obs_id] if k in ["pn", "mos1", "mos2"]][0]
            en = [k for k in self._products[obs_id][inst] if "-" in k][0]
            # Making an assumption here, that if there are regions there will be images
            # Getting the radec_wcs property from the Image object
            w = self._products[obs_id][inst][en]["image"].radec_wcs
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
            # Use the deleter for the hdulist to unload the astropy hdulist for this image
            del self._products[obs_id][inst][en]["image"].hdulist
        return reg_dict, match_dict

    def update_queue(self, cmd_arr: np.ndarray, p_type_arr: np.ndarray, p_path_arr: np.ndarray,
                     extra_info: np.ndarray, stack: bool = False):
        """
        Small function to update the numpy array that makes up the queue of products to be generated.
        :param np.ndarray cmd_arr: Array containing SAS commands.
        :param np.ndarray p_type_arr: Array of product type identifiers for the products generated
        by the cmd array. e.g. image or expmap.
        :param np.ndarray p_path_arr: Array of final product paths if cmd is successful
        :param np.ndarray extra_info: Array of extra information dictionaries
        :param stack: Should these commands be executed after a preceding line of commands,
        or at the same time.
        :return:
        """
        if self.queue is None:
            # I could have done all of these in one array with 3 dimensions, but felt this was easier to read
            # and with no real performance penalty
            self.queue = cmd_arr
            self.queue_type = p_type_arr
            self.queue_path = p_path_arr
            self.queue_extra_info = extra_info
        elif stack:
            self.queue = np.vstack((self.queue, cmd_arr))
            self.queue_type = np.vstack((self.queue_type, p_type_arr))
            self.queue_path = np.vstack((self.queue_path, p_path_arr))
            self.queue_extra_info = np.vstack((self.queue_extra_info, extra_info))
        else:
            self.queue = np.append(self.queue, cmd_arr, axis=0)
            self.queue_type = np.append(self.queue_type, p_type_arr, axis=0)
            self.queue_path = np.append(self.queue_path, p_path_arr, axis=0)
            self.queue_extra_info = np.append(self.queue_extra_info, extra_info, axis=0)

    def get_queue(self) -> Tuple[List[str], List[str], List[List[str]], List[dict]]:
        """
        Calling this indicates that the queue is about to be processed, so this function combines SAS
        commands along columns (command stacks), and returns N SAS commands to be run concurrently,
        where N is the number of columns.
        :return: List of strings, where the strings are bash commands to run SAS procedures, another
        list of strings, where the strings are expected output types for the commands, a list of
        lists of strings, where the strings are expected output paths for products of the SAS commands.
        :rtype: Tuple[List[str], List[str], List[List[str]]]
        """
        if self.queue is None:
            # This returns empty lists if the queue is undefined
            processed_cmds = []
            types = []
            paths = []
            extras = []
        elif len(self.queue.shape) == 1 or self.queue.shape[1] <= 1:
            processed_cmds = list(self.queue)
            types = list(self.queue_type)
            paths = [[str(path)] for path in self.queue_path]
            extras = list(self.queue_extra_info)
        else:
            processed_cmds = [";".join(col) for col in self.queue.T]
            types = list(self.queue_type[-1, :])
            paths = [list(col.astype(str)) for col in self.queue_path.T]
            extras = []
            for col in self.queue_path.T:
                # This nested dictionary comprehension combines a column of extra information
                # dictionaries into one, for ease of access.
                comb_extra = {k: v for ext_dict in col for k, v in ext_dict.items()}
                extras.append(comb_extra)

        # This is only likely to be called when processing is beginning, so this will wipe the queue.
        self.queue = None
        self.queue_type = None
        self.queue_path = None
        self.queue_extra_info = None
        # The returned paths are lists of strings because we want to include every file in a stack to be able
        # to check that exists
        return processed_cmds, types, paths, extras

    def get_att_file(self, obs_id: str) -> str:
        """
        Fetches the path to the attitude file for an XMM observation.
        :param obs_id: The ObsID to fetch the attitude file for.
        :return: The path to the attitude file.
        :rtype: str
        """
        if obs_id not in self._products:
            raise NotAssociatedError("{} is not associated with this source".format(obs_id))
        else:
            return self._att_files[obs_id]

    def get_odf_path(self, obs_id: str) -> str:
        """
        Fetches the path to the odf directory for an XMM observation.
        :param obs_id: The ObsID to fetch the ODF path for.
        :return: The path to the ODF path.
        :rtype: str
        """
        if obs_id not in self._products:
            raise NotAssociatedError("{} is not associated with this source".format(obs_id))
        else:
            return self._odf_paths[obs_id]

    @property
    def obs_ids(self) -> List[str]:
        """
        Property getter for ObsIDs associated with this source that are confirmed to have events files.
        :return: A list of the associated XMM ObsIDs.
        :rtype: List[str]
        """
        return self._obs

    def info(self):
        """
        Very simple function that just prints a summary of the BaseSource object.
        """
        print("-----------------------------------------------------")
        print("Source Name - {}".format(self.source_name))
        print("User Coordinates - ({0}, {1}) degrees".format(*self.ra_dec))
        print("X-ray Peak Coordinates - ({0}, {1}) degrees".format("N/A", "N/A"))
        print("XMM Observations - {}".format(self.__len__()))
        print("On-Axis - {}".format(self.onaxis.sum()))
        print("With regions - {}".format(len(self.regions)))
        print("Total regions - {}".format(sum([len(self.regions[o]) for o in self.regions])))
        print("Obs with one match - {}".format(sum([1 for o in self.matches if self.matches[o].sum() == 1])))
        print("Obs with >1 matches - {}".format(sum([1 for o in self.matches if self.matches[o].sum() > 1])))
        # print("Total livetime - {} [seconds]".format(round(self.livetime, 2)))
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





