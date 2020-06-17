#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 17/06/2020, 13:32. Copyright (c) David J Turner
import os
import warnings
from itertools import product
from typing import Tuple, List, Dict

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.cosmology.core import Cosmology
from astropy.units import Quantity, UnitBase
from regions import read_ds9, PixelRegion, SkyRegion, EllipseSkyRegion, CircleSkyRegion, \
    EllipsePixelRegion, CirclePixelRegion, CompoundSkyRegion
from xga import xga_conf
from xga.exceptions import NotAssociatedError, UnknownProductError, NoValidObservationsError, \
    MultipleMatchError, NoProductAvailableError, NoMatchFoundError
from xga.products import PROD_MAP, EventList, BaseProduct, Image, Spectrum, ExpMap
from xga.sourcetools import simple_xmm_match, nhlookup
from xga.utils import ALLOWED_PRODUCTS, XMM_INST, dict_search, xmm_det, xmm_sky, OUTPUT

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class BaseSource:
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15):
        self._ra_dec = np.array([ra, dec])
        if name is not None:
            self._name = name
        else:
            # self.ra_dec rather than _ra_dec because ra_dec is in astropy degree units
            s = SkyCoord(ra=self.ra_dec[0], dec=self.ra_dec[1])
            crd_str = s.to_string("hmsdms").replace("h", "").replace("m", "").replace("s", "").replace("d", "")
            ra_str, dec_str = crd_str.split(" ")
            # Use the standard naming convention if one wasn't passed on initialisation of the source
            # Need it because its used for naming files later on.
            self._name = "J" + ra_str[:ra_str.index(".")+2] + dec_str[:dec_str.index(".")+2]

        # Only want ObsIDs, not pointing coordinates as well
        # Don't know if I'll always use the simple method
        self._obs = simple_xmm_match(ra, dec)["ObsID"].values
        # Check in a box of half-side 5 arcminutes, should give an idea of which are on-axis
        try:
            on_axis_match = simple_xmm_match(ra, dec, 5)["ObsID"].values
        except NoMatchFoundError:
            on_axis_match = np.array([])
        self._onaxis = np.isin(self._obs, on_axis_match)
        # nhlookup returns average and weighted average values, so just take the first
        self._nH = nhlookup(ra, dec)[0]
        self._redshift = redshift
        self._products, region_dict, self._att_files, self._odf_paths = self._initial_products()

        # If there is an existing XGA output directory, then it makes sense to search for products that XGA
        #  may have already generated and load them in - saves us wasting time making them again.
        if os.path.exists(OUTPUT):
            self._existing_xga_products()

        # Want to update the ObsIDs associated with this source after seeing if all files are present
        self._obs = list(self._products.keys())
        # This is an important dictionary, mosaic images and exposure maps will live here, which is what most
        # users should be using for analyses
        self._merged_products = {}  # TODO Should this just be a part of _products?

        self._cosmo = cosmology
        if redshift is not None:
            self.lum_dist = self._cosmo.luminosity_distance(self._redshift)
        self._initial_regions, self._initial_region_matches = self._load_regions(region_dict)

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
        # Defining this here, although it won't be set to a boolean value in this superclass
        self._detected = None
        # This block defines various dictionaries that are used in the sub source classes, when context allows
        # us to find matching source regions.
        self._regions = None
        self._back_regions = None
        self._other_regions = None
        self._alt_match_regions = None
        self._reg_masks = None
        self._back_masks = None
        self._within_source_regions = None
        self._within_back_regions = None

    @property
    def ra_dec(self) -> Quantity:
        """
        A getter for the original ra and dec entered by the user.
        :return: The ra-dec coordinates entered by the user when the source was first defined
        :rtype: Quantity
        """
        # Easier for it be internally kep as a numpy array, but I want the user to have astropy coordinates
        return Quantity(self._ra_dec, 'deg')

    def _initial_products(self) -> Tuple[dict, dict, dict, dict]:
        """
        Assembles the initial dictionary structure of existing XMM data products associated with this source.
        :return: A dictionary structure detailing the data products available at initialisation, another
        dictionary containing paths to region files, and another dictionary containing paths to attitude files.
        :rtype: Tuple[dict, dict, dict]
        """

        def read_default_products(en_lims: tuple) -> Tuple[str, dict]:
            """
            This nested function takes pairs of energy limits defined in the config file and runs
            through the default XMM products defined in the config file, filling in the energy limits and
            checking if the file paths exist. Those that do exist are read into the relevant product object and
            returned.
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
                map_ret = map(read_default_products, en_comb)
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
            extra_key = "bound_{l}-{u}".format(l=float(en_bnds[0].value), u=float(en_bnds[1].value))
        elif type(prod_obj) == Spectrum:
            extra_key = prod_obj.reg_type
        else:
            extra_key = None

        # All information about where to place it in our storage hierarchy can be pulled from the product
        # object itself
        obs_id = prod_obj.obs_id
        inst = prod_obj.instrument
        p_type = prod_obj.type

        # The product gets the name of this source object added to it
        prod_obj.obj_name = self.name

        # TODO This will need to be able to write spectra to a region type as well, once that's implemented

        # Double check that something is trying to add products from another source to the current one.
        if obs_id != "combined" and obs_id not in self._products:
            raise NotAssociatedError("{o} is not associated with this X-ray source.".format(o=obs_id))
        elif inst != "combined" and inst not in self._products[obs_id]:
            raise NotAssociatedError("{i} is not associated with XMM observation {o}".format(i=inst, o=obs_id))

        if extra_key is not None and obs_id != "combined":
            # If there is no entry for this energy band already, we must make one
            if extra_key not in self._products[obs_id][inst]:
                self._products[obs_id][inst][extra_key] = {}
            self._products[obs_id][inst][extra_key][p_type] = prod_obj
        elif extra_key is None and obs_id != "combined":
            self._products[obs_id][inst][p_type] = prod_obj
        # Here we deal with merged products
        elif extra_key is not None and obs_id == "combined":
            # If there is no entry for this energy band already, we must make one
            if extra_key not in self._merged_products:
                self._merged_products[extra_key] = {}
            self._merged_products[extra_key][p_type] = prod_obj
        elif extra_key is None and obs_id == "combined":
            self._merged_products[p_type] = prod_obj

    # TODO MAKE IT ACTUALLY LOAD IN OLD FITS
    def _existing_xga_products(self):
        """
        A method specifically for searching an existing XGA output directory for relevant files and loading
        them in as XGA products. This will retrieve images, exposure maps, and spectra; then the source product
        structure is updated. The method also finds previous fit results and loads them in.
        """
        def parse_image_like(file_path: str, exact_type: str) -> BaseProduct:
            """
            Very simple little function that takes the path to an XGA generated image-like product (so either an
            image or an exposure map), parses the file path and makes an XGA object of the correct type by using
            the exact_type variable.
            :param file_path: Absolute path to an XGA-generated XMM data product.
            :param exact_type: Either 'image' or 'expmap', the type of product that the file_path leads to.
            :return: An XGA product object.
            :rtype: BaseProduct
            """
            # Get rid of the absolute part of the path, then split by _ to get the information from the file name
            im_info = file_path.split("/")[-1].split("_")
            # I know its hard coded but this will always be the case, these are files I generate with XGA.
            ins = im_info[1]
            lo_en, hi_en = im_info[-1].split("keV")[0].split("-")
            # Have to be astropy quantities before passing them into the Product declaration
            lo_en = Quantity(float(lo_en), "keV")
            hi_en = Quantity(float(hi_en), "keV")

            # Different types of Product objects, the empty strings are because I don't have the stdout, stderr,
            #  or original commands for these objects.
            if exact_type == "image":
                final_obj = Image(im, obs, ins, "", "", "", lo_en, hi_en)
            elif exact_type == "expmap":
                final_obj = ExpMap(im, obs, ins, "", "", "", lo_en, hi_en)
            else:
                raise TypeError("Only image and expmap are allowed.")

            return final_obj

        og_dir = os.getcwd()
        for obs in self._obs:
            if os.path.exists(OUTPUT + obs):
                os.chdir(OUTPUT + obs)
                # I've put as many checks as possible in this to make sure it only finds genuine XGA files,
                #  I'll probably put a few more checks later

                # Images read in, pretty simple process - the name of the current source doesn't matter because
                #  standard images/exposure maps are for the WHOLE observation.
                ims = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f) and
                       "img" in f and obs in f and (XMM_INST[0] in f or XMM_INST[1] in f or XMM_INST[2] in f)]
                for im in ims:
                    self.update_products(parse_image_like(im, "image"))

                # Exposure maps read in, same process as images
                exs = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f) and
                       "expmap" in f and obs in f and (XMM_INST[0] in f or XMM_INST[1] in f or XMM_INST[2] in f)]
                for ex in exs:
                    self.update_products(parse_image_like(ex, "expmap"))

                # For spectra we search for products that have the name of this object in, as they are for
                #  specific parts of the observation.
                named = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f) and self._name in f
                         and obs in f and (XMM_INST[0] in f or XMM_INST[1] in f or XMM_INST[2] in f)]
                specs = [f for f in named if "spec" in f and "back" not in f and "ann" not in f]
                for sp in specs:
                    # Filename contains a lot of useful information, so splitting it out to get it
                    sp_info = sp.split("/")[-1].split("_")
                    inst = sp_info[1]
                    reg_type = sp_info[-2]
                    # Fairly self explanatory, need to find all the separate products needed to define an XGA
                    #  spectrum
                    arf = [f for f in named if "arf" in f and "ann" not in f and "back" not in f
                           and inst in f and reg_type in f]
                    rmf = [f for f in named if "rmf" in f and "ann" not in f and "back" not in f
                           and inst in f and reg_type in f]
                    # As RMFs can be generated for source and background spectra separately, or one for both,
                    #  we need to check for matching RMFs to the spectrum we found
                    if len(rmf) == 0:
                        rmf = [f for f in named if "rmf" in f and "ann" not in f and "back" not in f
                               and inst in f and "universal" in f]

                    # Exact same checks for the background spectrum
                    back = [f for f in named if "backspec" in f and "ann" not in f and inst in f and reg_type in f]
                    back_arf = [f for f in named if "arf" in f and "ann" not in f and inst in f and reg_type in f
                                and "back" in f]
                    back_rmf = [f for f in named if "rmf" in f and "ann" not in f and "back" in f and inst in f
                                and reg_type in f]
                    if len(back_rmf) == 0:
                        back_rmf = rmf

                    # If exactly one match has been found for all of the products, we define an XGA spectrum and
                    #  add it the source object.
                    if len(arf) == 1 and len(rmf) == 1 and len(back) == 1 and len(back_arf) == 1 and \
                            len(back_rmf) == 1:
                        obj = Spectrum(sp, rmf[0], arf[0], back[0], back_rmf[0], back_arf[0], reg_type, obs, inst,
                                       "", "", "")
                        self.update_products(obj)

                # TODO Add different read in loops for annular spectra and maybe regioned images once I
                #  add them into XGA.

        os.chdir(og_dir)

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
            return np.sqrt(abs(ra - self._ra_dec[0]) ** 2 + abs(dec - self._ra_dec[1]) ** 2)

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
            within = np.array([reg.contains(SkyCoord(*self._ra_dec, unit='deg'), w)
                               for reg in reg_dict[obs_id][diff_sort[0:5]]])

            # Make sure to re-order the region list to match the sorted within array
            reg_dict[obs_id] = reg_dict[obs_id][diff_sort]

            # Expands it so it can be used as a mask on the whole set of regions for this observation
            within = np.pad(within, [0, len(diff_sort) - len(within)])
            match_dict[obs_id] = within
            # Use the deleter for the hdulist to unload the astropy hdulist for this image
            # del self._products[obs_id][inst][en]["image"].hdulist
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

    # TODO Could probably create a master list of regions using intersection, union etc.
    def _source_type_match(self, source_type: str) -> Tuple[Dict, Dict, Dict]:
        """
        A method that looks for matches not just based on position, but also on the type of source
        we want to find. Finding no matches is allowed, but the source will be declared as undetected.
        An error will be thrown if more than one match of the correct type per observation is found.
        :param str source_type: Should either be ext or pnt, describes what type of source I
        should be looking for in the region files.
        :return: A dictionary containing the matched region for each ObsID + a combined region, another
        dictionary containing any sources that matched to the coordinates and weren't chosen,
        and a final dictionary with sources that aren't the target, or in the 2nd dictionary.
        :rtype: Tuple[Dict, Dict, Dict]
        """
        if source_type == "ext":
            allowed_colours = ["green"]
        elif source_type == "pnt":
            allowed_colours = ["red"]
        else:
            raise ValueError("{} is not a recognised source type, please "
                             "don't use this internal function!".format(source_type))

        # TODO Decide about all of the other options that XAPA can spit out - this chunk is from XULL
        # elif reg_colour == "magenta":
        #   reg_type = "ext_psf"
        # elif reg_colour == "blue":
        #   reg_type = "ext_pnt_cont"
        # elif reg_colour == "cyan":
        #   reg_type = "ext_run1_cont"
        # elif reg_colour == "yellow":
        #   reg_type = "ext_less_ten_counts"

        # TODO Comment this method better
        # TODO Maybe remove the combined regions thing, I don't know if its actually useful for anything
        # Here we store the actual matched sources
        results_dict = {}
        # And in this one go all the sources that aren't the matched source, we'll need to subtract them.
        anti_results_dict = {}
        # Sources in this dictionary are within the target source region AND matched to initial coordinates,
        # but aren't the chosen source.
        alt_match_dict = {}
        combined = None
        for obs in self._initial_regions:
            if len(self._initial_regions[obs][self._initial_region_matches[obs]]) == 0:
                results_dict[obs] = None
            else:
                interim_reg = []
                for entry in self._initial_regions[obs][self._initial_region_matches[obs]]:
                    if entry.visual["color"] in allowed_colours:
                        interim_reg.append(entry)

                if len(interim_reg) == 0:
                    results_dict[obs] = None
                elif len(interim_reg) == 1:
                    if combined is None:
                        combined = interim_reg[0]
                    else:
                        combined = combined.union(interim_reg[0])
                    results_dict[obs] = interim_reg[0]
                elif len(interim_reg) > 1:
                    raise MultipleMatchError("More than one match to an extended is found in the region file"
                                             "for observation {}".format(obs))

            alt_match_reg = [entry for entry in self._initial_regions[obs][self._initial_region_matches[obs]]
                             if entry != results_dict[obs]]
            alt_match_dict[obs] = alt_match_reg

            not_source_reg = [reg for reg in self._initial_regions[obs] if reg != results_dict[obs]
                              and reg not in alt_match_reg]
            anti_results_dict[obs] = not_source_reg

        results_dict["combined"] = combined
        return results_dict, alt_match_dict, anti_results_dict

    @property
    def detected(self) -> bool:
        """
        A property getter to return if a match of the correct type has been found.
        :return: The detected boolean attribute.
        :rtype: bool
        """
        if self._detected is None:
            raise ValueError("detected is currently None, BaseSource objects don't have the type "
                             "context needed to define if the source is detected or not.")
        else:
            return self._detected

    def get_source_region(self, reg_type: str, obs_id: str = None) -> Tuple[SkyRegion, SkyRegion]:
        """
        A method to retrieve region objects associated with a source object.
        :param str reg_type: The type of region which we wish to get from the source.
        :param str obs_id: The ObsID that the region is associated with (if appropriate).
        :return: The method returns both the source region and the associated background region.
        :rtype: Tuple[SkyRegion, SkyRegion]
        """
        allowed_rtype = ["r2500", "r500", "r200", "region"]
        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region.")
        elif obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))
        elif reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))
        elif reg_type == "region" and obs_id is None:
            raise ValueError("ObsID cannot be None when getting region file regions.")
        elif reg_type == "region" and obs_id is not None:
            chosen = self._regions[obs_id]
            chosen_back = self._back_regions[obs_id]
        elif reg_type != "region" and not type(self) == GalaxyCluster:
            raise TypeError("Only GalaxyCluster source objects support over-density radii.")
        elif reg_type != "region" and type(self) == GalaxyCluster:
            chosen = self._regions[reg_type]
            chosen_back = self._back_regions[reg_type]
        else:
            raise ValueError("OH NO")

        return chosen, chosen_back

    def get_nuisance_regions(self, obs_id: str) -> Tuple[list, list]:
        """
        This fetches two lists of region objects that describe all the regions that AREN'T the source, and
        regions that also matched to the source coordinates but were not accepted as the source respectively.
        :param obs_id: The ObsID for which you wish to retrieve the nuisance regions.
        :return: A list of non-source regions, and a list of regions that matched to the user coordinates
        but were not accepted as the source.
        :rtype: Tuple[list, list]
        """
        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region, so it cannot know which regions are nuisances.")
        elif obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))

        return self._other_regions[obs_id], self._alt_match_regions[obs_id]

    def get_mask(self, obs_id: str, inst: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        A method to retrieve the mask generated for a particular observation-image combination. The mask
        can be used on an image in pixel coordinates.
        :param obs_id: The ObsID for which you wish to retrieve image masks.
        :param inst: The XMM instrument for which you wish to retrieve image masks.
        :return: Two boolean numpy arrays that can be used as image masks, the first is for the source,
        the second is for the source's background region.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region, so it cannot generate masks.")
        elif obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))
        return self._reg_masks[obs_id][inst], self._back_masks[obs_id][inst]

    def get_sas_region(self, reg_type: str, obs_id: str, inst: str,
                       output_unit: UnitBase = xmm_sky) -> Tuple[str, str]:
        """
        Converts region objects into strings that can be used as part of a SAS command; for instance producing
        a spectrum within one region. This method returns both the source region and associated background
        region with nuisance objects drilled out.
        :param str reg_type: The type of region to generate a SAS region string for.
        :param str obs_id: The ObsID for which we wish to generate the SAS region string.
        :param str inst: The XMM instrument for which we wish to generate the SAS region string.
        :param UnitBase output_unit: The distance unit used by the output SAS region string.
        :return: A SAS region which will include source emission and exclude nuisance sources, and
        another SAS region which will include background emission and exclude nuisance sources.
        :rtype: Tuple[str, str]
        """
        def sas_shape(reg: SkyRegion, im: Image) -> str:
            """
            This will convert the input SkyRegion into an appropriate SAS compatible region string, for use
            with tools such as evselect.
            :param SkyRegion reg: The region object to convert into a SAS region.
            :param Image im: An XGA image object for use in unit conversions.
            :return: A SAS region string describing the input SkyRegion
            :rtype: str
            """
            # This function is just the same process implemented for different region shapes and types
            # I convert the width/height/radius in degrees to the chosen output_unit
            # Then construct a SAS region string and return it.
            if type(reg) == EllipseSkyRegion:
                cen = Quantity([reg.center.ra.value, reg.center.dec.value], 'deg')
                conv_cen = im.coord_conv(cen, output_unit)
                # Have to divide the width by two, I need to know the half-width for SAS regions
                w = Quantity([reg.center.ra.value + (reg.width.value/2), reg.center.dec.value], 'deg')
                conv_w = abs((im.coord_conv(w, output_unit) - conv_cen)[0])
                # Have to divide the height by two, I need to know the half-height for SAS regions
                h = Quantity([reg.center.ra.value, reg.center.dec.value + (reg.height.value/2)], 'deg')
                conv_h = abs((im.coord_conv(h, output_unit) - conv_cen)[1])
                shape_str = "(({t}) IN ellipse({cx},{cy},{w},{h},{rot}))".format(t=c_str, cx=conv_cen[0].value,
                                                                                 cy=conv_cen[1].value,
                                                                                 w=conv_w.value, h=conv_h.value,
                                                                                 rot=reg.angle.value)
            elif type(reg) == CircleSkyRegion:
                cen = Quantity([reg.center.ra.value, reg.center.dec.value], 'deg')
                conv_cen = im.coord_conv(cen, output_unit)
                rad = Quantity([reg.center.ra.value + reg.radius.value, reg.center.dec.value], 'deg')
                conv_rad = abs((im.coord_conv(rad, output_unit) - conv_cen)[0])
                shape_str = "(({t}) IN circle({cx},{cy},{r}))".format(t=c_str, cx=conv_cen[0].value,
                                                                      cy=conv_cen[1].value, r=conv_rad.value)
            elif type(reg) == CompoundSkyRegion and type(reg.region1) == EllipseSkyRegion:
                cen = Quantity([reg.region1.center.ra.value, reg.region1.center.dec.value], 'deg')
                conv_cen = im.coord_conv(cen, output_unit)
                w_i = Quantity([reg.region1.center.ra.value + (reg.region2.width.value / 2),
                                reg.region1.center.dec.value], 'deg')
                conv_w_i = abs((im.coord_conv(w_i, output_unit) - conv_cen)[0])
                w_o = Quantity([reg.region1.center.ra.value + (reg.region1.width.value / 2),
                                reg.region1.center.dec.value], 'deg')
                conv_w_o = abs((im.coord_conv(w_o, output_unit) - conv_cen)[0])

                h_i = Quantity([reg.region1.center.ra.value,
                                reg.region1.center.dec.value + (reg.region2.height.value / 2)], 'deg')
                conv_h_i = abs((im.coord_conv(h_i, output_unit) - conv_cen)[1])
                h_o = Quantity([reg.region1.center.ra.value,
                                reg.region1.center.dec.value + (reg.region1.height.value / 2)], 'deg')
                conv_h_o = abs((im.coord_conv(h_o, output_unit) - conv_cen)[1])

                shape_str = "(({t}) IN elliptannulus({cx},{cy},{wi},{hi},{wo},{ho},{rot},{rot}))"
                shape_str = shape_str.format(t=c_str, cx=conv_cen[0].value, cy=conv_cen[1].value,
                                             wi=conv_w_i.value, hi=conv_h_i.value, wo=conv_w_o.value,
                                             ho=conv_h_o.value, rot=reg.region1.angle.value)
            elif type(reg) == CompoundSkyRegion and type(reg.region1) == CircleSkyRegion:
                cen = Quantity([reg.region1.center.ra.value, reg.region1.center.dec.value], 'deg')
                conv_cen = im.coord_conv(cen, output_unit)
                r_i = Quantity([reg.region1.center.ra.value + reg.region2.radius.value,
                                reg.region1.center.dec.value], 'deg')
                conv_r_i = abs((im.coord_conv(r_i, output_unit) - conv_cen)[0])
                r_o = Quantity([reg.region1.center.ra.value + reg.region1.radius.value,
                                reg.region1.center.dec.value], 'deg')
                conv_r_o = abs((im.coord_conv(r_o, output_unit) - conv_cen)[0])

                shape_str = "(({t}) IN annulus({cx},{cy},{ri},{ro}))"
                shape_str = shape_str.format(t=c_str, cx=conv_cen[0].value, cy=conv_cen[1].value,
                                             ri=conv_r_i.value, ro=conv_r_o.value)
            else:
                shape_str = ""
                raise TypeError("{} is an illegal region type for this method, "
                                "I don't even know how you got here".format(type(reg)))

            return shape_str

        allowed_rtype = ["r2500", "r500", "r200", "region"]
        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region.")
        elif obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))
        elif reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))

        if output_unit == xmm_det:
            c_str = "DETX,DETY"
        elif output_unit == xmm_sky:
            c_str = "X,Y"
        else:
            raise NotImplementedError("Only detector and sky coordinates are currently "
                                      "supported for generating SAS region strings.")

        rel_im = list(self.get_products("image", obs_id, inst))[0][-1]
        source = sas_shape(self._regions[obs_id], rel_im)
        src_interloper = [sas_shape(i, rel_im) for i in self._within_source_regions[obs_id]]
        back = sas_shape(self._back_regions[obs_id], rel_im)
        back_interloper = [sas_shape(i, rel_im) for i in self._within_back_regions[obs_id]]

        if len(src_interloper) == 0:
            final_src = source
        else:
            final_src = source + " &&! " + " &&! ".join(src_interloper)

        if len(back_interloper) == 0:
            final_back = back
        else:
            final_back = back + " &&! " + " &&! ".join(back_interloper)

        return final_src, final_back

    @property
    def nH(self) -> float:
        """
        Property getter for neutral hydrogen column attribute.
        :return: Neutral hydrogen column surface density.
        :rtype: float
        """
        return self._nH

    @property
    def redshift(self):
        """
        Property getter for the redshift of this source object.
        :return: Redshift value
        :rtype: float
        """
        return self._redshift

    @property
    def on_axis_obs_ids(self):
        """
        This method returns an array of ObsIDs that this source is approximately on axis in.
        :return: ObsIDs for which the source is approximately on axis.
        :rtype: np.ndarray
        """
        return self._obs[self._onaxis]

    @property
    def cosmo(self) -> Cosmology:
        """
        This method returns whatever cosmology object is associated with this source object.
        :return: An astropy cosmology object specified for this source on initialization.
        :rtype: Cosmology
        """
        return self._cosmo

    # This is used to name files and directories so this is not allowed to change.
    @property
    def name(self) -> str:
        """
        The name of the source, either given at initialisation or generated from the user-supplied coordinates.
        :return: The name of the source.
        :rtype: str
        """
        return self._name

    def info(self):
        """
        Very simple function that just prints a summary of the BaseSource object.
        """
        print("-----------------------------------------------------")
        print("Source Name - {}".format(self._name))
        print("User Coordinates - ({0}, {1}) degrees".format(*self._ra_dec))
        print("nH - {} cm^-2".format(self.nH))
        print("XMM Observations - {}".format(self.__len__()))
        print("On-Axis - {}".format(self._onaxis.sum()))
        print("With regions - {}".format(len(self._initial_regions)))
        print("Total regions - {}".format(sum([len(self._initial_regions[o]) for o in self._initial_regions])))
        print("Obs with one match - {}".format(sum([1 for o in self._initial_region_matches if
                                                    self._initial_region_matches[o].sum() == 1])))
        print("Obs with >1 matches - {}".format(sum([1 for o in self._initial_region_matches if
                                                     self._initial_region_matches[o].sum() > 1])))
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


# TODO I don't know how all this mask stuff will do with merged products - may have to rejig later
# TODO Don't forget to write another info() method for extended source
class ExtendedSource(BaseSource):
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15):
        super().__init__(ra, dec, redshift, name, cosmology)

        # This uses the added context of the type of source to find (or not find) matches in region files
        self._regions, self._alt_match_regions, self._other_regions = self._source_type_match("ext")

        # TODO Warning about alt_match regions?

        # Here we figure out what other sources are within the chosen extended source region
        self._within_source_regions = {}
        self._back_regions = {}
        self._within_back_regions = {}
        # TODO Can't decide if I need a mask for every instrument, should the coordinates be guaranteed to be
        #  the same by this point?
        self._reg_masks = {obs: {inst: {} for inst in self._products[obs]} for obs in self.obs_ids}
        self._back_masks = {obs: {inst: {} for inst in self._products[obs]} for obs in self.obs_ids}
        # TODO Actually when I implement checking the xga_output directory there could be merged products
        #  at initialisation
        # Iterating through obs_ids rather than _region keys because the _region dictionary will contain
        #  a combined region that cannot be used yet - the user cannot have generated any merged images yet.
        for obs_id in self.obs_ids:
            other_regs = self._other_regions[obs_id]
            im = list(self.get_products("image", obs_id))[0][-1]

            match_reg = self._regions[obs_id]
            m = match_reg.to_pixel(im.radec_wcs)
            crossover = np.array([match_reg.intersection(r).to_pixel(im.radec_wcs).to_mask().data.sum() != 0
                                  for r in other_regs])
            self._within_source_regions[obs_id] = np.array(other_regs)[crossover]

            # Here is where we initialise the background regions, first in pixel coords, then converting
            #  to ra-dec and adding to a dictionary of regions.
            if isinstance(match_reg, EllipseSkyRegion):
                # Here we multiply the inner width/height by 1.05 (to just slightly clear the source region),
                #  and the outer width/height by 1.5 (standard for XCS) - though probably that number should
                #  be dynamic
                # TODO Don't know if the 1.5 multiplier should just be hard-coded, might come back to this later.

                # Ideally this would be an annulus region, but they are bugged in regions v0.4, so we must bodge
                # b_reg = EllipseAnnulusPixelRegion(center=m.center, inner_width=m.width, outer_width=3*m.width,
                #                                   inner_height=m.height, outer_height=3*m.height, angle=m.angle)

                in_reg = EllipsePixelRegion(m.center, m.width*1.05, m.height*1.05, m.angle)
                b_reg = EllipsePixelRegion(m.center, m.width*1.5, m.height*1.5,
                                           m.angle).symmetric_difference(in_reg)
            elif isinstance(match_reg, CircleSkyRegion):
                # b_reg = CircleAnnulusPixelRegion(m.center, m.radius, m.radius*1.5)
                in_reg = CirclePixelRegion(m.center, m.radius * 1.05)
                b_reg = CirclePixelRegion(m.center, m.radius * 1.5).symmetric_difference(in_reg)

            self._back_regions[obs_id] = b_reg.to_sky(im.radec_wcs)
            # This part is dealing with the region in sky coordinates,
            b_reg = self._back_regions[obs_id]
            crossover = np.array([b_reg.intersection(r).to_pixel(im.radec_wcs).to_mask().data.sum() != 0
                                  for r in other_regs])
            self._within_back_regions[obs_id] = np.array(other_regs)[crossover]
            # Ensures we only do regions for instruments that do have at least an events list.
            for inst in self._products[obs_id]:
                self._reg_masks[obs_id][inst], self._back_masks[obs_id][inst] = self._generate_mask(obs_id, inst)

        if all([val is None for val in self._regions.values()]):
            self._detected = False
        else:
            self._detected = True

    def _generate_mask(self, obs_id: str, inst: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        This uses available region files to generate a mask for the source region in the form of a
        numpy array. It takes into account any sources that were detected within the target source,
        by drilling them out.
        :param str obs_id: The XMM ObsID of the image to generate a mask for, this is also allowed to
        be 'combined' when dealing with merged images.
        :param str inst: The XMM instrument of the image to generate a mask for, this is also allowed to
        be 'combined' when dealing with merged images.
        :return: A boolean numpy array that can be used to mask images loaded in as numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # The mask making code has to be here, as if new merged products are generated by the user it will
        #  have to be called again and make use of their WCS.
        inst_im = self.get_products("image", obs_id, inst)[0][-1]
        mask = self._regions[obs_id].to_pixel(inst_im.radec_wcs).to_mask().to_image(inst_im.shape)

        # Now need to drill out any interloping sources, make a mask for that
        interlopers = sum([reg.to_pixel(inst_im.radec_wcs).to_mask().to_image(inst_im.shape)
                           for reg in self._within_source_regions[obs_id]])
        # Wherever the interloper mask is not 0, the global mask must become 0 because there is an
        # interloper source there - circular sentences ftw
        mask[interlopers != 0] = 0

        back_mask = self._back_regions[obs_id].to_pixel(inst_im.radec_wcs).to_mask().to_image(inst_im.shape)
        interlopers = sum([reg.to_pixel(inst_im.radec_wcs).to_mask().to_image(inst_im.shape)
                           for reg in self._within_back_regions[obs_id]])
        # Wherever the interloper mask is not 0, the global mask must become 0 because there is an
        # interloper source there - circular sentences ftw
        back_mask[interlopers != 0] = 0

        return mask, back_mask


class GalaxyCluster(ExtendedSource):
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15):
        super().__init__(ra, dec, redshift, name, cosmology)
        # Don't know if these should be stored as astropy Quantity objects, may add that later
        self._central_coords = {obs: {inst: {} for inst in self._products[obs]} for obs in self.obs_ids}
        # The first coordinate will be the ra and dec that the user input to create the source instance
        # I don't know if this will stay living here, but its as good a start as any
        self._central_coords["user"] = self._ra_dec
        for obs_id in self.obs_ids:
            ra = self._regions[obs_id].center.ra.value
            dec = self._regions[obs_id].center.dec.value
            self._central_coords[obs_id]["region"] = np.array([ra, dec])

        # Once I can actually write this, it will be uncommented and write to _central_coords
        # self._calc_peaks()

        # TODO MOAAAR COMMENTS
        # TODO This should probably have PSF deconvolution applied first
        # TODO I can't really write this properly until I've solved the seg fault question
        # TODO Doc string this POS
        def _calc_peaks(self, lo_en: Quantity, hi_en: Quantity, obs_id: str = None, inst: str = None):
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
                        peak_dump.append(entry)
                    else:
                        # If the current element IS a list, then obviously we still have more unpacking to do,
                        # so we call this function recursively.
                        unpack_list(entry)

            # Just checking the values of the energy limits
            if lo_en > hi_en:
                raise ValueError("lo_en cannot be greater than hi_en")
            else:
                en_id = "bound_{l}-{u}".format(l=lo_en.value, u=hi_en.value)

            matches = []
            for match in dict_search(en_id, self._central_coords):
                peak_dump = []
                unpack_list(match)
                if (obs_id == peak_dump[0] or obs_id is None) and (inst == peak_dump[1] or inst is None):
                    matches.append(peak_dump)

            # If we've already calculated the peaks, then we can just return them now and be done
            if len(matches) != 0:
                return matches

            # Here we fetch the usable images with the energy bounds specified in the call
            # These are dictionaries just because I'm not sure the get_products return will always be in
            # the same order.
            ims = {"".join(im[:-2]): im[-1] for im in self.get_products("image", obs_id, inst)
                   if en_id in im and im[-1].usable}
            # This module shall find peaks in count-rate maps, not straight images, so we need the expmaps as well
            exs = {"".join(em[:-2]): em[-1] for em in self.get_products("expmap", obs_id, inst)
                   if en_id in em and em[-1].usable}

            if len(ims) == 0:
                raise NoProductAvailableError("No usable images available in the {l}{lu}-{u}{uu} band"
                                              "".format(l=lo_en.value, lu=lo_en.unit, u=hi_en.value, uu=hi_en.unit))
            elif len(exs) == 0:
                raise NoProductAvailableError("No usable exposure maps available in the {l}{lu}-{u}{uu} band"
                                              "".format(l=lo_en.value, lu=lo_en.unit, u=hi_en.value, uu=hi_en.unit))
            elif len(ims) > len(exs):
                raise ValueError("Not all images have exposure map counterparts")
            elif len(ims) < len(exs):
                raise ValueError("Not all exposure maps have image counterparts")

            # rate_maps = {ident: np.divide(ims[ident].im_data, exs[ident].im_data,
            #                               out=np.zeros_like(ims[ident].im_data),
            #                               where=exs[ident].im_data != 0) for ident in ims.keys()}

        @property
        def peak(self):
            # This one will return the peak of the merged peak, the user will have to use get_peaks if they wish
            #  for the peaks of the individual data products
            raise NotImplementedError("I'm working on it")
            return


class PointSource(BaseSource):
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15):
        super().__init__(ra, dec, redshift, name, cosmology)
        # This uses the added context of the type of source to find (or not find) matches in region files
        # This is the internal dictionary where all regions, defined by regfiles or by users, will be stored
        self._regions, self._alt_match_regions, self._other_sources = self._source_type_match("pnt")
        if all([val is None for val in self._regions.values()]):
            self._detected = False
        else:
            self._detected = True


