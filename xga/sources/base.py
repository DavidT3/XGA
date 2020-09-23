#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 22/09/2020, 16:20. Copyright (c) David J Turner

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
from fitsio import FITS
from numpy import ndarray
from regions import SkyRegion, EllipseSkyRegion, CircleSkyRegion, EllipsePixelRegion, CirclePixelRegion
from regions import read_ds9, PixelRegion, CompoundSkyRegion

from .. import xga_conf
from ..exceptions import NotAssociatedError, UnknownProductError, NoValidObservationsError, MultipleMatchError, \
    NoProductAvailableError, NoMatchFoundError, ModelNotAssociatedError, ParameterNotAssociatedError
from ..products import PROD_MAP, EventList, BaseProduct, BaseAggregateProduct, Image, Spectrum, ExpMap, \
    RateMap, PSFGrid
from ..sourcetools import simple_xmm_match, nh_lookup, ang_to_rad
from ..utils import ALLOWED_PRODUCTS, XMM_INST, dict_search, xmm_det, xmm_sky, OUTPUT, CENSUS

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class BaseSource:
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15, load_products=True, load_fits=False):
        self._ra_dec = np.array([ra, dec])
        if name is not None:
            self._name = name
        else:
            # self.ra_dec rather than _ra_dec because ra_dec is in astropy degree units
            s = SkyCoord(ra=self.ra_dec[0], dec=self.ra_dec[1])
            crd_str = s.to_string("hmsdms").replace("h", "").replace("m", "").replace("s", "").replace("d", "")
            ra_str, dec_str = crd_str.split(" ")
            # A bug popped up where a conversion ended up with no decimal point and the self._name part got
            #  really upset - so this adds one if there isn't one
            if "." not in ra_str:
                ra_str += "."
            if "." not in dec_str:
                dec_str += "."
            # Use the standard naming convention if one wasn't passed on initialisation of the source
            # Need it because its used for naming files later on.
            self._name = "J" + ra_str[:ra_str.index(".") + 2] + dec_str[:dec_str.index(".") + 2]

        # Only want ObsIDs, not pointing coordinates as well
        # Don't know if I'll always use the simple method
        matches = simple_xmm_match(ra, dec)
        obs = matches["ObsID"].values
        instruments = {o: [] for o in obs}
        for o in obs:
            if matches[matches["ObsID"] == o]["USE_PN"].values[0]:
                instruments[o].append("pn")
            if matches[matches["ObsID"] == o]["USE_MOS1"].values[0]:
                instruments[o].append("mos1")
            if matches[matches["ObsID"] == o]["USE_MOS2"].values[0]:
                instruments[o].append("mos2")

        # This checks that the observations have at least one usable instrument
        self._obs = [o for o in obs if len(instruments[o]) > 0]
        self._instruments = {o: instruments[o] for o in self._obs if len(instruments[o]) > 0}

        # Check in a box of half-side 5 arcminutes, should give an idea of which are on-axis
        try:
            on_axis_match = simple_xmm_match(ra, dec, Quantity(5, 'arcmin'))["ObsID"].values
        except NoMatchFoundError:
            on_axis_match = np.array([])
        self._onaxis = list(np.array(self._obs)[np.isin(self._obs, on_axis_match)])

        # nhlookup returns average and weighted average values, so just take the first
        self._nH = nh_lookup(self.ra_dec)[0]
        self._redshift = redshift
        self._products, region_dict, self._att_files, self._odf_paths = self._initial_products()

        # Want to update the ObsIDs associated with this source after seeing if all files are present
        self._obs = list(self._products.keys())

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
        self._other_regions = None
        self._alt_match_regions = None
        self._interloper_regions = []
        self._interloper_masks = {}

        # Set up an attribute where a default central coordinate will live
        self._default_coord = self.ra_dec

        # Init the the radius multipliers that define the outer and inner edges of a background annulus
        self._back_inn_factor = 1.05
        self._back_out_factor = 1.5

        # Initialisation of fit result attributes
        self._fit_results = {}
        self._test_stat = {}
        self._dof = {}
        self._total_count_rate = {}
        self._total_exp = {}
        self._luminosities = {}

        # Initialisation of attributes related to Extended and GalaxyCluster sources
        self._peaks = None
        # Initialisation of allowed overdensity radii as None
        self._r200 = None
        self._r500 = None
        self._r2500 = None
        # Also adding a radius dictionary attribute
        self._radii = {}
        # Initialisation of cluster observables as None
        self._richness = None
        self._richness_err = None

        self._wl_mass = None
        self._wl_mass_err = None

        # These attributes pertain to the cleaning of observations (as in disassociating them from the source if
        #  they don't include enough of the object we care about).
        self._disassociated = False
        self._disassociated_obs = {}

        # If there is an existing XGA output directory, then it makes sense to search for products that XGA
        #  may have already generated and load them in - saves us wasting time making them again.
        # The user does have control over whether this happens or not though.
        # This goes at the end of init to make sure everything necessary has been declared
        if os.path.exists(OUTPUT) and load_products:
            self._existing_xga_products(load_fits)

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
            # If both an image and an exposure map are present for this energy band, a RateMap object is generated
            if "image" in prod_objs and "expmap" in prod_objs:
                prod_objs["ratemap"] = RateMap(prod_objs["image"], prod_objs["expmap"])
            # Adds in the source name to the products
            for prod in prod_objs:
                prod_objs[prod].obj_name = self._name
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
            if inst not in self._instruments[obs_id]:
                continue
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
                else:
                    reg_dict[obs_id] = None

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
        # Aggregate products are things like PSF grids and sets of annular spectra.
        if not isinstance(prod_obj, (BaseProduct, BaseAggregateProduct)):
            raise TypeError("Only product objects can be assigned to sources.")

        en_bnds = prod_obj.energy_bounds
        if en_bnds[0] is not None and en_bnds[1] is not None:
            extra_key = "bound_{l}-{u}".format(l=float(en_bnds[0].value), u=float(en_bnds[1].value))
            # As the extra_key variable can be altered if the Image is PSF corrected, I'll also make
            #  this variable with just the energy key
            en_key = "bound_{l}-{u}".format(l=float(en_bnds[0].value), u=float(en_bnds[1].value))
        elif type(prod_obj) == Spectrum:
            extra_key = prod_obj.reg_type
        elif type(prod_obj) == PSFGrid:
            # The first part of the key is the model used (by default its ELLBETA for example), and
            #  the second part is the number of bins per side. - Enough to uniquely identify the PSF.
            extra_key = prod_obj.model + "_" + str(prod_obj.num_bins)
        else:
            extra_key = None

        # Secondary checking step now I've added PSF correction
        if type(prod_obj) == Image and prod_obj.psf_corrected:
            extra_key += "_" + prod_obj.psf_model + "_" + str(prod_obj.psf_bins) + "_" + \
                         prod_obj.psf_algorithm + str(prod_obj.psf_iterations)

        # All information about where to place it in our storage hierarchy can be pulled from the product
        # object itself
        obs_id = prod_obj.obs_id
        inst = prod_obj.instrument
        p_type = prod_obj.type

        # Previously, merged images/exposure maps were stored in a separate dictionary, but now everything lives
        #  together - merged products do get a 'combined' prefix on their product type key though
        if obs_id == "combined":
            p_type = "combined_" + p_type

        # 'Combined' will effectively be stored as another ObsID
        if "combined" not in self._products:
            self._products["combined"] = {}

        # The product gets the name of this source object added to it
        prod_obj.obj_name = self.name

        # Double check that something is trying to add products from another source to the current one.
        if obs_id != "combined" and obs_id not in self._products:
            raise NotAssociatedError("{o} is not associated with this X-ray source.".format(o=obs_id))
        elif inst != "combined" and inst not in self._products[obs_id]:
            raise NotAssociatedError("{i} is not associated with XMM observation {o}".format(i=inst, o=obs_id))

        if extra_key is not None and obs_id != "combined":
            # If there is no entry for this 'extra key' (energy band for instance) already, we must make one
            if extra_key not in self._products[obs_id][inst]:
                self._products[obs_id][inst][extra_key] = {}
            self._products[obs_id][inst][extra_key][p_type] = prod_obj
        elif extra_key is None and obs_id != "combined":
            self._products[obs_id][inst][p_type] = prod_obj
        # Here we deal with merged products, they live in the same dictionary, but with no instrument entry
        #  and ObsID = 'combined'
        elif extra_key is not None and obs_id == "combined":
            if extra_key not in self._products[obs_id]:
                self._products[obs_id][extra_key] = {}
            self._products[obs_id][extra_key][p_type] = prod_obj
        elif extra_key is None and obs_id == "combined":
            self._products[obs_id][p_type] = prod_obj

        # This is for an image being added, so we look for a matching exposure map. If it exists we can
        #  make a ratemap
        if p_type == "image":
            # No chance of an expmap being PSF corrected, so we just use the energy key to
            #  look for one that matches our new image
            exs = [prod for prod in self.get_products("expmap", obs_id, inst, just_obj=False) if en_key in prod]
            if len(exs) == 1:
                new_rt = RateMap(prod_obj, exs[0][-1])
                new_rt.obj_name = self.name
                self._products[obs_id][inst][extra_key]["ratemap"] = new_rt

        # However, if its an exposure map that's been added, we have to look for matching image(s). There
        #  could be multiple, because there could be a normal image, and a PSF corrected image
        elif p_type == "expmap":
            # PSF corrected extra keys are built on top of energy keys, so if the en_key is within the extra
            #  key string it counts as a match
            ims = [prod for prod in self.get_products("image", obs_id, inst, just_obj=False)
                   if en_key in prod[-2]]
            # If there is at least one match, we can go to work
            if len(ims) != 0:
                for im in ims:
                    new_rt = RateMap(im[-1], prod_obj)
                    new_rt.obj_name = self.name
                    self._products[obs_id][inst][im[-2]]["ratemap"] = new_rt

        # The same behaviours hold for combined_image and combined_expmap, but they get
        #  stored in slightly different places
        elif p_type == "combined_image":
            exs = [prod for prod in self.get_products("combined_expmap", just_obj=False) if en_key in prod]
            if len(exs) == 1:
                new_rt = RateMap(prod_obj, exs[0][-1])
                new_rt.obj_name = self.name
                # Remember obs_id for combined products is just 'combined'
                self._products[obs_id][extra_key]["combined_ratemap"] = new_rt

        elif p_type == "combined_expmap":
            ims = [prod for prod in self.get_products("combined_image", just_obj=False) if en_key in prod[-2]]
            if len(ims) != 0:
                for im in ims:
                    new_rt = RateMap(im[-1], prod_obj)
                    new_rt.obj_name = self.name
                    self._products[obs_id][im[-2]]["combined_ratemap"] = new_rt

    def _existing_xga_products(self, read_fits: bool):
        """
        A method specifically for searching an existing XGA output directory for relevant files and loading
        them in as XGA products. This will retrieve images, exposure maps, and spectra; then the source product
        structure is updated. The method also finds previous fit results and loads them in.
        :param bool read_fits: Boolean flag that controls whether past fits are read back in or not.
        """

        def parse_image_like(file_path: str, exact_type: str, merged: bool = False) -> BaseProduct:
            """
            Very simple little function that takes the path to an XGA generated image-like product (so either an
            image or an exposure map), parses the file path and makes an XGA object of the correct type by using
            the exact_type variable.
            :param str file_path: Absolute path to an XGA-generated XMM data product.
            :param str exact_type: Either 'image' or 'expmap', the type of product that the file_path leads to.
            :param bool merged: Whether this is a merged file or not.
            :return: An XGA product object.
            :rtype: BaseProduct
            """
            # Get rid of the absolute part of the path, then split by _ to get the information from the file name
            im_info = file_path.split("/")[-1].split("_")

            if not merged:
                # I know its hard coded but this will always be the case, these are files I generate with XGA.
                obs_id = im_info[0]
                ins = im_info[1]
            else:
                ins = "combined"
                obs_id = "combined"

            en_str = [entry for entry in im_info if "keV" in entry][0]
            lo_en, hi_en = en_str.split("keV")[0].split("-")

            # Have to be astropy quantities before passing them into the Product declaration
            lo_en = Quantity(float(lo_en), "keV")
            hi_en = Quantity(float(hi_en), "keV")

            # Different types of Product objects, the empty strings are because I don't have the stdout, stderr,
            #  or original commands for these objects.
            if exact_type == "image" and "psfcorr" not in file_path:
                final_obj = Image(file_path, obs_id, ins, "", "", "", lo_en, hi_en)
            elif exact_type == "image" and "psfcorr" in file_path:
                final_obj = Image(file_path, obs_id, ins, "", "", "", lo_en, hi_en)
                final_obj.psf_corrected = True
                final_obj.psf_bins = int([entry for entry in im_info if "bin" in entry][0].split('bin')[0])
                final_obj.psf_iterations = int([entry for entry in im_info if "iter" in
                                                entry][0].split('iter')[0])
                final_obj.psf_model = [entry for entry in im_info if "mod" in entry][0].split("mod")[0]
                final_obj.psf_algorithm = [entry for entry in im_info if "algo" in entry][0].split("algo")[0]
            elif exact_type == "expmap":
                final_obj = ExpMap(file_path, obs_id, ins, "", "", "", lo_en, hi_en)
            else:
                raise TypeError("Only image and expmap are allowed.")

            return final_obj

        def merged_file_check(file_path: str, obs_ids: Tuple, prod_type: str):
            """
            Checks that a passed file name is a merged image or exposure map, and matches the current source.
            :param str file_path: The name of the file in consideration
            :param Tuple obs_ids: The ObsIDs associated with this source.
            :param str prod_type: img or expmap, what type of merged product are we looking for?
            :return: A boolean flag as to whether the filename is a file that matches the source.
            :rtype: Bool
            """
            # First filter to only look at merged files
            if obs_str in file_path and "merged" in file_path and file_path[0] != "." and prod_type in file_path:
                # Stripped back to only the ObsIDs, and in the original order
                #  Got to strip away quite a few possible entries in the file name - all the PSF information for
                #  instance.
                split_out = [e for e in file_path.split("_") if "keV" not in e and ".fits" not in e and
                             "bin" not in e and "mod" not in e and "algo" not in e and "merged" not in e
                             and "iter" not in e]

                # If the ObsID list from parsing the file name is exactly the same as the ObsID list associated
                #  with this source, then we accept it. Otherwise it is rejected.
                if split_out != obs_ids:
                    right_merged = False
                else:
                    right_merged = True
            else:
                right_merged = False
            return right_merged

        og_dir = os.getcwd()
        for obs in self._obs:
            if os.path.exists(OUTPUT + obs):
                os.chdir(OUTPUT + obs)
                # I've put as many checks as possible in this to make sure it only finds genuine XGA files,
                #  I'll probably put a few more checks later

                # Images read in, pretty simple process - the name of the current source doesn't matter because
                #  standard images/exposure maps are for the WHOLE observation.
                ims = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f) and f[0] != "." and
                       "img" in f and obs in f and (XMM_INST[0] in f or XMM_INST[1] in f or XMM_INST[2] in f)]
                for im in ims:
                    self.update_products(parse_image_like(im, "image"))

                # Exposure maps read in, same process as images
                exs = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f) and f[0] != "." and
                       "expmap" in f and obs in f and (XMM_INST[0] in f or XMM_INST[1] in f or XMM_INST[2] in f)]
                for ex in exs:
                    self.update_products(parse_image_like(ex, "expmap"))

                # For spectra we search for products that have the name of this object in, as they are for
                #  specific parts of the observation.
                # Have to replace any + characters with x, as thats what we did in evselect_spectrum due to SAS
                #  having some issues with the + character in file names
                named = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f) and
                         self._name.replace("+", "x") in f and obs in f
                         and (XMM_INST[0] in f or XMM_INST[1] in f or XMM_INST[2] in f)]
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
        os.chdir(og_dir)

        # Merged products have all the ObsIDs that they are made up of in their name
        obs_str = "_".join(self._obs)
        # They are also always written to the xga_output folder with the name of the first ObsID that goes
        # into them
        if os.path.exists(OUTPUT + self._obs[0]):
            # Follows basically the same process as reading in normal images and exposure maps

            os.chdir(OUTPUT + self._obs[0])
            # Search for files that match the pattern of a merged image/exposure map
            merged_ims = [os.path.abspath(f) for f in os.listdir(".") if merged_file_check(f, self._obs, "img")]
            for im in merged_ims:
                self.update_products(parse_image_like(im, "image", merged=True))

            merged_exs = [os.path.abspath(f) for f in os.listdir(".") if merged_file_check(f, self._obs, "expmap")]
            for ex in merged_exs:
                self.update_products(parse_image_like(ex, "expmap", merged=True))

        # Now loading in previous fits
        if os.path.exists(OUTPUT + "XSPEC/" + self.name) and read_fits:
            prev_fits = [OUTPUT + "XSPEC/" + self.name + "/" + f
                         for f in os.listdir(OUTPUT + "XSPEC/" + self.name) if ".xcm" not in f and ".fits" in f]
            for fit in prev_fits:
                fit_info = fit.split("/")[-1].split("_")
                reg_type = fit_info[1]
                fit_model = fit_info[-1].split(".")[0]
                fit_data = FITS(fit)

                # This bit is largely copied from xspec.py, sorry for my laziness
                global_results = fit_data["RESULTS"][0]
                model = global_results["MODEL"].strip(" ")

                inst_lums = {}
                for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                    sp_info = line["SPEC_PATH"].strip(" ").split("/")[-1].split("_")
                    # Finds the appropriate matching spectrum object for the current table line
                    spec = [match for match in self.get_products("spectrum", sp_info[0], sp_info[1], just_obj=False)
                            if reg_type in match and match[-1].usable][0][-1]

                    # Adds information from this fit to the spectrum object.
                    spec.add_fit_data(str(model), line, fit_data["PLOT" + str(line_ind + 1)])
                    self.update_products(spec)  # Adds the updated spectrum object back into the source

                    # The add_fit_data method formats the luminosities nicely, so we grab them back out
                    #  to help grab the luminosity needed to pass to the source object 'add_fit_data' method
                    processed_lums = spec.get_luminosities(model)
                    if spec.instrument not in inst_lums:
                        inst_lums[spec.instrument] = processed_lums

                    # Ideally the luminosity reported in the source object will be a PN lum, but its not impossible
                    #  that a PN value won't be available. - it shouldn't matter much, lums across the cameras are
                    #  consistent
                if "pn" in inst_lums:
                    chosen_lums = inst_lums["pn"]
                    # mos2 generally better than mos1, as mos1 has CCD damage after a certain point in its life
                elif "mos2" in inst_lums:
                    chosen_lums = inst_lums["mos2"]
                else:
                    chosen_lums = inst_lums["mos1"]

                # Push global fit results, luminosities etc. into the corresponding source object.
                self.add_fit_data(model, reg_type, global_results, chosen_lums)

        os.chdir(og_dir)

    def get_products(self, p_type: str, obs_id: str = None, inst: str = None, extra_key: str = None,
                     just_obj: bool = True) -> List[BaseProduct]:
        """
        This is the getter for the products data structure of Source objects. Passing a 'product type'
        such as 'events' or 'images' will return every matching entry in the products data structure.
        :param str p_type: Product type identifier. e.g. image or expmap.
        :param str obs_id: Optionally, a specific obs_id to search can be supplied.
        :param str inst: Optionally, a specific instrument to search can be supplied.
        :param str extra_key: Optionally, an extra key (like an energy bound) can be supplied.
        :param bool just_obj: A boolean flag that controls whether this method returns just the product objects,
        or the other information that goes with it like ObsID and instrument.
        :return: List of matching products.
        :rtype: List[BaseProduct]
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
            prod_str = ", ".join(ALLOWED_PRODUCTS)
            raise UnknownProductError("{p} is not a recognised product type. Allowed product types are "
                                      "{l}".format(p=p_type, l=prod_str))
        elif obs_id not in self._products and obs_id is not None:
            raise NotAssociatedError("{0} is not associated with {1} .".format(obs_id, self.name))
        elif (obs_id is not None and obs_id in self._products) and \
                (inst is not None and inst not in self._products[obs_id]):
            raise NotAssociatedError("{0} is associated with {1}, but {2} is not associated with that "
                                     "observation".format(obs_id, self.name, inst))

        matches = []
        # Iterates through the dict search return, but each match is likely to be a very nested list,
        # with the degree of nesting dependant on product type (as event lists live a level up from
        # images for instance
        for match in dict_search(p_type, self._products):
            out = []
            unpack_list(match)
            # Only appends if this particular match is for the obs_id and instrument passed to this method
            # Though all matches will be returned if no obs_id/inst is passed
            if (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None) \
                    and (extra_key in out or extra_key is None) and not just_obj:
                matches.append(out)
            elif (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None) \
                    and (extra_key in out or extra_key is None) and just_obj:
                matches.append(out[-1])
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
            if reg_paths[obs_id] is not None:
                ds9_regs = read_ds9(reg_paths[obs_id])
                # Apparently can happen that there are no regions in a region file, so if that is the case
                #  then I just set the ds9_regs to [None] because I know the rest of the code can deal with that.
                #  It can't deal with an empty list
                if len(ds9_regs) == 0:
                    ds9_regs = [None]
            else:
                ds9_regs = [None]

            if isinstance(ds9_regs[0], PixelRegion):
                # If regions exist in pixel coordinates, we need an image WCS to convert them to RA-DEC, so we need
                #  one of the images supplied in the config file, not anything that XGA generates.
                #  But as this method is only run once, before XGA generated products are loaded in, it
                #  should be fine
                inst = [k for k in self._products[obs_id] if k in ["pn", "mos1", "mos2"]][0]
                en = [k for k in self._products[obs_id][inst] if "-" in k][0]
                # Making an assumption here, that if there are regions there will be images
                # Getting the radec_wcs property from the Image object
                im = [i for i in self.get_products("image", obs_id, inst, just_obj=False) if en in i]

                if len(im) != 1:
                    raise NoProductAvailableError("There is no image available for observation {o}, associated "
                                                  "with {n}. An image is require to translate pixel regions "
                                                  "to RA-DEC.".format(o=obs_id, n=self.name))
                w = im[0][-1].radec_wcs
                sky_regs = [reg.to_sky(w) for reg in ds9_regs]
                reg_dict[obs_id] = np.array(sky_regs)
            elif isinstance(ds9_regs[0], SkyRegion):
                reg_dict[obs_id] = np.array(ds9_regs)
            else:
                # So there is an entry in this for EVERY ObsID
                reg_dict[obs_id] = np.array([None])

            # Hopefully this bodge doesn't have any unforeseen consequences
            if reg_dict[obs_id][0] is not None:
                # Quickly calculating distance between source and center of regions, then sorting
                # and getting indices. Thus I only match to the closest 5 regions.
                diff_sort = np.array([dist_from_source(r) for r in reg_dict[obs_id]]).argsort()
                # Unfortunately due to a limitation of the regions module I think you need images
                #  to do this contains match...
                # TODO Come up with an alternative to this that can work without a WCS
                within = np.array([reg.contains(SkyCoord(*self._ra_dec, unit='deg'), w)
                                   for reg in reg_dict[obs_id][diff_sort[0:5]]])

                # Make sure to re-order the region list to match the sorted within array
                reg_dict[obs_id] = reg_dict[obs_id][diff_sort]

                # Expands it so it can be used as a mask on the whole set of regions for this observation
                within = np.pad(within, [0, len(diff_sort) - len(within)])
                match_dict[obs_id] = within
            else:
                match_dict[obs_id] = np.array([False])

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
        # Definitions of the colours of XCS regions can be found in the thesis of Dr Micheal Davidson
        #  University of Edinburgh - 2005.
        if source_type == "ext":
            allowed_colours = ["green", "magenta", "blue", "cyan", "yellow"]
        elif source_type == "pnt":
            allowed_colours = ["red"]
        else:
            raise ValueError("{} is not a recognised source type, please "
                             "don't use this internal function!".format(source_type))

        # Here we store the actual matched sources
        results_dict = {}
        # And in this one go all the sources that aren't the matched source, we'll need to subtract them.
        anti_results_dict = {}
        # Sources in this dictionary are within the target source region AND matched to initial coordinates,
        # but aren't the chosen source.
        alt_match_dict = {}
        # Goes through all the ObsIDs associated with this source, and checks if they have regions
        #  If not then Nones are added to the various dictionaries, otherwise you end up with a list of regions
        #  with missing ObsIDs
        for obs in self.obs_ids:
            if obs in self._initial_regions:
                # If there are no matches then the returned result is just None
                if len(self._initial_regions[obs][self._initial_region_matches[obs]]) == 0:
                    results_dict[obs] = None
                else:
                    interim_reg = []
                    # The only solution I could think of is to go by the XCS standard of region files, so green
                    #  is extended, red is point etc. - not ideal but I'll just explain in the documentation
                    for entry in self._initial_regions[obs][self._initial_region_matches[obs]]:
                        if entry.visual["color"] in allowed_colours:
                            interim_reg.append(entry)

                    # Different matching possibilities
                    if len(interim_reg) == 0:
                        results_dict[obs] = None
                    elif len(interim_reg) == 1:
                        results_dict[obs] = interim_reg[0]
                    # Matching to multiple extended sources would be very problematic, so throw an error
                    elif len(interim_reg) > 1:
                        raise MultipleMatchError("More than one match to an extended is found in the region file"
                                                 "for observation {}".format(obs))

                # Alt match is used for when there is a secondary match to a point source
                alt_match_reg = [entry for entry in self._initial_regions[obs][self._initial_region_matches[obs]]
                                 if entry != results_dict[obs]]
                alt_match_dict[obs] = alt_match_reg

                # These are all the sources that aren't a match, and so should be removed from any analysis
                not_source_reg = [reg for reg in self._initial_regions[obs] if reg != results_dict[obs]
                                  and reg not in alt_match_reg]
                anti_results_dict[obs] = not_source_reg

            else:
                results_dict[obs] = None
                alt_match_dict[obs] = []
                anti_results_dict[obs] = []

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

    def source_back_regions(self, reg_type: str, obs_id: str = None, central_coord: Quantity = None) \
            -> Tuple[SkyRegion, SkyRegion]:
        """
        A method to retrieve source region and background region objects for a given source type with a
        given central coordinate.
        :param str reg_type: The type of region which we wish to get from the source.
        :param str obs_id: The ObsID that the region is associated with (if appropriate).
        :param Quantity central_coord: The central coordinate of the region.
        :return: The method returns both the source region and the associated background region.
        :rtype:
        """
        # Doing an initial check so I can throw a warning if the user wants a region-list region AND has supplied
        #  custom central coordinates
        if reg_type == "region" and central_coord is not None:
            warnings.warn("You cannot use custom central coordinates with a region from supplied region files")
        # elif reg_type != "region" and central_coord is None:
        #     warnings.warn("No central coord supplied, using default (peak if use_peak is True), initial coordinates"
        #                   "otherwise.")

        if central_coord is None:
            central_coord = self._default_coord

        if type(central_coord) == Quantity:
            centre = SkyCoord(*central_coord.to("deg"))
        elif type(central_coord) == SkyCoord:
            centre = central_coord
        else:
            print(central_coord)
            print(type(central_coord))
            raise TypeError("BOI")

        # In case combined gets passed as the ObsID at any point
        if obs_id == "combined":
            obs_id = None

        # The search radius won't be used by the user, just peak finding solutions
        allowed_rtype = ["r2500", "r500", "r200", "region", "custom", "search"]
        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region.")
        elif obs_id is not None and obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))
        elif reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))
        elif reg_type == "region" and obs_id is None:
            raise ValueError("ObsID cannot be None when getting region file regions.")
        elif reg_type == "region" and obs_id is not None:
            src_reg = self._regions[obs_id]
        elif reg_type in ["r2500", "r500", "r200"] and reg_type not in self._radii:
            raise TypeError("There are no over-density radii associated with this source")
        elif reg_type != "region" and reg_type in self._radii:
            # We know for certain that the radius will be in degrees, but it has to be converted to degrees
            #  before being stored in the radii attribute
            radius = self._radii[reg_type]
            src_reg = CircleSkyRegion(centre, radius.to('deg'))
        elif reg_type != "region" and reg_type not in self._radii:
            raise ValueError("{} is a valid region type, but is not associated with this "
                             "source.".format(reg_type))
        else:
            raise ValueError("OH NO")

        # Here is where we initialise the background regions, first in pixel coords, then converting to ra-dec.
        # TODO Verify that just using the first image is okay
        im = self.get_products("image")[0]
        src_pix_reg = src_reg.to_pixel(im.radec_wcs)
        # TODO Try and remember why I had to convert to pixel regions to make it work
        if isinstance(src_reg, EllipseSkyRegion):
            # Here we multiply the inner width/height by 1.05 (to just slightly clear the source region),
            #  and the outer width/height by 1.5 (standard for XCS) - default values
            # Ideally this would be an annulus region, but they are bugged in regions v0.4, so we must bodge
            in_reg = EllipsePixelRegion(src_pix_reg.center, src_pix_reg.width * self._back_inn_factor,
                                        src_pix_reg.height * self._back_inn_factor, src_pix_reg.angle)
            out_reg = EllipsePixelRegion(src_pix_reg.center, src_pix_reg.width * self._back_out_factor,
                                         src_pix_reg.height * self._back_out_factor, src_pix_reg.angle)
            bck_reg = out_reg.symmetric_difference(in_reg)
        elif isinstance(src_reg, CircleSkyRegion):
            in_reg = CirclePixelRegion(src_pix_reg.center, src_pix_reg.radius * self._back_inn_factor)
            out_reg = CirclePixelRegion(src_pix_reg.center, src_pix_reg.radius * self._back_out_factor)
            bck_reg = out_reg.symmetric_difference(in_reg)

        bck_reg = bck_reg.to_sky(im.radec_wcs)

        return src_reg, bck_reg

    def within_region(self, region: SkyRegion) -> List[SkyRegion]:
        """
        This method finds interloper sources that lie within the user supplied region.
        :param SkyRegion region: The region in which we wish to search for interloper sources (for instance
        a source region or background region).
        :return: A list of regions that lie within the user supplied region.
        :rtype: List[SkyRegion]
        """
        im = self.get_products("image")[0]

        crossover = np.array([region.intersection(r).to_pixel(im.radec_wcs).to_mask().data.sum() != 0
                              for r in self._interloper_regions])
        reg_within = np.array(self._interloper_regions)[crossover]

        return reg_within

    def get_source_mask(self, reg_type: str, obs_id: str = None, central_coord: Quantity = None) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to retrieve source and background masks for the given region type.
        :param str reg_type: The type of region for which to retrieve the mask.
        :param str obs_id: The ObsID that the mask is associated with (if appropriate).
        :param Quantity central_coord: The central coordinate of the region.
        :return: The source and background masks for the requested ObsID (or the combined image if no ObsID).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        if obs_id == "combined":
            obs_id = None

        # Don't need to do a bunch of checks, because the method I call to make the
        #  mask does all the checks anyway
        src_reg, bck_reg = self.source_back_regions(reg_type, obs_id, central_coord)
        if central_coord is None:
            central_coord = self._default_coord

        # I assume that if no ObsID is supplied, then the user wishes to have a mask for the combined data
        if obs_id is None:
            mask_image = self.get_products("combined_image")[0]
        else:
            # Just grab the first instrument that comes out the get method, the masks should be the same.
            mask_image = self.get_products("image", obs_id)[0]

        mask = src_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
        back_mask = bck_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)

        return mask, back_mask

    def _generate_interloper_mask(self, mask_image: Image) -> ndarray:
        """
        Internal method that makes interloper masks in the first place; I allow this because interloper
        masks will never change, so can be safely generated and stored in an init of a source class.
        :param Image mask_image: The image for which to create the interloper mask.
        :return: A numpy array of 0s and 1s which acts as a mask to remove interloper sources.
        :rtype: ndarray
        """

        masks = [reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                 for reg in self._interloper_regions if reg is not None]
        interlopers = sum([m for m in masks if m is not None])

        mask = np.ones(mask_image.shape)
        mask[interlopers != 0] = 0

        return mask

    def get_interloper_mask(self, obs_id: str = None) -> ndarray:
        """
        Returns a mask for a given ObsID (or combined data if no ObsID given) that will remove any sources
        that have not been identified as the source of interest.
        :param str obs_id: The ObsID that the mask is associated with (if appropriate).
        :return: A numpy array of 0s and 1s which acts as a mask to remove interloper sources.
        :rtype: ndarray
        """
        if type(self) == BaseSource:
            raise TypeError("BaseSource objects don't have enough information to know which sources "
                            "are interlopers.")

        if obs_id is not None and obs_id != "combined" and obs_id not in self.obs_ids:
            raise NotAssociatedError("{o} is not associated with {s}; only {a} are "
                                     "available".format(o=obs_id, s=self.name, a=", ".join(self.obs_ids)))
        elif obs_id is not None and obs_id != "combined":
            mask = self._interloper_masks[obs_id]
        elif obs_id is None or obs_id == "combined" and "combined" not in self._interloper_masks:
            comb_ims = self.get_products("combined_image")
            if len(comb_ims) == 0:
                raise NoProductAvailableError("There are no combined images available for which to fetch"
                                              " interloper masks.")
            im = comb_ims[0]
            mask = self._generate_interloper_mask(im)
            self._interloper_masks["combined"] = mask
        elif obs_id is None or obs_id == "combined" and "combined" in self._interloper_masks:
            mask = self._interloper_masks["combined"]

        return mask

    def get_mask(self, reg_type: str, obs_id: str = None, central_coord: Quantity = None) -> \
            Tuple[np.ndarray, np.ndarray]:
        """
        Method to retrieve source and background masks for the given region type, WITH INTERLOPERS REMOVED.
        :param str reg_type: The type of region for which to retrieve the interloper corrected mask.
        :param str obs_id: The ObsID that the mask is associated with (if appropriate).
        :param Quantity central_coord: The central coordinate of the region.
        :return: The source and background masks for the requested ObsID (or the combined image if no ObsID).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Grabs the source masks without interlopers removed
        src_mask, bck_mask = self.get_source_mask(reg_type, obs_id, central_coord)
        # Grabs the interloper mask
        interloper_mask = self.get_interloper_mask(obs_id)

        # Multiplies the uncorrected source and background masks with the interloper masks to correct
        #  for interloper sources
        total_src_mask = src_mask * interloper_mask
        total_bck_mask = bck_mask * interloper_mask

        return total_src_mask, total_bck_mask

    def get_snr(self, reg_type: str, central_coord: Quantity = None) -> float:
        """
        This takes a region type and central coordinate and calculates the signal to noise ratio.
        The background region is constructed using the back_inn_rad_factor and back_out_rad_factor
        values, the defaults of which are 1.05*radius and 1.5*radius respectively.
        :param str reg_type: The type of region for which to calculate the signal to noise ratio.
        :param Quantity central_coord: The central coordinate of the region.
        :return: The signal to noise ratio.
        :rtype: float
        """
        # Grabs the interloper corrected source masks
        src_mask, bck_mask = self.get_mask(reg_type, None, central_coord)

        # Finds an appropriate ratemap
        en_key = "bound_{l}-{u}".format(l=self._peak_lo_en.value, u=self._peak_hi_en.value)
        comb_rt = self.get_products("combined_ratemap", extra_key=en_key)[0]

        # Sums the areas of the source and background masks
        src_area = src_mask.sum()
        bck_area = bck_mask.sum()
        # Calculates signal to noise
        ratio = ((comb_rt.data * src_mask).sum() / (comb_rt.data * bck_mask).sum()) * (bck_area / src_area)

        return ratio

    def get_sas_region(self, reg_type: str, obs_id: str, inst: str, output_unit: UnitBase = xmm_sky) \
            -> Tuple[str, str]:
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
                w = Quantity([reg.center.ra.value + (reg.width.value / 2), reg.center.dec.value], 'deg')
                conv_w = abs((im.coord_conv(w, output_unit) - conv_cen)[0])
                # Have to divide the height by two, I need to know the half-height for SAS regions
                h = Quantity([reg.center.ra.value, reg.center.dec.value + (reg.height.value / 2)], 'deg')
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

        allowed_rtype = ["r2500", "r500", "r200", "region", "custom"]

        if output_unit == xmm_det:
            c_str = "DETX,DETY"
        elif output_unit == xmm_sky:
            c_str = "X,Y"
        else:
            raise NotImplementedError("Only detector and sky coordinates are currently "
                                      "supported for generating SAS region strings.")

        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region.")
        elif obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))
        elif reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))
        elif reg_type == "region":
            source, back = self.source_back_regions("region", obs_id)
            source_interlopers = self.within_region(source)
            background_interlopers = self.within_region(back)
        elif reg_type in ["r2500", "r500", "r200"] and reg_type not in self._radii:
            raise TypeError("There are no over-density radii associated with this source")
        elif reg_type != "region" and reg_type in self._radii:
            source, back = self.source_back_regions(reg_type, obs_id)
            source_interlopers = self.within_region(source)
            background_interlopers = self.within_region(back)
        elif reg_type != "region" and reg_type not in self._radii:
            raise ValueError("{} is a valid region type, but is not associated with this "
                             "source.".format(reg_type))
        else:
            raise ValueError("OH NO")

        rel_im = self.get_products("image", obs_id, inst, just_obj=True)[0]
        source = sas_shape(source, rel_im)
        src_interloper = [sas_shape(i, rel_im) for i in source_interlopers]
        back = sas_shape(back, rel_im)
        back_interloper = [sas_shape(i, rel_im) for i in background_interlopers]

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

    # TODO Pass through units in column headers?
    def add_fit_data(self, model: str, reg_type: str, tab_line, lums: dict):
        """
        A method that stores fit results and global information about a the set of spectra in a source object.
        Any variable parameters in the fit are stored in an internal dictionary structure, as are any luminosities
        calculated. Other parameters of interest are store in other internal attributes.
        :param str model:
        :param str reg_type:
        :param tab_line:
        :param dict lums:
        """
        # Just headers that will always be present in tab_line that are not fit parameters
        not_par = ['MODEL', 'TOTAL_EXPOSURE', 'TOTAL_COUNT_RATE', 'TOTAL_COUNT_RATE_ERR',
                   'NUM_UNLINKED_THAWED_VARS', 'FIT_STATISTIC', 'TEST_STATISTIC', 'DOF']

        # Various global values of interest
        self._total_exp[reg_type] = float(tab_line["TOTAL_EXPOSURE"])
        if reg_type not in self._total_count_rate:
            self._total_count_rate[reg_type] = {}
            self._test_stat[reg_type] = {}
            self._dof[reg_type] = {}
        self._total_count_rate[reg_type][model] = [float(tab_line["TOTAL_COUNT_RATE"]),
                                                   float(tab_line["TOTAL_COUNT_RATE_ERR"])]
        self._test_stat[reg_type][model] = float(tab_line["TEST_STATISTIC"])
        self._dof[reg_type][model] = float(tab_line["DOF"])

        # The parameters available will obviously be dynamic, so have to find out what they are and then
        #  then for each result find the +- errors
        par_headers = [n for n in tab_line.dtype.names if n not in not_par]
        mod_res = {}
        for par in par_headers:
            # The parameter name and the parameter index used by XSPEC are separated by |
            par_info = par.split("|")
            par_name = par_info[0]

            # The parameter index can also have an - or + after it if the entry in question is an uncertainty
            if par_info[1][-1] == "-":
                ident = par_info[1][:-1]
                pos = 1
            elif par_info[1][-1] == "+":
                ident = par_info[1][:-1]
                pos = 2
            else:
                ident = par_info[1]
                pos = 0

            # Sets up the dictionary structure for the results
            if par_name not in mod_res:
                mod_res[par_name] = {ident: [0, 0, 0]}
            elif ident not in mod_res[par_name]:
                mod_res[par_name][ident] = [0, 0, 0]

            mod_res[par_name][ident][pos] = float(tab_line[par])

        # Storing the fit results
        if reg_type not in self._fit_results:
            self._fit_results[reg_type] = {}
        self._fit_results[reg_type][model] = mod_res

        # And now storing the luminosity results
        if reg_type not in self._luminosities:
            self._luminosities[reg_type] = {}
        self._luminosities[reg_type][model] = lums

    def get_results(self, reg_type: str, model: str, par: str = None):
        """
        Important method that will retrieve fit results from the source object. Either for a specific
        parameter of a given region-model combination, or for all of them. If a specific parameter is requested,
        all matching values from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+). If no parameter is specified, the return will be a dictionary
        of such numpy arrays, with the keys corresponding to parameter names.
        :param str reg_type: The type of region that the fitted spectra were generated from.
        :param str model: The name of the fitted model that you're requesting the results from (e.g. tbabs*apec).
        :param str par: The name of the parameter you want a result for.
        :return: The requested result value, and uncertainties.
        """
        # Bunch of checks to make sure the requested results actually exist
        if len(self._fit_results) == 0:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this source")
        elif reg_type not in self._fit_results:
            av_regs = ", ".join(self._fit_results.keys())
            raise ModelNotAssociatedError("{0} has no associated XSPEC fit to this source; available regions are "
                                          "{1}".format(reg_type, av_regs))
        elif model not in self._fit_results[reg_type]:
            av_mods = ", ".join(self._fit_results[reg_type].keys())
            raise ModelNotAssociatedError("{0} has not been fitted to {1} spectra of this source; available "
                                          "models are  {2}".format(model, reg_type, av_mods))
        elif par is not None and par not in self._fit_results[reg_type][model]:
            av_pars = ", ".join(self._fit_results[reg_type][model].keys())
            raise ParameterNotAssociatedError("{0} was not a free parameter in the {1} fit to this source, "
                                              "the options are {2}".format(par, model, av_pars))

        # Read out into variable for readabilities sake
        fit_data = self._fit_results[reg_type][model]
        proc_data = {}  # Where the output will ive
        for p_key in fit_data:
            # Used to shape the numpy array the data is transferred into
            num_entries = len(fit_data[p_key])
            # 'Empty' new array to write out the results into, done like this because results are stored
            #  in nested dictionaries with their XSPEC parameter number as an extra key
            new_data = np.zeros((num_entries, 3))

            # If a parameter is unlinked in a fit with multiple spectra (like normalisation for instance),
            #  there can be N entries for the same parameter, writing them out in order to a numpy array
            for incr, par_index in enumerate(fit_data[p_key]):
                new_data[incr, :] = fit_data[p_key][par_index]

            # Just makes the output a little nicer if there is only one entry
            if new_data.shape[0] == 1:
                proc_data[p_key] = new_data[0]
            else:
                proc_data[p_key] = new_data

        # If no specific parameter was requested, the user gets all of them
        if par is None:
            return proc_data
        else:
            return proc_data[par]

    def get_luminosities(self, reg_type: str, model: str, lo_en: Quantity = None, hi_en: Quantity = None):
        """
        Get method for luminosities calculated from model fits to spectra associated with this source.
        Either for given energy limits (that must have been specified when the fit was first performed), or
        for all luminosities associated with that model. Luminosities are returned as a 3 column numpy array;
        the 0th column is the value, the 1st column is the err-, and the 2nd is err+.
        :param str reg_type: The type of region that the fitted spectra were generated from.
        :param str model: The name of the fitted model that you're requesting the
        luminosities from (e.g. tbabs*apec).
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :return: The requested luminosity value, and uncertainties.
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
        elif reg_type not in self._luminosities:
            av_regs = ", ".join(self._luminosities.keys())
            raise ModelNotAssociatedError("{0} has no associated XSPEC fit to this source; available regions are "
                                          "{1}".format(reg_type, av_regs))
        elif model not in self._luminosities[reg_type]:
            av_mods = ", ".join(self._luminosities[reg_type].keys())
            raise ModelNotAssociatedError("{0} has not been fitted to {1} spectra of this source; "
                                          "available models are {2}".format(model, reg_type, av_mods))
        elif en_key is not None and en_key not in self._luminosities[reg_type][model]:
            av_bands = ", ".join([en.split("_")[-1] + "keV" for en in self._luminosities[reg_type][model].keys()])
            raise ParameterNotAssociatedError("{l}-{u}keV was not an energy band for the fit with {m}; available "
                                              "energy bands are {b}".format(l=lo_en.to("keV").value,
                                                                            u=hi_en.to("keV").value,
                                                                            m=model, b=av_bands))

        # If no limits specified,the user gets all the luminosities, otherwise they get the one they asked for
        if en_key is None:
            return self._luminosities[reg_type][model]
        else:
            return self._luminosities[reg_type][model][en_key]

    @property
    def num_pn_obs(self) -> int:
        """
        Getter method that gives the number of PN observations.
        :return: Integer number of PN observations associated with this source
        :rtype: int
        """
        return len([o for o in self.obs_ids if 'pn' in self._products[o]])

    @property
    def num_mos1_obs(self) -> int:
        """
        Getter method that gives the number of MOS1 observations.
        :return: Integer number of MOS1 observations associated with this source
        :rtype: int
        """
        return len([o for o in self.obs_ids if 'mos1' in self._products[o]])

    @property
    def num_mos2_obs(self) -> int:
        """
        Getter method that gives the number of MOS2 observations.
        :return: Integer number of MOS2 observations associated with this source
        :rtype: int
        """
        return len([o for o in self.obs_ids if 'mos2' in self._products[o]])

    # As this is an intrinsic property of which matched observations are valid, there will be no setter
    @property
    def instruments(self) -> Dict:
        """
        A property of a source that details which instruments have valid data for which observations.
        :return: A dictionary of ObsIDs and their associated valid instruments.
        :rtype: Dict
        """
        return self._instruments

    @property
    def disassociated(self) -> bool:
        """
        Property that describes whether this source has had ObsIDs disassociated from it.
        :return: A boolean flag, True means that ObsIDs/instruments have been removed, False means they haven't.
        :rtype: bool
        """
        return self._disassociated

    @property
    def disassociated_obs(self) -> dict:
        """
        Property that details exactly what data has been disassociated from this source, if any.
        :return: Dictionary describing which instruments of which ObsIDs have been disassociated from this source.
        :rtype: dict
        """
        return self._disassociated_obs

    def disassociate_obs(self, to_remove: dict):
        """
        Method that uses the supplied dictionary to safely remove data from the source. This data will no longer
        be used in any analyses, and would typically be removed because it is of poor quality, or doesn't contribute
        enough to justify its presence.
        :param dict to_remove: A dictionary of observations to remove, in the style of the source.instruments
        dictionary, with the top level keys being ObsIDs, and the lower levels being instrument names.
        """
        if not self._disassociated:
            self._disassociated = True
        if len(self._disassociated_obs) == 0:
            self._disassociated_obs = to_remove
        else:
            for o in to_remove:
                if o not in self._disassociated_obs:
                    self._disassociated_obs[o] = to_remove[o]
                else:
                    self._disassociated_obs[o] += to_remove[o]

        # If we're un-associating certain observations, odds on the combined products are no longer valid
        if "combined" in self._products:
            del self._products["combined"]
            if "combined" in self._interloper_masks:
                del self._interloper_masks["combined"]
            self._fit_results = {}
            self._test_stat = {}
            self._dof = {}
            self._total_count_rate = {}
            self._total_exp = {}
            self._luminosities = {}

        for o in to_remove:
            for i in to_remove[o]:
                del self._products[o][i]
                # del self._reg_masks[o][i]
                # del self._back_masks[o][i]
                del self._instruments[o][self._instruments[o].index(i)]

            if len(self._instruments[o]) == 0:
                del self._products[o]
                del self._detected[o]
                del self._initial_regions[o]
                del self._initial_region_matches[o]
                del self._regions[o]
                # del self._back_regions[o]
                del self._other_regions[o]
                del self._alt_match_regions[o]
                # del self._within_source_regions[o]
                # del self._within_back_regions[o]
                del self._peaks[o]

                del self._obs[self._obs.index(o)]
                if o in self._onaxis:
                    del self._onaxis[self._onaxis.index(o)]
                del self._instruments[o]

    def info(self):
        """
        Very simple function that just prints a summary of important information related to the source object..
        """
        print("\n-----------------------------------------------------")
        print("Source Name - {}".format(self._name))
        print("User Coordinates - ({0}, {1}) degrees".format(*self._ra_dec))
        if self._peaks is not None:
            print("X-ray Centroid - ({0}, {1}) degrees".format(*self._peaks["combined"].value))
        print("nH - {}".format(self.nH))
        if self._redshift is not None:
            print("Redshift - {}".format(round(self._redshift, 3)))
        print("XMM ObsIDs - {}".format(self.__len__()))
        print("PN Observations - {}".format(self.num_pn_obs))
        print("MOS1 Observations - {}".format(self.num_mos1_obs))
        print("MOS2 Observations - {}".format(self.num_mos2_obs))
        print("On-Axis - {}".format(len(self._onaxis)))
        print("With regions - {}".format(len(self._initial_regions)))
        print("Total regions - {}".format(sum([len(self._initial_regions[o]) for o in self._initial_regions])))
        print("Obs with one match - {}".format(sum([1 for o in self._initial_region_matches if
                                                    self._initial_region_matches[o].sum() == 1])))
        print("Obs with >1 matches - {}".format(sum([1 for o in self._initial_region_matches if
                                                     self._initial_region_matches[o].sum() > 1])))
        print("Images associated - {}".format(len(self.get_products("image"))))
        print("Exposure maps associated - {}".format(len(self.get_products("expmap"))))
        print("Combined Ratemaps associated - {}".format(len(self.get_products("combined_ratemap"))))
        print("Spectra associated - {}".format(len(self.get_products("spectrum"))))

        if len(self._fit_results) != 0:
            fits = [k + " - " + ", ".join(models) for k, models in self._fit_results.items()]
            print("Available fits - {}".format(" | ".join(fits)))

        if self._regions is not None and "custom" in self._radii:
            if self._redshift is not None:
                region_radius = ang_to_rad(self._custom_region_radius, self._redshift, cosmo=self._cosmo)
            else:
                region_radius = self._custom_region_radius.to("deg")
            print("Custom Region Radius - {}".format(region_radius.round(2)))
            print("Custom Region SNR - {}".format(self.get_snr("custom", self._default_coord).round(2)))

        if self._r200 is not None:
            print("R200 - {}".format(self._r200))
            print("R200 SNR - {}".format(self.get_snr("r200", self._default_coord).round(2)))
        if self._r500 is not None:
            print("R500 - {}".format(self._r500))
            print("R500 SNR - {}".format(self.get_snr("r500", self._default_coord).round(2)))
        if self._r2500 is not None:
            print("R2500 - {}".format(self._r500))
            print("R2500 SNR - {}".format(self.get_snr("r2500", self._default_coord).round(2)))

        # There's probably a neater way of doing the observables - maybe a formatting function?
        if self._richness is not None and self._richness_err is not None \
                and not isinstance(self._richness_err, (list, tuple, ndarray)):
            print("Richness - {0}±{1}".format(self._richness, self._richness_err))
        elif self._richness is not None and self._richness_err is not None \
                and isinstance(self._richness_err, (list, tuple, ndarray)):
            print("Richness - {0} -{1}+{2}".format(self._richness, self._richness_err[0], self._richness_err[1]))
        elif self._richness is not None and self._richness_err is None:
            print("Richness - {0}".format(self._richness))

        if self._wl_mass is not None and self._wl_mass_err is not None \
                and not isinstance(self._wl_mass_err, (list, tuple, ndarray)):
            print("Weak Lensing Mass - {0}±{1}".format(self._wl_mass, self._richness_err))
        elif self._wl_mass is not None and self._wl_mass_err is not None \
                and isinstance(self._wl_mass_err, (list, tuple, ndarray)):
            print("Weak Lensing Mass - {0} -{1}+{2}".format(self._wl_mass, self._wl_mass_err[0],
                                                            self._wl_mass_err[1]))
        elif self._wl_mass is not None and self._wl_mass_err is None:
            print("Weak Lensing Mass - {0}".format(self._wl_mass))

        print("-----------------------------------------------------\n")

    def __len__(self) -> int:
        """
        Method to return the length of the products dictionary (which means the number of
        individual ObsIDs associated with this source), when len() is called on an instance of this class.
        :return: The integer length of the top level of the _products nested dictionary.
        :rtype: int
        """
        return len(self.obs_ids)


# Was going to write this as a subclass of BaseSource, as it will behave largely the same, but I don't
#  want it declaring XGA products for tens of thousands of images etc.
# As such will replicate the base functionality of BaseSource that will allow evselect_image, expmap, cifbuild
# SAS wrappers to work.
# This does have a lot of straight copied code from BaseSource, but I don't mind in this instance
class NullSource:
    def __init__(self, obs: List[str] = None):
        """

        :param obs:
        """
        # To find all census entries with non-na coordinates
        cleaned_census = CENSUS.dropna()
        self._ra_dec = np.array([None, None])
        # The user can specify ObsIDs to associate with the NullSource, or associate all
        #  of them by leaving it as None
        if obs is None:
            self._name = "AllObservations"
            obs = cleaned_census["ObsID"].values
        else:
            # I know this is an ugly nested if statements, but I only wanted to run obs_check once
            obs = np.array(obs)
            obs_check = [o in cleaned_census["ObsID"].values for o in obs]
            # If all user entered ObsIDs are in the census, then all is fine
            if all(obs_check):
                self._name = "{}Observations".format(len(obs))
            # If they aren't all in the census then that is decidedly not fine
            elif not all(obs_check):
                not_valid = np.array(obs)[~np.array(obs_check)]
                raise ValueError("The following are not present in the XGA census, "
                                 "{}".format(", ".join(not_valid)))

        # Find out which
        instruments = {o: [] for o in obs}
        for o in obs:
            if cleaned_census[cleaned_census["ObsID"] == o]["USE_PN"].values[0]:
                instruments[o].append("pn")
            if cleaned_census[cleaned_census["ObsID"] == o]["USE_MOS1"].values[0]:
                instruments[o].append("mos1")
            if cleaned_census[cleaned_census["ObsID"] == o]["USE_MOS2"].values[0]:
                instruments[o].append("mos2")

        # This checks that the observations have at least one usable instrument
        self._obs = [o for o in obs if len(instruments[o]) > 0]
        self._instruments = {o: instruments[o] for o in self._obs if len(instruments[o]) > 0}

        # The SAS generation routine might need this information
        self._att_files = {o: xga_conf["XMM_FILES"]["attitude_file"].format(obs_id=o) for o in self._obs}
        self._odf_paths = {o: xga_conf["XMM_FILES"]["odf_path"].format(obs_id=o) for o in self._obs}

        # Need the event list objects declared unfortunately
        self._products = {o: {} for o in self._obs}
        for o in self._obs:
            for inst in self._instruments[o]:
                evt_key = "clean_{}_evts".format(inst)
                evt_file = xga_conf["XMM_FILES"][evt_key].format(obs_id=o)
                self._products[o][inst] = {"events": EventList(evt_file, obs_id=o, instrument=inst, stdout_str="",
                                                               stderr_str="", gen_cmd="")}

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

    @property
    def instruments(self) -> Dict:
        """
        A property of a source that details which instruments have valid data for which observations.
        :return: A dictionary of ObsIDs and their associated valid instruments.
        :rtype: Dict
        """
        return self._instruments

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

    def update_products(self, prod_obj: BaseProduct):
        """
        This method will not actually store new products in this NullSource. It exists only because my SAS wrappers
        will expect it to, and as such it doesn't do anything at all. This is because NullSource source could have
        tens of thousands of products associated with them, and are only used to bulk generate basic products
        (images, expmaps etc), I don't want the memory overhead of storing them.
        :param BaseProduct prod_obj: The new product object to be added to the source object.
        """
        pass

    def get_products(self, p_type: str, obs_id: str = None, inst: str = None, extra_key: str = None,
                     just_obj: bool = True) -> List[BaseProduct]:
        """
        This is the getter for the products data structure of Source objects. Passing a 'product type'
        such as 'events' or 'images' will return every matching entry in the products data structure.
        :param str p_type: Product type identifier. e.g. image or expmap.
        :param str obs_id: Optionally, a specific obs_id to search can be supplied.
        :param str inst: Optionally, a specific instrument to search can be supplied.
        :param str extra_key: Optionally, an extra key (like an energy bound) can be supplied.
        :param bool just_obj: A boolean flag that controls whether this method returns just the product objects,
        or the other information that goes with it like ObsID and instrument.
        :return: List of matching products.
        :rtype: List[BaseProduct]
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
            prod_str = ", ".join(ALLOWED_PRODUCTS)
            raise UnknownProductError("{p} is not a recognised product type. Allowed product types are "
                                      "{l}".format(p=p_type, l=prod_str))
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
            if (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None) \
                    and (extra_key in out or extra_key is None) and not just_obj:
                matches.append(out)
            elif (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None) \
                    and (extra_key in out or extra_key is None) and just_obj:
                matches.append(out[-1])
        return matches

    # This is used to name files and directories so this is not allowed to change.
    @property
    def name(self) -> str:
        """
        The name of the source, either given at initialisation or generated from the user-supplied coordinates.
        :return: The name of the source.
        :rtype: str
        """
        return self._name

    @property
    def num_pn_obs(self) -> int:
        """
        Getter method that gives the number of PN observations.
        :return: Integer number of PN observations associated with this source
        :rtype: int
        """
        return len([o for o in self.obs_ids if 'pn' in self._products[o]])

    @property
    def num_mos1_obs(self) -> int:
        """
        Getter method that gives the number of MOS1 observations.
        :return: Integer number of MOS1 observations associated with this source
        :rtype: int
        """
        return len([o for o in self.obs_ids if 'mos1' in self._products[o]])

    @property
    def num_mos2_obs(self) -> int:
        """
        Getter method that gives the number of MOS2 observations.
        :return: Integer number of MOS2 observations associated with this source
        :rtype: int
        """
        return len([o for o in self.obs_ids if 'mos2' in self._products[o]])

    def info(self):
        """
        Just prints a couple of pieces of information about the NullSource
        """
        print("\n-----------------------------------------------------")
        print("Source Name - {}".format(self._name))
        print("XMM ObsIDs - {}".format(self.__len__()))
        print("PN Observations - {}".format(self.num_pn_obs))
        print("MOS1 Observations - {}".format(self.num_mos1_obs))
        print("MOS2 Observations - {}".format(self.num_mos2_obs))
        print("-----------------------------------------------------\n")

    def __len__(self) -> int:
        """
        Method to return the length of the products dictionary (which means the number of
        individual ObsIDs associated with this source), when len() is called on an instance of this class.
        :return: The integer length of the top level of the _products nested dictionary.
        :rtype: int
        """
        return len(self.obs_ids)




