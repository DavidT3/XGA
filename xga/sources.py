#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/06/2020, 15:43. Copyright (c) David J Turner
import os
import warnings
from itertools import product
from typing import Tuple, List, Dict

import numpy as np
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck15
from astropy.cosmology.core import Cosmology
from astropy.units import Quantity, UnitBase, deg, UnitConversionError
from fitsio import FITS
from numpy import ndarray
from regions import read_ds9, PixelRegion, SkyRegion, EllipseSkyRegion, CircleSkyRegion, \
    EllipsePixelRegion, CirclePixelRegion, CompoundSkyRegion

from xga import xga_conf
from xga.exceptions import NotAssociatedError, UnknownProductError, NoValidObservationsError, \
    MultipleMatchError, NoProductAvailableError, NoMatchFoundError, ModelNotAssociatedError, \
    ParameterNotAssociatedError, PeakConvergenceFailedError, NoRegionsError
from xga.products import PROD_MAP, EventList, BaseProduct, Image, Spectrum, ExpMap, RateMap
from xga.sourcetools import simple_xmm_match, nhlookup, rad_to_ang, ang_to_rad
from xga.utils import ALLOWED_PRODUCTS, XMM_INST, dict_search, xmm_det, xmm_sky, OUTPUT

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
warnings.simplefilter('ignore', wcs.FITSFixedWarning)


class BaseSource:
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15, load_products=False, load_fits=False):
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
        self._back_regions = None
        self._other_regions = None
        self._alt_match_regions = None
        self._reg_masks = None
        self._back_masks = None
        self._within_source_regions = None
        self._within_back_regions = None

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
        # Initialisation of cluster observables as None
        self._richness = None
        self._richness_err = None

        self._wl_mass = None
        self._wl_mass_err = None

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

        # Finally, we do a quick check for matching pairs of images and exposure maps, because if they
        #  exist then we can generate a RateMap product object.
        if p_type == "image" or p_type == "expmap":
            # Check for existing images, exposure maps, and rate maps that match the product that has just
            #  been added (if that product is an image or exposure map).
            ims = [prod for prod in self.get_products("image", obs_id, inst, just_obj=False) if extra_key in prod]
            exs = [prod for prod in self.get_products("expmap", obs_id, inst, just_obj=False) if extra_key in prod]
            rts = [prod for prod in self.get_products("ratemap", obs_id, inst, just_obj=False) if extra_key in prod]
            # If we find that there is one match each for image and exposure map,
            #  and no ratemap, then we make one
            if len(ims) == 1 and len(exs) == 1 and ims[0][-1].usable and exs[0][-1].usable and len(rts) == 0:
                new_rt = RateMap(ims[0][-1], exs[0][-1])
                new_rt.obj_name = self.name
                self._products[obs_id][inst][extra_key]["ratemap"] = new_rt

        # The combined images and exposure maps do much the same thing but they're in a separate part
        #  of the if statement because they get named and stored in slightly different ways
        elif p_type == "combined_image" or p_type == "combined_expmap":
            ims = [prod for prod in self.get_products("combined_image", just_obj=False) if extra_key in prod]
            exs = [prod for prod in self.get_products("combined_expmap", just_obj=False) if extra_key in prod]
            rts = [prod for prod in self.get_products("combined_ratemap", just_obj=False) if extra_key in prod]
            if len(ims) == 1 and len(exs) == 1 and ims[0][-1].usable and exs[0][-1].usable and len(rts) == 0:
                new_rt = RateMap(ims[0][-1], exs[0][-1])
                new_rt.obj_name = self.name
                self._products[obs_id][extra_key]["combined_ratemap"] = new_rt

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
                ins = im_info[1]
                obs_id = im_info[0]
                en_str = im_info[-1]
            else:
                ins = "combined"
                obs_id = "combined"
                en_str = im_info[-2]

            lo_en, hi_en = en_str.split("keV")[0].split("-")
            # Have to be astropy quantities before passing them into the Product declaration
            lo_en = Quantity(float(lo_en), "keV")
            hi_en = Quantity(float(hi_en), "keV")

            # Different types of Product objects, the empty strings are because I don't have the stdout, stderr,
            #  or original commands for these objects.
            if exact_type == "image":
                final_obj = Image(file_path, obs_id, ins, "", "", "", lo_en, hi_en)
            elif exact_type == "expmap":
                final_obj = ExpMap(file_path, obs_id, ins, "", "", "", lo_en, hi_en)
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
            # TODO Make this an exact match to the obs_str, otherwise its possible we might read
            #  in an old merged image if new observations are added to the obs census
            merged_ims = [os.path.abspath(f) for f in os.listdir(".") if obs_str in f and "merged_image" in f
                          and f[0] != "."]
            for im in merged_ims:
                self.update_products(parse_image_like(im, "image", merged=True))

            merged_exs = [os.path.abspath(f) for f in os.listdir(".") if obs_str in f and "merged_expmap" in f
                          and f[0] != "."]
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
                    and (extra_key == out[2] or extra_key is None) and not just_obj:
                matches.append(out)
            elif (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None) \
                    and (extra_key == out[2] or extra_key is None) and just_obj:
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
            ds9_regs = read_ds9(reg_paths[obs_id])
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
                    raise NoProductAvailableError("There is no image available to translate pixel regions "
                                                  "to RA-DEC.")
                w = im[0][-1].radec_wcs
                sky_regs = [reg.to_sky(w) for reg in ds9_regs]
                reg_dict[obs_id] = np.array(sky_regs)
            else:
                reg_dict[obs_id] = np.array(ds9_regs)

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

    def get_source_region(self, reg_type: str, obs_id: str = None) -> Tuple[SkyRegion, SkyRegion]:
        """
        A method to retrieve region objects associated with a source object.
        :param str reg_type: The type of region which we wish to get from the source.
        :param str obs_id: The ObsID that the region is associated with (if appropriate).
        :return: The method returns both the source region and the associated background region.
        :rtype: Tuple[SkyRegion, SkyRegion]
        """
        allowed_rtype = ["r2500", "r500", "r200", "region", "custom"]
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
            chosen = self._regions[obs_id]
            chosen_back = self._back_regions[obs_id]
        elif reg_type in ["r2500", "r500", "r200"] and not type(self) == GalaxyCluster:
            raise TypeError("Only GalaxyCluster source objects support over-density radii.")
        elif reg_type != "region" and reg_type in self._regions:
            chosen = self._regions[reg_type]
            chosen_back = self._back_regions[reg_type]
        elif reg_type != "region" and reg_type not in self._regions:
            raise ValueError("{} is a valid region type, but is not associated with this "
                             "source.".format(reg_type))
        else:
            raise ValueError("OH NO")

        return chosen, chosen_back

    def get_nuisance_regions(self, obs_id: str = None) -> Tuple[list, list]:
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

    def get_mask(self, reg_type: str, obs_id: str = None, inst: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        A method to retrieve the mask generated for a particular observation-image combination. The mask
        can be used on an image in pixel coordinates.
        :param str reg_type: The type of region which we wish to get from the source.
        :param obs_id: The ObsID for which you wish to retrieve image masks.
        :param inst: The XMM instrument for which you wish to retrieve image masks.
        :return: Two boolean numpy arrays that can be used as image masks, the first is for the source,
        the second is for the source's background region.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        allowed_rtype = ["r2500", "r500", "r200", "region", "custom"]
        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region, so it cannot generate masks.")
        elif obs_id is not None and obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {} is not associated with this source.".format(obs_id))
        elif obs_id is not None and inst is not None and inst not in self._reg_masks[obs_id]:
            raise NotAssociatedError("The instrument {i} is not associated with observation {o} of this "
                                     "source.".format(i=inst, o=obs_id))
        elif reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))
        elif reg_type == "region" and obs_id is None:
            raise ValueError("ObsID cannot be None when getting region file regions.")
        elif reg_type == "region" and obs_id is not None and inst is not None:
            chosen = self._reg_masks[obs_id][inst]
            chosen_back = self._back_masks[obs_id][inst]
        elif reg_type == "region" and obs_id is not None and inst is None:
            raise ValueError("Inst cannot be None when getting region file regions.")
        elif reg_type in ["r2500", "r500", "r200"] and not type(self) == GalaxyCluster:
            raise TypeError("Only GalaxyCluster source objects support over-density radii.")
        elif reg_type != "region" and reg_type in self._reg_masks:
            chosen = self._reg_masks[reg_type]
            chosen_back = self._back_masks[reg_type]
        elif reg_type != "region" and reg_type not in self._reg_masks:
            raise ValueError("{} is a valid region type, but is not associated with this "
                             "source.".format(reg_type))
        else:
            raise ValueError("OH NO")

        return chosen, chosen_back

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
            source = self._regions[obs_id]
            source_interlopers = self._within_source_regions[obs_id]
            back = self._back_regions[obs_id]
            background_interlopers = self._within_back_regions[obs_id]
        elif reg_type in ["r2500", "r500", "r200"] and not type(self) == GalaxyCluster:
            raise TypeError("Only GalaxyCluster source objects support over-density radii.")
        elif reg_type != "region" and reg_type in self._regions:
            source = self._regions[reg_type]
            source_interlopers = self._within_source_regions[reg_type]
            back = self._back_regions[reg_type]
            background_interlopers = self._within_back_regions[reg_type]
        elif reg_type != "region" and reg_type not in self._regions:
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
            av_bands = ", ".join([en.split("_")[-1]+"keV" for en in self._luminosities[reg_type][model].keys()])
            raise ParameterNotAssociatedError("{l}-{u}keV was not an energy band for the fit with {m}; available "
                                              "energy bands are {b}".format(l=lo_en.to("keV").value,
                                                                            u=hi_en.to("keV").value,
                                                                            m=model, b=av_bands))

        # If no limits specified,the user gets all the luminosities, otherwise they get the one they asked for
        if en_key is None:
            return self._luminosities[reg_type][model]
        else:
            return self._luminosities[reg_type][model][en_key]

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
        print("XMM Observations - {}".format(self.__len__()))
        print("On-Axis - {}".format(self._onaxis.sum()))
        print("With regions - {}".format(len(self._initial_regions)))
        print("Total regions - {}".format(sum([len(self._initial_regions[o]) for o in self._initial_regions])))
        print("Obs with one match - {}".format(sum([1 for o in self._initial_region_matches if
                                                    self._initial_region_matches[o].sum() == 1])))
        print("Obs with >1 matches - {}".format(sum([1 for o in self._initial_region_matches if
                                                     self._initial_region_matches[o].sum() > 1])))
        print("Images associated - {}".format(len(self.get_products("image"))))
        print("Exposure maps associated - {}".format(len(self.get_products("expmap"))))
        print("Spectra associated - {}".format(len(self.get_products("spectrum"))))

        if len(self._fit_results) != 0:
            fits = [k + " - " + ", ".join(models) for k, models in self._fit_results.items()]
            print("Available fits - {}".format(" | ".join(fits)))

        if self._regions is not None and "custom" in self._regions:
            if self._custom_region_radius.unit.is_equivalent('deg'):
                region_radius = ang_to_rad(self._custom_region_radius, self._redshift, cosmo=self._cosmo)
            elif self._custom_region_radius.unit.is_equivalent('kpc'):
                region_radius = self._custom_region_radius.to("kpc")
            print("Custom Region Radius - {}".format(region_radius.round(2)))
            print("Custom Region SNR - {}".format(self._snr["custom"].round(2)))

        if self._r200 is not None:
            print("R200 - {}".format(self._r200))
            print("R200 SNR - {}".format(round(self._snr["r200"], 2)))
        if self._r500 is not None:
            print("R500 - {}".format(self._r500))
            print("R500 SNR - {}".format(round(self._snr["r500"], 2)))
        if self._r2500 is not None:
            print("R2500 - {}".format(self._r500))
            print("R2500 SNR - {}".format(round(self._snr["r2500"], 2)))

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


class ExtendedSource(BaseSource):
    # TODO Make a view method for this class that plots the measured peaks on the combined ratemap.
    def __init__(self, ra, dec, redshift=None, name=None, custom_region_radius=None, use_peak=True,
                 peak_lo_en=Quantity(0.5, "keV"), peak_hi_en=Quantity(2.0, "keV"),
                 back_inn_rad_factor=1.05, back_out_rad_factor=1.5, cosmology=Planck15,
                 load_products=False, load_fits=False):
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
                    in_reg = EllipsePixelRegion(m.center, m.width*self._back_inn_factor,
                                                m.height*self._back_inn_factor, m.angle)
                    b_reg = EllipsePixelRegion(m.center, m.width*self._back_out_factor,
                                               m.height*self._back_out_factor,
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
            #  and converts anyway, but I want the internal unit of the custom radii to be kpc.
            if self._custom_region_radius.unit.is_equivalent("deg"):
                rad = ang_to_rad(self._custom_region_radius, self._redshift, self._cosmo).to("kpc")
                self._custom_region_radius = rad
            else:
                self._custom_region_radius = self._custom_region_radius.to("kpc")

    # TODO There really has to be a better solution to the all_interlopers thing, its so inefficient
    def _generate_mask(self, mask_image: Image, source_region: SkyRegion, back_reg: SkyRegion = None,
                       all_interlopers: bool = False) -> Tuple[ndarray, ndarray]:
        """
        This uses available region files to generate a mask for the source region in the form of a
        numpy array. It takes into account any sources that were detected within the target source,
        by drilling them out.
        :param Image mask_image: An XGA image object that donates its WCS to convert SkyRegions to pixels.
        :param SkyRegion source_region: The SkyRegion containing the source to generate a mask for.
        :param SkyRegion back_reg: The SkyRegion containing the background emission to
        generate a mask for.
        :param bool all_interlopers: If this is true, all non source objects from all observations
        will be iterated through and removed from the mask - its not very efficient...
        :return: A boolean numpy array that can be used to mask images loaded in as numpy arrays.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        obs_id = mask_image.obs_id
        mask = source_region.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)

        if not all_interlopers:
            # Now need to drill out any interloping sources, make a mask for that
            interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                               for reg in self._within_source_regions[obs_id]])
        else:
            all_within = []
            for o in self._within_source_regions:
                all_within += list(self._within_source_regions[o])
            for o in self._other_regions:
                all_within += list(self._other_regions[o])

            interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                               for reg in all_within])
        # Wherever the interloper mask is not 0, the global mask must become 0 because there is an
        # interloper source there - circular sentences ftw
        mask[interlopers != 0] = 0

        if back_reg is not None:
            back_mask = back_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
            if not all_interlopers:
                # Now need to drill out any interloping sources, make a mask for that
                interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                                   for reg in self._within_back_regions[obs_id]])
            else:
                all_within = []
                for o in self._within_back_regions:
                    all_within += list(self._within_back_regions[o])
                for o in self._other_regions:
                    all_within += list(self._other_regions[o])
                interlopers = sum([reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
                                   for reg in all_within])

            # Wherever the interloper mask is not 0, the global mask must become 0 because there is an
            # interloper source there - circular sentences ftw
            back_mask[interlopers != 0] = 0

            return mask, back_mask
        return mask

    def _find_peak(self, rt: RateMap) -> Tuple[Quantity, bool, bool]:
        """
        An internal method that will find the X-ray centroid for the RateMap that has been passed in. It takes
        the user supplied coordinates from source initialisation as a starting point, finds the peak within a 500kpc
        radius, re-centres the region, and iterates until the centroid converges to within 15kpc, or until 20
        20 iterations has been reached.
        :param rt: The ratemap which we want to find the peak (local to our user supplied coordinates) of.
        :return: The peak coordinate, a boolean flag as to whether the returned coordinates are near
         a chip gap/edge, and a boolean flag as to whether the peak converged.
        :rtype: Tuple[Quantity, bool, bool]
        """
        central_coords = SkyCoord(*self.ra_dec.to("deg"))

        # 500kpc in degrees, for the current redshift and cosmology
        search_aperture = rad_to_ang(Quantity(500, "kpc"), self._redshift, cosmo=self._cosmo)
        # Set an absurdly high initial separation, to make sure it does an initial iteration
        separation = Quantity(10000, "kpc")
        # Iteration counter just to kill it if it doesn't converge
        count = 0

        # Allow 20 iterations before we kill this - alternatively loop will exit when centre converges
        #  to within 15kpc
        while count < 20 and separation > Quantity(15, "kpc"):
            # Define a 500kpc radius region centered on the current central_coords
            cust_reg = CircleSkyRegion(central_coords, search_aperture)
            # Generate the source mask for the peak finding method
            aperture_mask = self._generate_mask(rt, cust_reg, all_interlopers=True)
            # Find the peak using the experimental clustering_peak method
            peak, near_edge = rt.clustering_peak(aperture_mask, deg)
            # Calculate the distance between new peak and old central coordinates
            separation = Quantity(np.sqrt(abs(peak[0].value - central_coords.ra.value) ** 2 +
                                          abs(peak[1].value - central_coords.dec.value) ** 2), deg)
            separation = ang_to_rad(separation, self._redshift, self._cosmo)
            central_coords = SkyCoord(*peak.copy())
            count += 1

            if count == 20 and separation > Quantity(15, "kpc"):
                converged = False
                # To do the least amount of damage, if the peak doesn't converge then we just return the
                #  user supplied coordinates
                peak = self.ra_dec
                near_edge = rt.near_edge(peak)
            else:
                converged = True

        return peak, near_edge, converged

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

        coord, near_edge, converged = self._find_peak(comb_rt)
        # Unfortunately if the peak convergence fails for the combined ratemap I have to raise an error
        if converged:
            self._peaks["combined"] = coord
            self._peaks_near_edge["combined"] = near_edge
        else:
            raise PeakConvergenceFailedError("Peak finding on the combined ratemap failed to converge within "
                                             "15kpc for {n} in the {l}-{u} energy "
                                             "band.".format(n=self.name, l=self._peak_lo_en, u=self._peak_hi_en))

        for obs in self.obs_ids:
            for rt in self.get_products("ratemap", obs_id=obs, extra_key=en_key, just_obj=True):
                coord, near_edge, converged = self._find_peak(rt)
                if converged:
                    self._peaks[obs][rt.instrument] = coord
                    self._peaks_near_edge[obs][rt.instrument] = near_edge
                else:
                    self._peaks[obs][rt.instrument] = None
                    self._peaks_near_edge[obs][rt.instrument] = None

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

        # Make the final masks for source and background regions.
        src_mask, bck_mask = self._generate_mask(comb_rt, cust_reg, cust_back_reg, all_interlopers=True)

        # Setting up useful lists for adding regions to
        reg_crossover = []
        bck_crossover = []
        # I check through all available region lists to find regions that are within the custom region
        for obs_id in self._other_regions:
            other_regs = self._other_regions[obs_id]
            # Which regions are within the custom source region
            cross = np.array([cust_reg.intersection(r).to_pixel(comb_rt.radec_wcs).to_mask().data.sum()
                              != 0 for r in other_regs])
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

        src_area = src_mask.sum()
        bck_area = bck_mask.sum()
        rate_ratio = ((comb_rt.data*src_mask).sum() / (comb_rt.data*bck_mask).sum()) * (bck_area / src_area)

        self._regions[reg_type] = cust_reg
        self._back_regions[reg_type] = cust_back_reg
        self._reg_masks[reg_type] = src_mask
        self._back_masks[reg_type] = bck_mask
        self._within_source_regions[reg_type] = reg_crossover
        self._within_back_regions[reg_type] = bck_crossover
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


class GalaxyCluster(ExtendedSource):
    def __init__(self, ra, dec, redshift, r200: Quantity = None, r500: Quantity = None, r2500: Quantity = None,
                 richness: float = None, richness_err: float = None, wl_mass: Quantity = None,
                 wl_mass_err: Quantity = None, name=None, custom_region_radius=None, use_peak=True,
                 peak_lo_en=Quantity(0.5, "keV"), peak_hi_en=Quantity(2.0, "keV"), back_inn_rad_factor=1.05,
                 back_out_rad_factor=1.5, cosmology=Planck15, load_products=False, load_fits=False):
        super().__init__(ra, dec, redshift, name, custom_region_radius, use_peak, peak_lo_en, peak_hi_en,
                         back_inn_rad_factor, back_out_rad_factor, cosmology, load_products, load_fits)

        if r200 is None and r500 is None and r2500 is None:
            raise ValueError("You must set at least one overdensity radius")

        # Here we don't need to check if a non-null redshift was supplied, a redshift is required for
        #  initialising a GalaxyCluster object. These chunks just convert the radii to kpc.
        # I know its ugly to have the same code three times, but I want these to be in attributes.
        if r200 is not None and r200.unit.is_equivalent("deg"):
            self._r200 = ang_to_rad(r200, self._redshift, self._cosmo).to("kpc")
        elif r200 is not None and r200.unit.is_equivalent("kpc"):
            self._r200 = r200.to("kpc")
        elif r200 is not None and not r200.unit.is_equivalent("kpc") and not r200.unit.is_equivalent("deg"):
            raise UnitConversionError("R200 radius must be in either angular or distance units.")

        if r500 is not None and r500.unit.is_equivalent("deg"):
            self._r500 = ang_to_rad(r500, self._redshift, self._cosmo).to("kpc")
        elif r500 is not None and r500.unit.is_equivalent("kpc"):
            self._r500 = r500.to("kpc")
        elif r500 is not None and not r500.unit.is_equivalent("kpc") and not r500.unit.is_equivalent("deg"):
            raise UnitConversionError("R500 radius must be in either angular or distance units.")

        if r2500 is not None and r2500.unit.is_equivalent("deg"):
            self._r2500 = ang_to_rad(r2500, self._redshift, self._cosmo).to("kpc")
        elif r2500 is not None and r2500.unit.is_equivalent("kpc"):
            self._r2500 = r2500.to("kpc")
        elif r2500 is not None and not r2500.unit.is_equivalent("kpc") and not r2500.unit.is_equivalent("deg"):
            raise UnitConversionError("R2500 radius must be in either angular or distance units.")

        if r200 is not None:
            self._setup_new_region(self._r200, "r200")
        if r500 is not None:
            self._setup_new_region(self._r500, "r500")
        if r2500 is not None:
            self._setup_new_region(self._r2500, "r2500")

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

    # Property getters for the overdensity radii, they don't get setters as various things are defined on init
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


class PointSource(BaseSource):
    def __init__(self, ra, dec, redshift=None, name=None, cosmology=Planck15, load_products=False, load_fits=False):
        super().__init__(ra, dec, redshift, name, cosmology, load_products, load_fits)
        # This uses the added context of the type of source to find (or not find) matches in region files
        # This is the internal dictionary where all regions, defined by regfiles or by users, will be stored
        self._regions, self._alt_match_regions, self._other_sources = self._source_type_match("pnt")



