#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 06/06/2025, 11:29. Copyright (c) The Contributors

import os
import pickle
from copy import deepcopy
from itertools import product
from shutil import copyfile
from typing import Tuple, List, Dict, Union
from warnings import warn, simplefilter

import numpy as np
import pandas as pd
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.cosmology.core import Cosmology
from astropy.units import Quantity, UnitBase, Unit, UnitConversionError, deg
from fitsio import FITS
from numpy import ndarray
from regions import (SkyRegion, EllipseSkyRegion, CircleSkyRegion, EllipsePixelRegion, CirclePixelRegion, PixelRegion,
                     Regions)

from .. import xga_conf, BLACKLIST
from ..exceptions import NotAssociatedError, NoValidObservationsError, MultipleMatchError, \
    NoProductAvailableError, NoMatchFoundError, ModelNotAssociatedError, ParameterNotAssociatedError, \
    NotSampleMemberError, XGADeveloperError, FitConfNotAssociatedError
from ..imagetools.misc import pix_deg_scale
from ..imagetools.misc import sky_deg_scale
from ..imagetools.profile import annular_mask
from ..products import PROD_MAP, EventList, BaseProduct, BaseAggregateProduct, Image, Spectrum, ExpMap, \
    RateMap, PSFGrid, BaseProfile1D, AnnularSpectra
from ..products.lightcurve import LightCurve, AggregateLightCurve
from ..sourcetools import simple_xmm_match, nh_lookup, ang_to_rad, rad_to_ang
from ..sourcetools.misc import coord_to_name
from ..utils import ALLOWED_PRODUCTS, XMM_INST, dict_search, xmm_det, xmm_sky, OUTPUT, CENSUS, SRC_REGION_COLOURS, \
    DEFAULT_COSMO

# This disables an annoying astropy warning that pops up all the time with XMM images
# Don't know if I should do this really
simplefilter('ignore', wcs.FITSFixedWarning)


class BaseSource:
    """
    The overlord of all XGA classes, the superclass for all source classes. This contains a huge amount of
    functionality upon which the rest of XGA is built, includes selecting observations, reading in data products,
    and storing newly created data products. Base functionality is included, but this type of source shouldn't
    often need to be instantiated by a user.

    :param float ra: The right ascension (in degrees) of the source.
    :param float dec: The declination (in degrees) of the source.
    :param float redshift: The redshift of the source, default is None. Not supplying a redshift means that
        proper distance units such as kpc cannot be used.
    :param str name: The name of the source, default is None in which case a name will be assembled from the
        coordinates given.
    :param Cosmology cosmology: An astropy cosmology object to use for analysis of this source, default is a
        concordance flat LambdaCDM model.
    :param bool load_products: Should existing XGA generated products for this source be loaded in, default
        is True.
    :param bool load_fits: Should existing XSPEC fits for this source be loaded in, will only work if
        load_products is True. Default is False.
    :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or not, setting
        to True suppresses some warnings so that they can be displayed at the end of the sample progress bar. Default
        is False. User should only set to True to remove warnings.
    """
    def __init__(self, ra: float, dec: float, redshift: float = None, name: str = None,
                 cosmology: Cosmology = DEFAULT_COSMO, load_products: bool = True, load_fits: bool = False,
                 in_sample: bool = False):
        """
        The init method for the BaseSource, the most general type of XGA source which acts as a superclass for all
        others.

        :param float ra: The right ascension (in degrees) of the source.
        :param float dec: The declination (in degrees) of the source.
        :param float redshift: The redshift of the source, default is None. Not supplying a redshift means that
            proper distance units such as kpc cannot be used.
        :param str name: The name of the source, default is None in which case a name will be assembled from the
            coordinates given.
        :param Cosmology cosmology: An astropy cosmology object to use for analysis of this source, default is a
            concordance flat LambdaCDM model.
        :param bool load_products: Should existing XGA generated products for this source be loaded in, default
            is True.
        :param bool load_fits: Should existing XSPEC fits for this source be loaded in, will only work if
            load_products is True. Default is False.
        :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or
            not, setting to True suppresses some warnings so that they can be displayed at the end of the sample
            progress bar. Default is False. User should only set to True to remove warnings.
        """
        # This tells the source that it is a part of a sample, which we will check to see whether to suppress warnings
        self._samp_member = in_sample
        # This is what the warnings (or warning codes) are stored in instead, so an external process working on the
        #  sample (or in the sample init) can look up what warnings are there.
        self._supp_warn = []

        # This sets up the user-defined coordinate attribute
        self._ra_dec = np.array([ra, dec])
        if name is not None:
            # We don't be liking spaces in source names, we also don't like underscores
            self._name = name.replace(" ", "").replace("_", "-")
        else:
            self._name = coord_to_name(self.ra_dec)

        # This is where profile products generated by XGA will live for this source
        if not os.path.exists(OUTPUT + "profiles/{}".format(self.name)):
            os.makedirs(OUTPUT + "profiles/{}".format(self.name))

        # And create an inventory file for that directory
        if not os.path.exists(OUTPUT + "profiles/{}/inventory.csv".format(self.name)):
            with open(OUTPUT + "profiles/{}/inventory.csv".format(self.name), 'w') as inven:
                inven.writelines(["file_name,obs_ids,insts,info_key,src_name,type"])

        # We now create a directory for custom region files for the source to be stored in
        if not os.path.exists(OUTPUT + "regions/{0}/{0}_custom.reg".format(self.name)):
            os.makedirs(OUTPUT + "regions/{}".format(self.name))
            # And a start to the custom file itself, with red (pnt src) as the default colour
            with open(OUTPUT + "regions/{0}/{0}_custom.reg".format(self.name), 'w') as reggo:
                reggo.write("global color=white\n")

        # Only want ObsIDs, not pointing coordinates as well
        # Don't know if I'll always use the simple method
        matches, excluded = simple_xmm_match(ra, dec)

        # This will store information on the observations that were never included in analysis (so it's distinct from
        #  the disassociated_obs information) - I don't know if this is the solution I'll stick with, but it'll do
        blacklisted_obs = {}
        for row_ind, row in excluded.iterrows():
            # Just blacklist all instruments because for an ObsID to be in the excluded return
            #  from simple_xmm_match this has to be the case
            blacklisted_obs[row['ObsID']] = ['pn', 'mos1', 'mos2']

        # This checks that the observations have at least one usable instrument
        obs = matches["ObsID"].values
        instruments = {o: [] for o in obs}
        for o in obs:
            # As the simple_xmm_match will only tell us about observations in which EVERY instrument is
            #  blacklisted, I have to check in the blacklist to see whether some individual instruments
            #  have to be excluded
            excl_pn = False
            excl_mos1 = False
            excl_mos2 = False
            if o in BLACKLIST['ObsID'].values:
                if BLACKLIST[BLACKLIST['ObsID'] == o]['EXCLUDE_PN'].values[0] == 'T':
                    excl_pn = True
                if BLACKLIST[BLACKLIST['ObsID'] == o]['EXCLUDE_MOS1'].values[0] == 'T':
                    excl_mos1 = True
                if BLACKLIST[BLACKLIST['ObsID'] == o]['EXCLUDE_MOS2'].values[0] == 'T':
                    excl_mos2 = True

            # Here we see if PN is allowed by the census (things like CalClosed observations are excluded in
            #  the census) and if PN is allowed by the blacklist (individual instruments can be blacklisted).
            if matches[matches["ObsID"] == o]["USE_PN"].values[0] and not excl_pn:
                instruments[o].append("pn")
            # If excluded by the blacklist, then that needs
            elif excl_pn:
                # The behaviour writing PN to the dictionary changes slightly depending on whether the ObsID
                #  has an entry yet or not
                if o not in blacklisted_obs:
                    blacklisted_obs[o] = ["pn"]
                else:
                    blacklisted_obs[o] += ['pn']

            # Now we repeat the same process for MOS1 and 2 - its quite clunky and there's probably a more
            #  elegant way that I could write this, but ah well
            if matches[matches["ObsID"] == o]["USE_MOS1"].values[0] and not excl_mos1:
                instruments[o].append("mos1")
            # If excluded by the blacklist, then that needs
            elif excl_mos1:
                # The behaviour writing MOS1 to the dictionary changes slightly depending on whether the ObsID
                #  has an entry yet or not
                if o not in blacklisted_obs:
                    blacklisted_obs[o] = ["mos1"]
                else:
                    blacklisted_obs[o] += ['mos1']

            if matches[matches["ObsID"] == o]["USE_MOS2"].values[0] and not excl_mos2:
                instruments[o].append("mos2")
            # If excluded by the blacklist, then that needs
            elif excl_mos2:
                # The behaviour writing MOS2 to the dictionary changes slightly depending on whether the ObsID
                #  has an entry yet or not
                if o not in blacklisted_obs:
                    blacklisted_obs[o] = ["mos2"]
                else:
                    blacklisted_obs[o] += ['mos2']

        # Information about which ObsIDs/instruments are available, and which have been blacklisted, is stored
        #  in class attributes here.
        self._obs = [o for o in obs if len(instruments[o]) > 0]
        self._instruments = {o: instruments[o] for o in self._obs if len(instruments[o]) > 0}
        self._blacklisted_obs = blacklisted_obs

        # self._obs can be empty after this cleaning step, so do quick check and raise error if so.
        if len(self._obs) == 0:
            raise NoValidObservationsError("{s} has {n} observations ({a}), none of which have the necessary"
                                           " files.".format(s=self.name, n=len(self._obs), a=", ".join(self._obs)))

        # Here I set up the ObsID directories for products generated by XGA to be stored in, they also get an
        #  inventory file to store information about them - largely because some of the informative file names
        #  I was using were longer than 256 characters which my OS does not support
        for o in self._obs:
            if not os.path.exists(OUTPUT + o):
                os.mkdir(OUTPUT + o)

            if not os.path.exists(OUTPUT + '{}/inventory.csv'.format(o)):
                with open(OUTPUT + '{}/inventory.csv'.format(o), 'w') as inven:
                    inven.writelines(['file_name,obs_id,inst,info_key,src_name,type'])

        # Check in a box of half-side 5 arcminutes, should give an idea of which are on-axis
        try:
            on_axis_match = simple_xmm_match(ra, dec, Quantity(5, 'arcmin'))[0]["ObsID"].values
        except NoMatchFoundError:
            on_axis_match = np.array([])
        self._onaxis = list(np.array(self._obs)[np.isin(self._obs, on_axis_match)])

        # nhlookup returns average and weighted average values, so just take the first
        self._nH = nh_lookup(self.ra_dec)[0]
        self._redshift = redshift
        # This method uses the instruments attribute to check and see whether a particular ObsID-Instrument
        #  combination is allowed for this source. As that attribute was constructed using the blacklist information
        #  we can be sure that every ObsID-Instrument combination loaded in here is allowed to be here. The only
        #  other way for them to change is through using the dissociate observation capability
        self._products, region_dict, self._att_files = self._initial_products()

        # Want to update the ObsIDs associated with this source after seeing if all files are present
        self._obs = list(self._products.keys())
        # This (no using the self._products keys) should ensure that the instruments attribute is adjusted properly
        #  to the realities reflected by what event lists are actually available (per the initial products method)
        self._instruments = {o: [cur_i for cur_i in list(self._products[o].keys())
                                 if cur_i != 'combined'] for o in self._obs if len(instruments[o]) > 0}

        self._cosmo = cosmology
        if redshift is not None:
            self._lum_dist = self._cosmo.luminosity_distance(self._redshift)
            self._ang_diam_dist = self._cosmo.angular_diameter_distance(self._redshift)
        else:
            self._lum_dist = None
            self._ang_diam_dist = None
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
        self._fit_stat = {}
        self._dof = {}
        self._total_count_rate = {}
        self._total_exp = {}
        self._luminosities = {}
        self._failed_fits = {}

        # Initialisation of attributes related to Extended and GalaxyCluster sources
        self._peaks = None
        # Initialisation of allowed overdensity radii as None
        if not hasattr(self, 'r200'):
            self._r200 = None
        if not hasattr(self, 'r500'):
            self._r500 = None
        if not hasattr(self, 'r2500'):
            self._r2500 = None
        # Also adding a radius dictionary attribute
        if not hasattr(self, "_radii"):
            self._radii = {}
        # Initialisation of cluster observables as None
        self._richness = None
        self._richness_err = None

        self._wl_mass = None
        self._wl_mass_err = None

        self._peak_lo_en = Quantity(0.5, 'keV')
        self._peak_hi_en = Quantity(2.0, 'keV')
        # Peaks don't really have any meaning for the BaseSource class, so even though this is a boolean variable
        #  when populated properly I set it to None here
        self._use_peak = None

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

        # Now going to save load_fits in an attribute, just because if the observation is cleaned we need to
        #  run _existing_xga_products again, same for load_products
        self._load_fits = load_fits
        self._load_products = load_products

    @property
    def ra_dec(self) -> Quantity:
        """
        A getter for the original ra and dec entered by the user.

        :return: The ra-dec coordinates entered by the user when the source was first defined
        :rtype: Quantity
        """
        # Easier for it be internally kep as a numpy array, but I want the user to have astropy coordinates
        return Quantity(self._ra_dec, 'deg')

    @property
    def default_coord(self) -> Quantity:
        """
        A getter for the default analysis coordinate of this source.
        :return: An Astropy quantity containing the default analysis coordinate.
        :rtype: Quantity
        """
        return self._default_coord

    @default_coord.setter
    def default_coord(self, new_coord: Quantity):
        """
        Setter for the default analysis coordinate of this source.

        :param Quantity new_coord: The new default coordinate.
        """
        if not new_coord.unit.is_equivalent('deg'):
            raise UnitConversionError("The new coordinate must be in degrees")
        else:
            new_coord = new_coord.to("deg")

        self._default_coord = new_coord

    def _initial_products(self) -> Tuple[dict, dict, dict]:
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
            not_these = ["root_xmm_dir", "lo_en", "hi_en", evt_key, "attitude_file"]
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
                prod_objs[prod].src_name = self._name
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
            # This is the path to the region file specified in the configuration file, but the next step is that
            #  we make a local copy (if the original file exists) and then make use of that so that any modifications
            #  don't harm the original file.
            reg_file = xga_conf["XMM_FILES"]["region_file"].format(obs_id=obs_id)

            # Attitude file is a special case of data product, only SAS should ever need it, so it doesn't
            # have a product object
            att_file = xga_conf["XMM_FILES"]["attitude_file"].format(obs_id=obs_id)

            if os.path.exists(evt_file) and os.path.exists(att_file):
                # An instrument subsection of an observation will ONLY be populated if the events file exists
                # Otherwise nothing can be done with it.
                obs_dict[obs_id][inst] = {"events": EventList(evt_file, obs_id=obs_id, instrument=inst,
                                                              stdout_str="", stderr_str="", gen_cmd="")}
                att_dict[obs_id] = att_file
                # Dictionary updated with derived product names
                map_ret = map(read_default_products, en_comb)
                obs_dict[obs_id][inst].update({gen_return[0]: gen_return[1] for gen_return in map_ret})

                # As mentioned above, we make a local copy of the region file if the original file path exists
                #  and if a local copy DOESN'T already exist
                reg_copy_path = OUTPUT+"{o}/{o}_xga_copy.reg".format(o=obs_id)
                if os.path.exists(reg_file) and not os.path.exists(reg_copy_path):
                    # A local copy of the region file is made and used
                    copyfile(reg_file, reg_copy_path)
                    # Regions dictionary updated with path to local region file, if it exists
                    reg_dict[obs_id] = reg_copy_path
                # In the case where there is already a local copy of the region file
                elif os.path.exists(reg_copy_path):
                    reg_dict[obs_id] = reg_copy_path
                else:
                    reg_dict[obs_id] = None

        # Cleans any observations that don't have at least one instrument associated with them
        obs_dict = {o: v for o, v in obs_dict.items() if len(v) != 0}

        if len(obs_dict) == 0:
            raise NoValidObservationsError("{s} has {n} observations ({a}), none of which have the necessary"
                                           " files.".format(s=self.name, n=len(self._obs), a=", ".join(self._obs)))
        return obs_dict, reg_dict, att_dict

    def update_products(self, prod_obj: Union[BaseProduct, BaseAggregateProduct, BaseProfile1D, List[BaseProduct],
                                              List[BaseAggregateProduct], List[BaseProfile1D]],
                        update_inv: bool = True):
        """
        Setter method for the products attribute of source objects. Cannot delete existing products,
        but will overwrite existing products. Raises errors if the ObsID is not associated
        with this source or the instrument is not associated with the ObsID. Lists of products can also be passed
        and will be added to the source storage structure, these lists may also contain None values, as typically
        XGA will return None if a profile fails to generate (for instance), in which case that entry will simply
        be ignored.

        :param BaseProduct/BaseAggregateProduct/BaseProfile1D/List[BaseProduct]/List[BaseProfile1D] prod_obj: The
            new product object(s) to be added to the source object.
        :param bool update_inv: This flag is to avoid unnecessary read-writes when this method is called by a method
            (such as _existing_xga_products) which want to add products to the source storage structure, but don't
            want the inventory file altered (as they know the product is already in there).
        """
        # Aggregate products are things like PSF grids and sets of annular spectra.
        if not isinstance(prod_obj, (BaseProduct, BaseAggregateProduct, BaseProfile1D, list)) and prod_obj is not None:
            raise TypeError("Only product objects can be assigned to sources.")
        elif isinstance(prod_obj, list) and not all([isinstance(p, (BaseProduct, BaseAggregateProduct, BaseProfile1D))
                                                     or p is None for p in prod_obj]):
            raise TypeError("If a list is passed, only product objects (or None values) may be included.")
        elif not isinstance(prod_obj, list):
            prod_obj = [prod_obj]

        for po in prod_obj:
            if po is not None:
                if isinstance(po, Image) or isinstance(po, LightCurve):
                    extra_key = po.storage_key
                    en_key = "bound_{l}-{u}".format(l=float(po.energy_bounds[0].value),
                                                    u=float(po.energy_bounds[1].value))
                elif type(po) == Spectrum or type(po) == AnnularSpectra or isinstance(po, BaseProfile1D) or \
                        isinstance(po, AggregateLightCurve):
                    extra_key = po.storage_key
                elif type(po) == PSFGrid:
                    # The first part of the key is the model used (by default its ELLBETA for example), and
                    #  the second part is the number of bins per side. - Enough to uniquely identify the PSF.
                    extra_key = po.model + "_" + str(po.num_bins)
                else:
                    extra_key = None

                # All information about where to place it in our storage hierarchy can be pulled from the product
                # object itself
                obs_id = po.obs_id
                inst = po.instrument
                p_type = po.type

                # Previously, merged images/exposure maps were stored in a separate dictionary, but now everything lives
                #  together - merged products do get a 'combined' prefix on their product type key though
                if obs_id == "combined":
                    p_type = "combined_" + p_type

                # 'Combined' will effectively be stored as another ObsID
                if "combined" not in self._products:
                    self._products["combined"] = {}

                # The product gets the name of this source object added to it
                po.src_name = self.name

                # Double check that something is trying to add products from another source to the current one.
                if obs_id != "combined" and obs_id not in self._products:
                    raise NotAssociatedError("{o} is not associated with this X-ray source.".format(o=obs_id))
                elif inst != "combined" and inst not in self._products[obs_id]:
                    raise NotAssociatedError("{i} is not associated with XMM observation {o}".format(i=inst, o=obs_id))

                if extra_key is not None and obs_id != "combined":
                    # If there is no entry for this 'extra key' (energy band for instance) already, we must make one
                    if extra_key not in self._products[obs_id][inst]:
                        self._products[obs_id][inst][extra_key] = {}
                    self._products[obs_id][inst][extra_key][p_type] = po

                elif extra_key is None and obs_id != "combined":
                    self._products[obs_id][inst][p_type] = po

                # Here we deal with merged products, they live in the same dictionary, but with no instrument entry
                #  and ObsID = 'combined'
                elif extra_key is not None and obs_id == "combined":
                    if extra_key not in self._products[obs_id]:
                        self._products[obs_id][extra_key] = {}
                    self._products[obs_id][extra_key][p_type] = po

                elif extra_key is None and obs_id == "combined":
                    self._products[obs_id][p_type] = po

                # This is for an image being added, so we look for a matching exposure map. If it exists we can
                #  make a ratemap
                if p_type == "image":
                    # No chance of an expmap being PSF corrected, so we just use the energy key to
                    #  look for one that matches our new image
                    exs = [prod for prod in self.get_products("expmap", obs_id, inst, just_obj=False) if en_key in prod]
                    if len(exs) == 1:
                        new_rt = RateMap(po, exs[0][-1])
                        new_rt.src_name = self.name
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
                            new_rt = RateMap(im[-1], po)
                            new_rt.src_name = self.name
                            self._products[obs_id][inst][im[-2]]["ratemap"] = new_rt

                # The same behaviours hold for combined_image and combined_expmap, but they get
                #  stored in slightly different places
                elif p_type == "combined_image":
                    exs = [prod for prod in self.get_products("combined_expmap", just_obj=False) if en_key in prod]
                    if len(exs) == 1:
                        new_rt = RateMap(po, exs[0][-1])
                        new_rt.src_name = self.name
                        # Remember obs_id for combined products is just 'combined'
                        self._products[obs_id][extra_key]["combined_ratemap"] = new_rt

                elif p_type == "combined_expmap":
                    ims = [prod for prod in self.get_products("combined_image", just_obj=False) if en_key in prod[-2]]
                    if len(ims) != 0:
                        for im in ims:
                            new_rt = RateMap(im[-1], po)
                            new_rt.src_name = self.name
                            self._products[obs_id][im[-2]]["combined_ratemap"] = new_rt

                if isinstance(po, BaseProfile1D) and not os.path.exists(po.save_path):
                    po.save()
                # Here we make sure to store a record of the added product in the relevant inventory file
                if isinstance(po, BaseProduct) and po.obs_id != 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "{}/inventory.csv".format(po.obs_id), dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    # I want only the name of the file as it is in the storage directory, I don't want an
                    #  absolute path, so I remove the leading information about the absolute location in
                    #  the .path string
                    f_name = po.path.split(OUTPUT + "{}/".format(po.obs_id))[-1]

                    # Images, exposure maps, and other such things are not source specific, so I don't want
                    #  the inventory file to assign them a specific source
                    if isinstance(po, Image):
                        s_name = ''
                    else:
                        s_name = po.src_name

                    # Creates new pandas series to be appended to the inventory dataframe
                    new_line = pd.Series([f_name, po.obs_id, po.instrument, info_key, s_name, po.type],
                                         ['file_name', 'obs_id', 'inst', 'info_key', 'src_name', 'type'], dtype=str)
                    # Concatenates the series with the inventory dataframe
                    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)

                    # Checks for rows that are exact duplicates, this should never happen as far as I can tell, but
                    #  if it did I think it would cause problems so better to be safe and add this.
                    inven.drop_duplicates(subset=None, keep='first', inplace=True)
                    # Saves the updated inventory file
                    inven.to_csv(OUTPUT + "{}/inventory.csv".format(po.obs_id), index=False)

                elif isinstance(po, BaseProduct) and po.obs_id == 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "combined/inventory.csv".format(po.obs_id), dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    # We know that this particular product is a combination of multiple ObsIDs, and those ObsIDs
                    #  are not stored explicitly within the product object. However we are currently within the
                    #  source object that they were generated from, thus we do have that information available
                    # Using the _instruments attribute also gives us access to inst information
                    i_str = "/".join([i for o in self._instruments for i in self._instruments[o]])
                    o_str = "/".join([o for o in self._instruments for i in self._instruments[o]])
                    # They cannot be stored as lists for a single column entry in a csv though, so I am smushing
                    #  them into strings

                    f_name = po.path.split(OUTPUT + "combined/")[-1]
                    if isinstance(po, Image):
                        s_name = ''
                    else:
                        s_name = po.src_name

                    # Creates new pandas series to be appended to the inventory dataframe
                    new_line = pd.Series([f_name, o_str, i_str, info_key, s_name, po.type],
                                         ['file_name', 'obs_ids', 'insts', 'info_key', 'src_name', 'type'], dtype=str)
                    # Concatenates the series with the inventory dataframe
                    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)
                    inven.drop_duplicates(subset=None, keep='first', inplace=True)
                    inven.to_csv(OUTPUT + "combined/inventory.csv".format(po.obs_id), index=False)

                elif isinstance(po, BaseProfile1D) and po.obs_id != 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "profiles/{}/inventory.csv".format(self.name), dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    f_name = po.save_path.split(OUTPUT + "profiles/{}/".format(self.name))[-1]
                    i_str = po.instrument
                    o_str = po.obs_id
                    # Creates new pandas series to be appended to the inventory dataframe
                    new_line = pd.Series([f_name, o_str, i_str, info_key, po.src_name, po.type],
                                         ['file_name', 'obs_ids', 'insts', 'info_key', 'src_name', 'type'], dtype=str)
                    # Concatenates the series with the inventory dataframe
                    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)
                    inven.drop_duplicates(subset=None, keep='first', inplace=True)
                    inven.to_csv(OUTPUT + "profiles/{}/inventory.csv".format(self.name), index=False)

                elif isinstance(po, BaseProfile1D) and po.obs_id == 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "profiles/{}/inventory.csv".format(self.name), dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    f_name = po.save_path.split(OUTPUT + "profiles/{}/".format(self.name))[-1]
                    i_str = "/".join([i for o in self._instruments for i in self._instruments[o]])
                    o_str = "/".join([o for o in self._instruments for i in self._instruments[o]])
                    # Creates new pandas series to be appended to the inventory dataframe
                    new_line = pd.Series([f_name, o_str, i_str, info_key, po.src_name, po.type],
                                         ['file_name', 'obs_ids', 'insts', 'info_key', 'src_name', 'type'], dtype=str)
                    # Concatenates the series with the inventory dataframe
                    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)
                    inven.drop_duplicates(subset=None, keep='first', inplace=True)
                    inven.to_csv(OUTPUT + "profiles/{}/inventory.csv".format(self.name), index=False)

    def _existing_xga_products(self, read_fits: bool):
        """
        A method specifically for searching an existing XGA output directory for relevant files and loading
        them in as XGA products. This will retrieve images, exposure maps, and spectra; then the source product
        structure is updated. The method also finds previous fit results and loads them in.

        :param bool read_fits: Boolean flag that controls whether past fits are read back in or not.
        """

        def parse_image_like(file_path: str, exact_type: str, merged: bool = False) -> Union[Image, ExpMap]:
            """
            Very simple little function that takes the path to an XGA generated image-like product (so either an
            image or an exposure map), parses the file path and makes an XGA object of the correct type by using
            the exact_type variable.

            :param str file_path: Absolute path to an XGA-generated data product.
            :param str exact_type: Either 'image' or 'expmap', the type of product that the file_path leads to.
            :param bool merged: Whether this is a merged file or not.
            :return: An XGA Image or ExpMap object.
            :rtype: Union[Image, ExpMap]
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

        def parse_lightcurve(inven_entry: pd.Series) -> LightCurve:
            """
            Very simple little function that takes information on an XGA-generated lightcurve (including a path to
            the file), and sets up a LightCurve product that can be added to the product storage structure
            of the source.

            :param pd.Series inven_entry: The inventory entry from which a LightCurve object should be parsed.
            :return: An XGA LightCurve object
            :rtype: LightCurve
            """
            if inven_entry['src_name'] == self.name:
                # The path, ObsID, and instrument can be read directly from inventory entries - we also use the
                #  'cur_d' parameter from the upper scope to provide an absolute path, as the object will need it
                #  later to read in the data
                rel_path = cur_d + inven_entry['file_name']
                rel_obs_id = inven_entry['obs_id']
                rel_inst = inven_entry['inst']

                # Make sure that the current ObsID and instrument are actually associated with the source
                if rel_obs_id in self.obs_ids and rel_inst in self.instruments[rel_obs_id]:
                    # We split up the information contained in the info key - this is going to tell us what
                    #  settings were used to generate the lightcurve
                    lc_info = inven_entry['info_key'].split("_")

                    # Pull out the energy bounds of the lightcurve, then make them Astropy Quantity
                    rel_lo_en, rel_hi_en = lc_info[1].split("-")
                    rel_lo_en = Quantity(float(rel_lo_en), "keV")
                    rel_hi_en = Quantity(float(rel_hi_en), "keV")

                    # We also need to grab the central coordinates and turn them into an Astropy Quantity
                    rel_central_coord = Quantity([float(lc_info[2].replace('ra', '')),
                                                  float(lc_info[3].replace('dec', ''))], 'deg')

                    # The inner and outer radii are always in degrees at this stage, because then we are
                    #  independent of a cosmology or redshift
                    rel_inn_rad = Quantity(lc_info[4].replace('ri', ''), 'deg')
                    rel_out_rad = Quantity(lc_info[5].replace('ro', ''), 'deg')

                    # The timebin size is always in seconds
                    rel_time_bin = Quantity(lc_info[6].replace('timebin', ''), 's')

                    rel_patt = lc_info[7].replace('pattern', '')

                    # Setting up the lightcurve to be passed back out and stored in the source
                    final_obj = LightCurve(rel_path, rel_obs_id, rel_inst, "", "", "", rel_central_coord, rel_inn_rad,
                                           rel_out_rad, rel_lo_en, rel_hi_en, rel_time_bin, rel_patt, is_back_sub=True)

                else:
                    final_obj = None

            else:
                final_obj = None

            return final_obj

        og_dir = os.getcwd()
        # This is used for spectra that should be part of an AnnularSpectra object
        ann_spec_constituents = {}
        # This is to store whether all components could be loaded in successfully
        ann_spec_usable = {}
        for obs in self._obs:
            if os.path.exists(OUTPUT + obs):
                os.chdir(OUTPUT + obs)
                cur_d = os.getcwd() + '/'
                # Loads in the inventory file for this ObsID
                inven = pd.read_csv("inventory.csv", dtype=str)

                # Here we read in instruments and exposure maps which are relevant to this source
                im_lines = inven[(inven['type'] == 'image') | (inven['type'] == 'expmap')]
                # Instruments is a dictionary with ObsIDs on the top level and then valid instruments on
                #  the lower level. As such we can be sure here we're only reading in instruments we decided
                #  are valid
                for i in self.instruments[obs]:
                    # Fetches lines of the inventory which match the current ObsID and instrument
                    rel_ims = im_lines[(im_lines['obs_id'] == obs) & (im_lines['inst'] == i)]
                    for r_ind, r in rel_ims.iterrows():
                        self.update_products(parse_image_like(cur_d+r['file_name'], r['type']), update_inv=False)

                # This finds the lines of the inventory that are lightCurve entries
                lc_lines = inven[inven['type'] == 'lightcurve']
                for row_ind, row in lc_lines.iterrows():
                    # The parse lightcurve function does check to see if an inventory entry is relevant to this
                    #  source (using the source name), and if the ObsID and instrument are still associated.
                    self.update_products(parse_lightcurve(row), update_inv=False)

                # For spectra we search for products that have the name of this object in, as they are for
                #  specific parts of the observation.
                # Have to replace any + characters with x, as that's what we did in evselect_spectrum due to SAS
                #  having some issues with the + character in file names
                named = [os.path.abspath(f) for f in os.listdir(".") if os.path.isfile(f) and
                         self._name.replace("+", "x") in f and obs in f
                         and (XMM_INST[0] in f or XMM_INST[1] in f or XMM_INST[2] in f)]
                specs = [f for f in named if "_spec" in f.split('/')[-1] and "back" not in f.split('/')[-1]]

                for sp in specs:
                    # Filename contains a lot of useful information, so splitting it out to get it
                    sp_info = sp.split("/")[-1].split("_")
                    # Reading these out into variables mostly for my own sanity while writing this
                    obs_id = sp_info[0]
                    inst = sp_info[1]
                    # I now store the central coordinate in the file name, and read it out into astropy quantity
                    #  for when I need to define the spectrum object
                    central_coord = Quantity([float(sp_info[3].strip('ra')), float(sp_info[4].strip('dec'))], 'deg')
                    # Also read out the inner and outer radii into astropy quantities (I know that
                    #  they will be in degree units).
                    r_inner = Quantity(np.array(sp_info[5].strip('ri').split('and')).astype(float), 'deg')
                    r_outer = Quantity(np.array(sp_info[6].strip('ro').split('and')).astype(float), 'deg')
                    # Check if there is only one r_inner and r_outer value each, if so its a circle
                    #  (otherwise it's an ellipse)
                    if len(r_inner) == 1:
                        r_inner = r_inner[0]
                        r_outer = r_outer[0]

                    # Only check the actual filename, as I have no knowledge of what strings might be in the
                    #  user's path to xga output
                    if 'grpTrue' in sp.split('/')[-1]:
                        grp_ind = sp_info.index('grpTrue')
                        grouped = True
                    else:
                        grouped = False

                    # mincnt or minsn information will only be in the filename if the spectrum is grouped
                    if grouped and 'mincnt' in sp.split('/')[-1]:
                        min_counts = int(sp_info[grp_ind+1].split('mincnt')[-1])
                        min_sn = None
                    elif grouped and 'minsn' in sp.split('/')[-1]:
                        min_sn = float(sp_info[grp_ind+1].split('minsn')[-1])
                        min_counts = None
                    else:
                        # We still need to pass the variables to the spectrum definition, even if it isn't
                        #  grouped
                        min_sn = None
                        min_counts = None

                    # Only if oversampling was applied will it appear in the filename
                    if 'ovsamp' in sp.split('/')[-1]:
                        over_sample = int(sp_info[-2].split('ovsamp')[-1])
                    else:
                        over_sample = None

                    if "region" in sp.split('/')[-1]:
                        region = True
                    else:
                        region = False

                    # I split the 'spec' part of the end of the name of the spectrum, and can use the parts of the
                    #  file name preceding it to search for matching arf/rmf files
                    sp_info_str = cur_d + sp.split('/')[-1].split('_spec')[0]

                    # Fairly self explanatory, need to find all the separate products needed to define an XGA
                    #  spectrum
                    arf = [f for f in named if "arf" in f and "back" not in f and sp_info_str == f.split('.arf')[0]]
                    rmf = [f for f in named if "rmf" in f and "back" not in f and sp_info_str == f.split('.rmf')[0]]

                    # As RMFs can be generated for source and background spectra separately, or one for both,
                    #  we need to check for matching RMFs to the spectrum we found
                    if len(rmf) == 0:
                        rmf = [f for f in named if "rmf" in f and "back" not in f and inst in f and "universal" in f]

                    # Exact same checks for the background spectrum
                    back = [f for f in named if "backspec" in f and inst in f
                            and sp_info_str == f.split('_backspec')[0]]
                    back_arf = [f for f in named if "arf" in f and inst in f
                                and sp_info_str == f.split('_back.arf')[0] and "back" in f]
                    back_rmf = [f for f in named if "rmf" in f and "back" in f and inst in f
                                and sp_info_str == f.split('_back.rmf')[0]]
                    if len(back_rmf) == 0:
                        back_rmf = rmf

                    # If exactly one match has been found for all of the products, we define an XGA spectrum and
                    #  add it the source object.
                    if len(arf) == 1 and len(rmf) == 1 and len(back) == 1 and len(back_arf) == 1 and len(back_rmf) == 1:
                        # Defining our XGA spectrum instance
                        obj = Spectrum(sp, rmf[0], arf[0], back[0], central_coord, r_inner, r_outer, obs_id, inst,
                                       grouped, min_counts, min_sn, over_sample, "", "", "", region, back_rmf[0],
                                       back_arf[0])

                        if "ident" in sp.split('/')[-1]:
                            set_id = int(sp.split('ident')[-1].split('_')[0])
                            ann_id = int(sp.split('ident')[-1].split('_')[1])
                            obj.annulus_ident = ann_id
                            obj.set_ident = set_id
                            if set_id not in ann_spec_constituents:
                                ann_spec_constituents[set_id] = []
                                ann_spec_usable[set_id] = True
                            ann_spec_constituents[set_id].append(obj)
                        else:
                            # And adding it to the source storage structure, but only if its not a member
                            #  of an AnnularSpectra
                            try:
                                self.update_products(obj, update_inv=False)
                            except NotAssociatedError:
                                pass

                    elif len(arf) == 1 and len(rmf) == 1 and len(back) == 1 and len(back_arf) == 0:
                        # Defining our XGA spectrum instance
                        obj = Spectrum(sp, rmf[0], arf[0], back[0], central_coord, r_inner, r_outer, obs_id, inst,
                                       grouped, min_counts, min_sn, over_sample, "", "", "", region)

                        if "ident" in sp.split('/')[-1]:
                            set_id = int(sp.split('ident')[-1].split('_')[0])
                            ann_id = int(sp.split('ident')[-1].split('_')[1])
                            obj.annulus_ident = ann_id
                            obj.set_ident = set_id
                            if set_id not in ann_spec_constituents:
                                ann_spec_constituents[set_id] = []
                                ann_spec_usable[set_id] = True
                            ann_spec_constituents[set_id].append(obj)
                        else:
                            # And adding it to the source storage structure, but only if its not a member
                            #  of an AnnularSpectra
                            try:
                                self.update_products(obj, update_inv=False)
                            except NotAssociatedError:
                                pass
                    else:
                        warn_text = "{src} spectrum {sp} cannot be loaded in due to a mismatch in available" \
                                    " ancillary files".format(src=self.name, sp=sp)
                        if not self._samp_member:
                            warn(warn_text, stacklevel=2)
                        else:
                            self._supp_warn.append(warn_text)
                        if "ident" in sp.split("/")[-1]:
                            set_id = int(sp.split('ident')[-1].split('_')[0])
                            ann_spec_usable[set_id] = False

        os.chdir(og_dir)

        # Here we will load in existing xga profile objects
        os.chdir(OUTPUT + "profiles/{}".format(self.name))
        saved_profs = [pf for pf in os.listdir('.') if '.xga' in pf and 'profile' in pf and self.name in pf]
        for pf in saved_profs:
            try:
                with open(pf, 'rb') as reado:
                    temp_prof = pickle.load(reado)
                    try:
                        self.update_products(temp_prof, update_inv=False)
                    except (NotAssociatedError, AttributeError):
                        pass
            except (EOFError, pickle.UnpicklingError):
                warn_text = "A profile save ({}) appears to be corrupted, it has not been " \
                            "loaded; you can safely delete this file".format(os.getcwd() + '/' + pf)
                if not self._samp_member:
                    # If these errors have been raised then I think that the pickle file has been broken (see issue #935)
                    warn(warn_text, stacklevel=2)
                else:
                    self._supp_warn.append(warn_text)
        os.chdir(og_dir)

        # If spectra that should be a part of annular spectra object(s) have been found, then I need to create
        #  those objects and add them to the storage structure
        if len(ann_spec_constituents) != 0:
            for set_id in ann_spec_constituents:
                try:
                    if ann_spec_usable[set_id]:
                        ann_spec_obj = AnnularSpectra(ann_spec_constituents[set_id])
                        if self._redshift is not None:
                            # If we know the redshift we will add the radii to the annular spectra in proper
                            #  distance units
                            ann_spec_obj.proper_radii = self.convert_radius(ann_spec_obj.radii, 'kpc')

                        # This tries to find any cross-arfs that might have been generated that match the current
                        #  annular spectrum - we'll read them in and add them to the spectrum
                        # TODO This would fall over if 'cross' were in a source name I think? - this will be replaced
                        #  when Jess and I's work on the multi-mission branch, and will be far more efficient as it
                        #  will involve far fewer read operations
                        search_set_ident = "ident{}_cross".format(ann_spec_obj.set_ident)
                        rel_c_arfs = [os.path.join(OUTPUT, oi, f) for oi in ann_spec_obj.obs_ids
                                      for f in os.listdir(OUTPUT + oi)
                                      if search_set_ident in f and "arf" in f and "back" not in f]
                        # SO MANY FOR LOOPS, but I can't think of a better way right now
                        for rel_c_arf in rel_c_arfs:
                            cur_f_name = os.path.basename(rel_c_arf)
                            f_name_parts = cur_f_name.split('.arf')[0].split('_')
                            cur_oi = f_name_parts[0]
                            cur_inst = f_name_parts[1]
                            inn_ann = f_name_parts[-2]
                            out_ann = f_name_parts[-1]
                            ann_spec_obj.add_cross_arf(rel_c_arf, cur_oi, cur_inst, int(inn_ann), int(out_ann),
                                                       ann_spec_obj.set_ident)

                        self.update_products(ann_spec_obj, update_inv=False)
                except AttributeError:
                    # This should hopefully act to catch the NoneType has no attribute problems that have plagued this
                    #  class - while I find a better general solution that should solve this properly
                    pass

        # Here we load in any combined images and exposure maps that may have been generated
        os.chdir(OUTPUT + 'combined')
        cur_d = os.getcwd() + '/'
        # This creates a set of observation-instrument strings that describe the current combinations associated
        #  with this source, for testing against to make sure we're loading in combined images/expmaps that
        #  do belong with this source
        src_oi_set = set([o+i for o in self._instruments for i in self._instruments[o]])

        # Loads in the inventory file for this ObsID
        inven = pd.read_csv("inventory.csv", dtype=str)
        rel_inven = inven[(inven['type'] == 'image') | (inven['type'] == 'expmap')]
        for row_ind, row in rel_inven.iterrows():
            o_split = row['obs_ids'].split('/')
            i_split = row['insts'].split('/')
            # Assemble a set of observations-instrument strings for the current row, to test against the
            #  src_oi_set we assembled earlier
            test_oi_set = set([o+i_split[o_ind] for o_ind, o in enumerate(o_split)])
            # First we make sure the sets are the same length, if they're not then we know before starting that this
            #  row's file can't be okay for us to load in. Then we compute the union between the test_oi_set and
            #  the src_oi_set, and if that is the same length as the original src_oi_set then we know that they match
            #  exactly and the product can be loaded
            if len(src_oi_set) == len(test_oi_set) and len(src_oi_set | test_oi_set) == len(src_oi_set):
                self.update_products(parse_image_like(cur_d+row['file_name'], row['type'], merged=True),
                                     update_inv=False)

        os.chdir(og_dir)

        # Now loading in previous fits
        if os.path.exists(os.path.join(OUTPUT, "XSPEC", self.name, 'inventory.csv')) and read_fits:
            # TODO I NEED TO PUT SO MANY COMMENTS ON THE NEW WAY THIS HAS BEEN SET UP
            # Everything in this file will be relevant to the current source
            cur_fit_inv = pd.read_csv(os.path.join(OUTPUT, 'XSPEC', self.name, 'inventory.csv'),
                                      dtype={'set_ident': 'Int64'})

            # TODO DECIDE HOW TO HANDLE THIS PROPERLY - JUST DROPPING DUPLICATES IS NOT SUFFICIENT FOR ANNULAR SPECTRA
            #  IN PARTICULAR
            # cur_fit_inv = cur_fit_inv.drop_duplicates(['spec_key', 'fit_conf_key', 'obs_ids', 'insts', 'src_name',
            #                                            'type', 'set_ident']).reset_index(drop=True)
            glob_cur_fit_inv = cur_fit_inv[cur_fit_inv['type'] == 'global']

            for row_ind, row in glob_cur_fit_inv.iterrows():
                # We'll read out some key information from the row into variables to make our life a little neater
                fit_file = os.path.join(OUTPUT, 'XSPEC', self.name, row['results_file'])
                spec_key = row['spec_key']
                fit_conf = row['fit_conf_key']
                fit_ois = np.array(row['obs_ids'].split('/')).flatten()
                fit_insts = np.array(row['insts'].split('/')).flatten()

                oi_dict = {oi: list(fit_insts[np.argwhere(fit_ois == oi).T[0]].astype(str))
                           for oi in list(set(fit_ois))}

                # Now we check to see if the same observations are associated with the source currently as they
                #  were at the time of the original fit - if they are not then we are stopping the load process
                #  here and moving onto the next entry
                if oi_dict != self.instruments:
                    break

                # Load in the results table
                fit_data = FITS(fit_file)
                global_results = fit_data["RESULTS"][0]
                model = global_results["MODEL"].strip(" ")

                inst_lums = {}

                assign_res = True
                rel_sps = []
                for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                    sp_oi, sp_inst = line["SPEC_PATH"].strip(" ").split("/")[-1].split("_")[:2]
                    rel_sp = self.get_products('spectrum', sp_oi, sp_inst, spec_key)
                    if len(rel_sp) != 1:
                        assign_res = False
                        break
                    else:
                        rel_sps.append(rel_sp[0])
                # TODO I was going to use this
                #  or len(rel_sps) != len(self.get_products('spectrum', extra_key=spec_key))
                #  to check for spectra that match our description but weren't included in the fit output, but
                #  I realise that would break cases where spectrum_checking has excluded some spectra
                if not assign_res:
                    break

                for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                    rel_sp = rel_sps[line_ind]

                    # Adds information from this fit to the spectrum object.
                    rel_sp.add_fit_data(str(model), line, fit_data["PLOT"+str(line_ind+1)], fit_conf)

                    # The add_fit_data method formats the luminosities nicely, so we grab them back out
                    #  to help grab the luminosity needed to pass to the source object 'add_fit_data' method
                    processed_lums = rel_sp.get_luminosities(model, fit_conf=fit_conf)
                    if rel_sp.instrument not in inst_lums:
                        inst_lums[rel_sp.instrument] = processed_lums

                # Ideally the luminosity reported in the source object will be a PN lum, but its not impossible
                #  that a PN value won't be available. - it shouldn't matter much, lums across the cameras are
                #  consistent
                if "pn" in inst_lums:
                    chosen_lums = inst_lums["pn"]
                    # mos2 generally better than mos1, as mos1 has CCD damage after a certain point in its life
                elif "mos2" in inst_lums:
                    chosen_lums = inst_lums["mos2"]
                elif "mos1" in inst_lums:
                    chosen_lums = inst_lums["mos1"]
                else:
                    chosen_lums = None
                fit_data.close()

                # Push global fit results, luminosities etc. into the corresponding source object.
                self.add_fit_data(model, global_results, chosen_lums, spec_key, fit_conf)

            # ------------ ANNULAR FIT READ IN ------------
            ann_cur_fit_inv = cur_fit_inv[cur_fit_inv['type'] == 'ann'].reset_index(drop=True)

            ann_cur_fit_inv['fit_ident'] = ann_cur_fit_inv['results_file'].apply(lambda x: x.split("_")[1])
            ann_ids = ann_cur_fit_inv['results_file'].apply(lambda x: x.split("_annid")[-1].replace('.fits', ''))
            ann_cur_fit_inv['ann_id'] = ann_ids

            for fit_ident in ann_cur_fit_inv['fit_ident'].unique():
                fit_ann_inv_ent = ann_cur_fit_inv[ann_cur_fit_inv['fit_ident'] == fit_ident].reset_index(drop=True)

                obs_order = {int(an_id): [] for an_id in fit_ann_inv_ent['ann_id'].values}
                ann_lums = {int(an_id): None for an_id in fit_ann_inv_ent['ann_id'].values}
                ann_res = {int(an_id): None for an_id in fit_ann_inv_ent['ann_id'].values}

                rel_ann_sp = self.get_annular_spectra(set_id=fit_ann_inv_ent.iloc[0]['set_ident'])

                assign_res = True
                for row_ind, row in fit_ann_inv_ent.iterrows():
                    if not assign_res:
                        break

                    # We'll read out some key information from the row into variables to make our life a little neater
                    fit_file = os.path.join(OUTPUT, 'XSPEC', self.name, row['results_file'])
                    fit_conf = row['fit_conf_key']
                    fit_ois = np.array(row['obs_ids'].split('/')).flatten()
                    fit_insts = np.array(row['insts'].split('/')).flatten()

                    oi_dict = {oi: list(fit_insts[np.argwhere(fit_ois == oi).T[0]].astype(str))
                               for oi in list(set(fit_ois))}

                    # Now we check to see if the same observations are associated with the source currently as they
                    #  were at the time of the original fit - if they are not then we are stopping the load process
                    #  here and moving onto the next entry
                    if oi_dict != self.instruments:
                        assign_res = False
                        break

                    # Load in the results table
                    fit_data = FITS(fit_file)
                    global_results = fit_data["RESULTS"][0]
                    model = global_results["MODEL"].strip(" ")

                    rel_sps = []
                    inst_lums = {}
                    for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                        spec_info = line["SPEC_PATH"].strip(" ").split("/")[-1].split("_")
                        sp_oi, sp_inst = spec_info[:2]
                        sp_ann_id = int(spec_info[-2])

                        try:
                            rel_sp = rel_ann_sp.get_spectra(sp_ann_id, sp_oi, sp_inst)
                            rel_sps.append(rel_sp)
                        except NoProductAvailableError:
                            assign_res = False
                            break

                        obs_order[sp_ann_id].append([sp_oi, sp_inst])

                    if not assign_res:
                        break

                    for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                        rel_sp = rel_sps[line_ind]

                        # Adds information from this fit to the spectrum object.
                        rel_sp.add_fit_data(str(model), line, fit_data["PLOT"+str(line_ind+1)], fit_conf)

                        # The add_fit_data method formats the luminosities nicely, so we grab them back out
                        #  to help grab the luminosity needed to pass to the source object 'add_fit_data' method
                        processed_lums = rel_sp.get_luminosities(model, fit_conf=fit_conf)
                        if rel_sp.instrument not in inst_lums:
                            inst_lums[rel_sp.instrument] = processed_lums

                    # Ideally the luminosity reported in the source object will be a PN lum, but its not impossible
                    #  that a PN value won't be available. - it shouldn't matter much, lums across the cameras are
                    #  consistent
                    if "pn" in inst_lums:
                        chosen_lums = inst_lums["pn"]
                        # mos2 generally better than mos1, as mos1 has CCD damage after a certain point in its life
                    elif "mos2" in inst_lums:
                        chosen_lums = inst_lums["mos2"]
                    elif "mos1" in inst_lums:
                        chosen_lums = inst_lums["mos1"]
                    else:
                        chosen_lums = None

                    ann_lums[int(row['ann_id'])] = chosen_lums
                    ann_res[int(row['ann_id'])] = global_results
                    fit_data.close()

                if assign_res:
                    try:
                        rel_ann_sp.add_fit_data(model, ann_res, ann_lums, obs_order, fit_conf)
                    except ValueError:
                        # If the results dictionaries don't have the right number of entries a value error may be
                        #  thrown
                        pass

            # ------------ CROSS-ARF ANNULAR FIT READ IN ------------
            carf_cur_fit_inv = cur_fit_inv[cur_fit_inv['type'] == 'ann_carf']

            for row_ind, row in carf_cur_fit_inv.iterrows():
                # Grab the relevant annular spectrum
                rel_ann_sp = self.get_annular_spectra(set_id=row['set_ident'])

                # We'll read out some key information from the row into variables to make our life a little neater
                fit_file = os.path.join(OUTPUT, 'XSPEC', self.name, row['results_file'])
                spec_key = row['spec_key']
                fit_conf = row['fit_conf_key']
                fit_ois = np.array(row['obs_ids'].split('/')).flatten()
                fit_insts = np.array(row['insts'].split('/')).flatten()

                oi_dict = {oi: list(fit_insts[np.argwhere(fit_ois == oi).T[0]].astype(str))
                           for oi in list(set(fit_ois))}

                # Now we check to see if the same observations are associated with the source currently as they
                #  were at the time of the original fit - if they are not then we are stopping the load process
                #  here and moving onto the next entry
                if oi_dict != self.instruments:
                    break

                # Load in the results table
                fit_data = FITS(fit_file)
                global_results = fit_data["RESULTS"][0]
                model = global_results["MODEL"].strip(" ")

                inst_lums = {ann_id: {} for ann_id in rel_ann_sp.annulus_ids}
                obs_order = {int(an_id): [] for an_id in rel_ann_sp.annulus_ids}

                assign_res = True
                rel_sps = []
                for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                    spec_info = line["SPEC_PATH"].strip(" ").split("/")[-1].split("_")
                    sp_oi, sp_inst = spec_info[:2]
                    sp_ann_id = int(spec_info[-2])

                    try:
                        rel_sp = rel_ann_sp.get_spectra(sp_ann_id, sp_oi, sp_inst)
                    except NoProductAvailableError:
                        assign_res = False
                        break

                    rel_sps.append(rel_sp)

                if not assign_res:
                    break

                for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                    rel_sp = rel_sps[line_ind]

                    # Adds information from this fit to the spectrum object.
                    rel_sp.add_fit_data(str(model), line, fit_data["PLOT" + str(line_ind + 1)], fit_conf)

                    obs_order[rel_sp.annulus_ident].append([rel_sp.obs_id, rel_sp.instrument])

                    # The add_fit_data method formats the luminosities nicely, so we grab them back out
                    #  to help grab the luminosity needed to pass to the source object 'add_fit_data' method
                    processed_lums = rel_sp.get_luminosities(model, fit_conf=fit_conf)
                    if rel_sp.instrument not in inst_lums[rel_sp.annulus_ident]:
                        inst_lums[rel_sp.annulus_ident][rel_sp.instrument] = processed_lums

                # Ideally the luminosity reported in the source object will be a PN lum, but its not impossible
                #  that a PN value won't be available. - it shouldn't matter much, lums across the cameras are
                #  consistent
                chosen_lums = {}
                for cur_ann_id in inst_lums:
                    if "pn" in inst_lums[cur_ann_id]:
                        cur_chos_lum = inst_lums[cur_ann_id]["pn"]
                    elif "mos2" in inst_lums[cur_ann_id]:
                        cur_chos_lum = inst_lums[cur_ann_id]["mos2"]
                    else:
                        cur_chos_lum = inst_lums[cur_ann_id]["mos1"]
                    chosen_lums[cur_ann_id] = cur_chos_lum

                # Here our main problem is untangling the parameters in the results table for this fit, as
                #  we need to be able to assign them to our N annuli. This starts by reading out all
                #  the column names, and figuring out where the fit parameters (which will be relevant
                #  to a particular annulus) start.
                col_names = np.array(global_results.dtype.names)
                # We know that fit parameters start after the DOF entry, because that is how we designed
                #  the output files, so we can figure out what index to split on that will let us get
                #  fit parameters in one array and the general parameters in the other.
                arg_split = np.argwhere(col_names == 'DOF')[0][0]
                # We split off the columns that aren't parameters
                not_par_names = col_names[:arg_split + 1]
                # Then we tile them, as we're going to be reading out these values repeatedly (i.e. N times
                #  where N is the number of annuli). Strictly speaking all the goodness of fit info is not
                #  for individual annuli like it is when we don't cross-arf-fit, but the annular spectrum
                #  still expects there to be an entry per annulus
                not_par_names = np.tile(not_par_names[..., None], rel_ann_sp.num_annuli).T
                # We select only the column names which were fit parameters, these we need to split up
                #  by figuring out which belong to each annulus
                col_names = col_names[arg_split + 1:]
                # Now we figure out how many parameters per annuli there are, this approach is valid
                #  because the model setups of each annuli are going to be identical
                par_per_ann = len(col_names) / rel_ann_sp.num_annuli
                if (par_per_ann % 1) != 0:
                    raise XGADeveloperError("Assigning results to annular spectrum after cross-arf fit"
                                            " has resulted in a non-integer number of parameters per"
                                            " annulus. This is the fault of the developers.")
                # Now we can split the parameter names into those that belong with each
                par_for_ann = col_names.reshape(rel_ann_sp.num_annuli, int(par_per_ann))
                # Now we're adding the not-fit-parameters back on to the front of each row - that way
                #  the not-fit-parameter info will be added into each annulus' information to be passed
                #  to the annular spectrum
                par_for_ann = np.concatenate([not_par_names, par_for_ann], axis=1)

                # Then we put the results in a dictionary, the way the annulus wants it
                ann_results = {ann_id: fit_data['RESULTS'][par_for_ann[ann_id]][0]
                               for ann_id in rel_ann_sp.annulus_ids}

                rel_ann_sp.add_fit_data(model, ann_results, chosen_lums, obs_order, fit_conf)
                fit_data.close()

        os.chdir(og_dir)

        # And finally loading in any conversion factors that have been calculated using XGA's fakeit interface
        if os.path.exists(OUTPUT + "XSPEC/" + self.name) and read_fits:
            conv_factors = [OUTPUT + "XSPEC/" + self.name + "/" + f for f in os.listdir(OUTPUT + "XSPEC/" + self.name)
                            if ".xcm" not in f and "conv_factors" in f]
            for conv_path in conv_factors:
                res_table = pd.read_csv(conv_path, dtype={"lo_en": str, "hi_en": str})
                # Gets the model name from the file name of the output results table
                model = conv_path.split("_")[-3]

                # We can infer the storage key from the name of the results table, just makes it easier to
                #  grab the correct spectra
                storage_key = conv_path.split('/')[-1].split(self.name)[-1][1:].split(model)[0][:-1]

                # Grabs the ObsID+instrument combinations from the headers of the csv. Makes sure they are unique
                #  by going to a set (because there will be two columns for each ObsID+Instrument, rate and Lx)
                # First two columns are skipped because they are energy limits
                combos = list(set([c.split("_")[1] for c in res_table.columns[2:]]))
                # Getting the spectra for each column, then assigning rates and luminosities.
                # Due to the danger of a fit using a piece of data (an ObsID-instrument combo) that isn't currently
                #  associated with the source, we first fetch the spectra, then in a second loop we assign the factors
                rel_spec = []
                try:
                    for comb in combos:
                        spec = self.get_products("spectrum", comb[:10], comb[10:], extra_key=storage_key)[0]
                        rel_spec.append(spec)

                    for comb_ind, comb in enumerate(combos):
                        rel_spec[comb_ind].add_conv_factors(res_table["lo_en"].values, res_table["hi_en"].values,
                                                            res_table["rate_{}".format(comb)].values,
                                                            res_table["Lx_{}".format(comb)].values, model)

                # This triggers in the case of something like issue #738, where a previous fit used data that is
                #  not loaded into this source (either because it was manually removed, or because the central
                #  position has changed etc.)
                except NotAssociatedError:
                    warn_text = "Existing fit for {s} could not be loaded due to a mismatch in available " \
                                "data".format(s=self.name)
                    if not self._samp_member:
                        warn(warn_text, stacklevel=2)
                    else:
                        self._supp_warn.append(warn_text)

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

        if obs_id not in self._products and obs_id is not None:
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

        :return: Two dictionaries, the first contains the regions for each of the ObsIDs and the second contains
            the regions that have been very simply matched to the source. These should be ordered from closest to
            furthest from the passed source coordinates.
        :rtype: Tuple[dict, dict]
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

        # Read in the custom region file that every XGA has associated with it. Sources within will be added to the
        #  source list for every ObsID?
        custom_regs = Regions.read(OUTPUT + "regions/{0}/{0}_custom.reg".format(self.name), format='ds9').regions
        for reg in custom_regs:
            if not isinstance(reg, SkyRegion):
                raise TypeError("Custom sources can only be defined in RA-Dec coordinates.")

        reg_dict = {}
        match_dict = {}
        # As we only allow one set of regions per observation, we shall assume that we can use the
        # WCS transform from ANY of the images to convert pixels to degrees

        for obs_id in reg_paths:
            if reg_paths[obs_id] is not None:
                # With the newer versions of regions we're now using, there is an explicit check for width and height
                #  of regions being positive (definitely a good idea) - finding such a region in a file will trigger
                #  a ValueError, and I'd like to catch it and add more context
                try:
                    ds9_regs = Regions.read(reg_paths[obs_id], format='ds9').regions
                except ValueError as err:
                    err.args = (err.args[0] + "- {o} is the associated ObsID.".format(o=obs_id), )
                    raise err

                # Apparently can happen that there are no regions in a region file, so if that is the case
                #  then I just set the ds9_regs to [None] because I know the rest of the code can deal with that.
                #  It can't deal with an empty list
                if len(ds9_regs) == 0:
                    ds9_regs = [None]
            else:
                ds9_regs = [None]

            # If either of these are fulfilled then we MUST have a WCS - even though custom regions are always
            #  in RA-DEC, the regions module requires a passed wcs to be able to use 'contains' - will have to
            #  change all this at some point
            if ds9_regs[0] is not None or len(custom_regs) != 0:
                # Grab all images for the ObsID, instruments across an ObsID have the same WCS (other than in cases
                #  where they were generated with different resolutions).
                #  TODO see issue #908, figure out how to support different resolutions of image
                try:
                    ims = self.get_images(obs_id)
                    # Make sure that the return is in a list
                    if not isinstance(ims, list):
                        ims = [ims]
                except NoProductAvailableError:
                    raise NoProductAvailableError("There is no image available for observation {o}, associated "
                                                  "with {n}. An image is currently required to check for sky "
                                                  "coordinates being present within a sky region - though hopefully "
                                                  "no-one will ever see this because I'll have fixed "
                                                  "it!".format(o=obs_id, n=self.name))
                    w = None
                else:
                    w = ims[0].radec_wcs

            if isinstance(ds9_regs[0], PixelRegion):
                # If regions exist in pixel coordinates, we need an image WCS to convert them to RA-DEC, so we need
                #  one of the images supplied in the config file, not anything that XGA generates.
                #  But as this method is only run once, before XGA generated products are loaded in, it
                #  should be fine
                if w is None:
                    raise NoProductAvailableError("There is no image available for observation {o}, associated "
                                                  "with {n}. An image is currently required to translate pixel regions "
                                                  "to RA-DEC.".format(o=obs_id, n=self.name))

                sky_regs = [reg.to_sky(w) for reg in ds9_regs]
                reg_dict[obs_id] = np.array(sky_regs)
            elif isinstance(ds9_regs[0], SkyRegion):
                reg_dict[obs_id] = np.array(ds9_regs)
            else:
                # So there is an entry in this for EVERY ObsID
                reg_dict[obs_id] = np.array([None])

            # Here we add the custom sources to the source list, we know they are sky regions as we have
            #  already enforced it. If there was no region list for a particular ObsID (detected by the first
            #  entry in the reg dict being None) and there IS a custom region, we just replace the None with the
            #  custom region
            if reg_dict[obs_id][0] is not None:
                reg_dict[obs_id] = np.append(reg_dict[obs_id], custom_regs)
            elif reg_dict[obs_id][0] is None and len(custom_regs) != 0:
                reg_dict[obs_id] = np.array(custom_regs)
            else:
                reg_dict[obs_id] = np.array([None])

            # I'm going to ensure that all regions are elliptical, I don't want to hunt through every place in XGA
            #  where I made that assumption
            for reg_ind, reg in enumerate(reg_dict[obs_id]):
                if isinstance(reg, CircleSkyRegion):
                    # Multiply radii by two because the ellipse based sources want HEIGHT and WIDTH, not RADIUS
                    # Give small angle (though won't make a difference as circular) to avoid problems with angle=0
                    #  that I've noticed previously
                    new_reg = EllipseSkyRegion(reg.center, reg.radius*2, reg.radius*2, Quantity(3, 'deg'))
                    new_reg.visual['edgecolor'] = reg.visual['edgecolor']
                    new_reg.visual['facecolor'] = reg.visual['facecolor']
                    reg_dict[obs_id][reg_ind] = new_reg

            # Hopefully this bodge doesn't have any unforeseen consequences
            if reg_dict[obs_id][0] is not None and len(reg_dict[obs_id]) > 1:
                # Quickly calculating distance between source and center of regions, then sorting
                # and getting indices. Thus I only match to the closest 5 regions.
                diff_sort = np.array([dist_from_source(r) for r in reg_dict[obs_id]]).argsort()
                # Unfortunately due to a limitation of the regions module I think you need images
                #  to do this contains match...
                within = np.array([reg.contains(SkyCoord(*self._ra_dec, unit='deg'), w)
                                   for reg in reg_dict[obs_id][diff_sort[0:5]]])

                # Make sure to re-order the region list to match the sorted within array
                reg_dict[obs_id] = reg_dict[obs_id][diff_sort]

                # Expands it so it can be used as a mask on the whole set of regions for this observation
                within = np.pad(within, [0, len(diff_sort) - len(within)])
                match_dict[obs_id] = within
            # In the case of only one region being in the list, we simplify the above expression
            elif reg_dict[obs_id][0] is not None and len(reg_dict[obs_id]) == 1:
                if reg_dict[obs_id][0].contains(SkyCoord(*self._ra_dec, unit='deg'), w):
                    match_dict[obs_id] = np.array([True])
                else:
                    match_dict[obs_id] = np.array([False])
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
            raise NotAssociatedError("{o} is not associated with {s}".format(o=obs_id, s=self.name))
        else:
            return self._att_files[obs_id]

    @property
    def obs_ids(self) -> List[str]:
        """
        Property getter for ObsIDs associated with this source that are confirmed to have events files.

        :return: A list of the associated XMM ObsIDs.
        :rtype: List[str]
        """
        return self._obs

    @property
    def blacklisted(self) -> Dict:
        """
        A property getter that returns the dictionary of ObsIDs and their instruments which have been
        blacklisted, and thus not considered for use in any analysis of this source.

        :return: The dictionary (with ObsIDs as keys) of blacklisted data.
        :rtype: Dict
        """
        return self._blacklisted_obs

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
        #  University of Edinburgh - 2005. These are the default XGA colour meanings
        # Red - Point source
        # Green - Extended source
        # Magenta - PSF-sized extended source
        # Blue - Extended source with significant point source contribution
        # Cyan - Extended source with significant Run1 contribution
        # Yellow - Extended source with less than 10 counts
        try:
            # Gets the allowed colours for the current source type
            allowed_colours = SRC_REGION_COLOURS[source_type]
        except KeyError:
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
                # This sets up an array of matched regions, accounting for the problems that can occur when
                #  there is only one region in the region list (numpy's indexing gets very angry). The array
                #  of matched region(s) set up here is used in this method.
                if len(self._initial_regions[obs]) == 1 and not self._initial_region_matches[obs][0]:
                    init_region_matches = np.array([])
                elif len(self._initial_regions[obs]) == 1 and self._initial_region_matches[obs][0]:
                    init_region_matches = self._initial_regions[obs]
                elif len(self._initial_regions[obs][self._initial_region_matches[obs]]) == 0:
                    init_region_matches = np.array([])
                else:
                    init_region_matches = self._initial_regions[obs][self._initial_region_matches[obs]]

                # If there are no matches then the returned result is just None.
                if len(init_region_matches) == 0:
                    results_dict[obs] = None
                else:
                    interim_reg = []
                    # The only solution I could think of is to go by the XCS standard of region files, so green
                    #  is extended, red is point etc. - not ideal but I'll just explain in the documentation
                    # for entry in self._initial_regions[obs][self._initial_region_matches[obs]]:
                    for entry in init_region_matches:
                        if entry.visual["edgecolor"] in allowed_colours:
                            interim_reg.append(entry)

                    # Different matching possibilities
                    if len(interim_reg) == 0:
                        results_dict[obs] = None
                    elif len(interim_reg) == 1:
                        results_dict[obs] = interim_reg[0]
                    # Matching to multiple sources would be very problematic, so throw an error
                    elif len(interim_reg) > 1 and source_type == "pnt":
                        # I made the _load_regions method sort the outputted region dictionaries by distance from the
                        #  input coordinates, so I know that the 0th entry will be the closest to the source coords.
                        #  Hence I choose that one for pnt source multi-matches like this, see comment 2 of issue #639
                        #  for an example.
                        results_dict[obs] = interim_reg[0]
                        warn_text = "{ns} matches for the point source {n} are found in the {o} region " \
                                    "file. The source nearest to the passed coordinates is accepted, all others " \
                                    "will be placed in the alternate match category and will not be removed " \
                                    "by masks.".format(o=obs, n=self.name, ns=len(interim_reg))
                        if not self._samp_member:
                            warn(warn_text, stacklevel=2)
                        else:
                            self._supp_warn.append(warn_text)

                    elif len(interim_reg) > 1 and source_type == "ext":
                        raise MultipleMatchError("More than one match for {n} is found in the region file "
                                                 "for observation {o}, this cannot yet be dealt with "
                                                 "for extended sources.".format(o=obs, n=self.name))

                # Alt match is used for when there is a secondary match to a point source
                alt_match_reg = [entry for entry in init_region_matches if entry != results_dict[obs]]
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
    def detected(self) -> dict:
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

    @property
    def matched_regions(self) -> dict:
        """
        Property getter for the matched regions associated with this particular source.

        :return: A dictionary of matching regions, or None if such a match has not been performed.
        :rtype: dict
        """
        return self._regions

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
            warn("You cannot use custom central coordinates with a region from supplied region files", stacklevel=2)

        if central_coord is None:
            central_coord = self._default_coord

        if type(central_coord) == Quantity:
            centre = SkyCoord(*central_coord.to("deg"))
        elif type(central_coord) == SkyCoord:
            centre = central_coord
        else:
            raise TypeError("central_coord must be of type Quantity or SkyCoord.")

        # In case combined gets passed as the ObsID at any point
        if obs_id == "combined":
            obs_id = None

        # The search radius won't be used by the user, just peak finding solutions
        allowed_rtype = ["r2500", "r500", "r200", "region", "custom", "search", "point"]
        if type(self) == BaseSource:
            raise TypeError("BaseSource class does not have the necessary information "
                            "to select a source region.")
        elif obs_id is not None and obs_id not in self.obs_ids:
            raise NotAssociatedError("The ObsID {o} is not associated with {s}.".format(o=obs_id, s=self.name))
        elif reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))
        elif reg_type == "region" and obs_id is None:
            raise ValueError("ObsID cannot be None when getting region file regions.")
        elif reg_type == "region" and obs_id is not None:
            src_reg = self._regions[obs_id]
        elif reg_type in ["r2500", "r500", "r200"] and reg_type not in self._radii:
            raise ValueError("There is no {r} associated with {s}".format(r=reg_type, s=self.name))
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

    def get_interloper_regions(self, flattened: bool = False) -> Union[List, Dict]:
        """
        This get method provides a way to access the regions that have been designated as interlopers (i.e.
        not the source region that a particular Source has been designated to investigate) for all observations.
        They can either be retrieved in a dictionary with ObsIDs as the keys, or a flattened single list with no
        ObsID context.

        :param bool flattened: If true then the regions are returned as a single list of region objects. Otherwise
            they are returned as a dictionary with ObsIDs as keys. Default is False.
        :return: Either a list of region objects, or a dictionary with ObsIDs as keys.
        :rtype: Union[List,Dict]
        """
        if type(self) == BaseSource:
            raise TypeError("BaseSource objects don't have enough information to know which sources "
                            "are interlopers.")

        # If flattened then a list is returned rather than the original dictionary with
        if not flattened:
            ret_reg = self._other_regions
        else:
            # Iterate through the ObsIDs in the dictionary and add the resulting lists together
            ret_reg = []
            for o in self._other_regions:
                ret_reg += self._other_regions[o]

        return ret_reg

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

        if central_coord is None:
            central_coord = self._default_coord

        # Don't need to do a bunch of checks, because the method I call to make the
        #  mask does all the checks anyway
        src_reg, bck_reg = self.source_back_regions(reg_type, obs_id, central_coord)

        # I assume that if no ObsID is supplied, then the user wishes to have a mask for the combined data
        if obs_id is None:
            comb_images = self.get_products("combined_image")
            if len(comb_images) != 0:
                mask_image = comb_images[0]
            else:
                raise NoProductAvailableError("There are no combined products available to generate a mask for.")
        else:
            # Just grab the first instrument that comes out the get method, the masks should be the same.
            mask_image = self.get_products("image", obs_id)[0]

        mask = src_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
        back_mask = bck_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)

        # If the masks are None, then they are set to an array of zeros
        if mask is None:
            mask = np.zeros(mask_image.shape)
        if back_mask is None:
            back_mask = np.zeros(mask_image.shape)

        return mask, back_mask

    def _generate_interloper_mask(self, mask_image: Image) -> ndarray:
        """
        Internal method that makes interloper masks in the first place; I allow this because interloper
        masks will never change, so can be safely generated and stored in an init of a source class.

        :param Image mask_image: The image for which to create the interloper mask.
        :return: A numpy array of 0s and 1s which acts as a mask to remove interloper sources.
        :rtype: ndarray
        """
        masks = []
        for r in self._interloper_regions:
            if r is not None:
                # The central coordinate of the current region
                c = Quantity([r.center.ra.value, r.center.dec.value], 'deg')
                try:
                    # Checks if the central coordinate can be converted to pixels for the mask_image, if it fails then
                    #  its likely off of the image, as a ValueError will be thrown if a pixel coordinate is less
                    #  than zero, or greater than the size of the image in that axis
                    cp = mask_image.coord_conv(c, 'pix')
                    pr = r.to_pixel(mask_image.radec_wcs)

                    # If the rotation angle is zero then the conversion to mask by the regions module will be upset,
                    #  so I perturb the angle by 0.1 degrees
                    if isinstance(pr, EllipsePixelRegion) and pr.angle.value == 0:
                        pr.angle += Quantity(0.1, 'deg')
                    masks.append(pr.to_mask().to_image(mask_image.shape))
                except ValueError:
                    pass

        # masks = [reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
        #          for reg in self._interloper_regions if reg is not None]
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

    def get_custom_mask(self, outer_rad: Quantity, inner_rad: Quantity = Quantity(0, 'arcsec'), obs_id: str = None,
                        central_coord: Quantity = None, remove_interlopers: bool = True) -> np.ndarray:
        """
        A simple, but powerful method, to generate mask a mask within a custom radius for a given ObsID.

        :param Quantity outer_rad: The outer radius of the mask.
        :param Quantity inner_rad: The inner radius of the mask, the default is zero arcseconds.
        :param str obs_id: The ObsID for which to generate the mask, default is None which will return a mask
            generated from a combined image.
        :param Quantity central_coord: The central coordinates of the mask, the default is None which
            will use the default coordinates of the source.
        :param bool remove_interlopers: Whether an interloper mask should be combined with the custom mask to
            remove interloper point sources.
        :return: A numpy array containing the desired mask.
        :rtype: np.ndarray
        """
        if central_coord is None:
            central_coord = self._default_coord

        if obs_id is None:
            # Doesn't matter which combined ratemap, just need the size and coord conversion powers
            rt = self.get_combined_ratemaps()
        else:
            # Again so long as the image matches the ObsID passed in by the user I don't care what instrument
            #  its from
            rt = self.get_ratemaps(obs_id=obs_id)

        # If its not an instance of RateMap that means a list of RateMaps has been returned, and as I only want
        #  the WCS information and the shape of the image I don't care which one we use
        if not isinstance(rt, RateMap):
            rt = rt[0]

        # Convert the inner and outer radii to degrees so they can be easily converted to pixels
        outer_rad = self.convert_radius(outer_rad, 'deg')
        inner_rad = self.convert_radius(inner_rad, 'deg')
        pix_to_deg = pix_deg_scale(central_coord, rt.radec_wcs)

        # Making sure the inner and outer radii are whole integer numbers, as they are now in pixel units
        outer_rad = np.array([int(np.ceil(outer_rad / pix_to_deg).value)])
        inner_rad = np.array([int(np.floor(inner_rad / pix_to_deg).value)])
        # Convert the chosen central coordinates to pixels
        pix_centre = rt.coord_conv(central_coord, 'pix')

        # Generate our custom mask
        custom_mask = annular_mask(pix_centre, inner_rad, outer_rad, rt.shape)

        # And applying an interloper mask if the user wants that.
        if remove_interlopers:
            interloper_mask = self.get_interloper_mask(obs_id)
            custom_mask = custom_mask*interloper_mask
        return custom_mask

    def get_snr(self, outer_radius: Union[Quantity, str], central_coord: Quantity = None, lo_en: Quantity = None,
                hi_en: Quantity = None, obs_id: str = None, inst: str = None, psf_corr: bool = False,
                psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15,
                allow_negative: bool = False,  exp_corr: bool = True) -> float:
        """
        This takes a region type and central coordinate and calculates the signal to noise ratio.
        The background region is constructed using the back_inn_rad_factor and back_out_rad_factor
        values, the defaults of which are 1.05*radius and 1.5*radius respectively.

        :param Quantity/str outer_radius: The radius that SNR should be calculated within, this can either be a
            named radius such as r500, or an astropy Quantity.
        :param Quantity central_coord: The central coordinate of the region.
        :param Quantity lo_en: The lower energy bound of the ratemap to use to calculate the SNR. Default is None,
            in which case the lower energy bound for peak finding will be used (default is 0.5keV).
        :param Quantity hi_en: The upper energy bound of the ratemap to use to calculate the SNR. Default is None,
            in which case the upper energy bound for peak finding will be used (default is 2.0keV).
        :param str obs_id: An ObsID of a specific ratemap to use for the SNR calculation. Default is None, which
            means the combined ratemap will be used. Please note that inst must also be set to use this option.
        :param str inst: The instrument of a specific ratemap to use for the SNR calculation. Default is None, which
            means the combined ratemap will be used.
        :param bool psf_corr: Sets whether you wish to use a PSF corrected ratemap or not.
        :param str psf_model: If the ratemap you want to use is PSF corrected, this is the PSF model used.
        :param int psf_bins: If the ratemap you want to use is PSF corrected, this is the number of PSFs per
            side in the PSF grid.
        :param str psf_algo: If the ratemap you want to use is PSF corrected, this is the algorithm used.
        :param int psf_iter: If the ratemap you want to use is PSF corrected, this is the number of iterations.
        :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
            zero, which results in a lower signal to noise (and can result in a negative signal to noise).
        :param bool exp_corr: Should signal to noises be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
        :return: The signal to noise ratio.
        :rtype: float
        """
        # Checking if the user passed any energy limits of their own
        if lo_en is None:
            lo_en = self._peak_lo_en
        if hi_en is None:
            hi_en = self._peak_hi_en

        # Parsing the ObsID and instrument options, see if they want to use a specific ratemap
        if all([obs_id is None, inst is None]):
            # Here the user hasn't set ObsID or instrument, so we use the combined data
            rt = self.get_combined_ratemaps(lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)

        elif all([obs_id is not None, inst is not None]):
            # Both ObsID and instrument have been set by the user
            rt = self.get_ratemaps(obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)
        else:
            raise ValueError("If you wish to use a specific ratemap for {s}'s signal to noise calculation, please "
                             " pass both obs_id and inst.".format(s=self.name))

        if isinstance(outer_radius, str):
            # Grabs the interloper removed source and background region masks. If the ObsID is None the get_mask
            #  method understands that means it should return the mask for the combined data
            src_mask, bck_mask = self.get_mask(outer_radius, obs_id, central_coord)
        else:
            # Here we have the case where the user has passed a custom outer radius, so we need to generate a
            #  custom mask for it
            src_mask = self.get_custom_mask(outer_radius, obs_id=obs_id, central_coord=central_coord)
            bck_mask = self.get_custom_mask(outer_radius*self._back_out_factor, outer_radius*self._back_inn_factor,
                                            obs_id=obs_id, central_coord=central_coord)

        # We use the ratemap's built in signal to noise calculation method
        sn = rt.signal_to_noise(src_mask, bck_mask, exp_corr, allow_negative)

        return sn

    def get_counts(self, outer_radius: Union[Quantity, str], central_coord: Quantity = None, lo_en: Quantity = None,
                   hi_en: Quantity = None, obs_id: str = None, inst: str = None, psf_corr: bool = False,
                   psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15) -> Quantity:
        """
        This takes a region type and central coordinate and calculates the background subtracted X-ray counts.
        The background region is constructed using the back_inn_rad_factor and back_out_rad_factor
        values, the defaults of which are 1.05*radius and 1.5*radius respectively.

        :param Quantity/str outer_radius: The radius that counts should be calculated within, this can either be a
            named radius such as r500, or an astropy Quantity.
        :param Quantity central_coord: The central coordinate of the region.
        :param Quantity lo_en: The lower energy bound of the ratemap to use to calculate the counts. Default is None,
            in which case the lower energy bound for peak finding will be used (default is 0.5keV).
        :param Quantity hi_en: The upper energy bound of the ratemap to use to calculate the counts. Default is None,
            in which case the upper energy bound for peak finding will be used (default is 2.0keV).
        :param str obs_id: An ObsID of a specific ratemap to use for the counts calculation. Default is None, which
            means the combined ratemap will be used. Please note that inst must also be set to use this option.
        :param str inst: The instrument of a specific ratemap to use for the counts calculation. Default is None, which
            means the combined ratemap will be used.
        :param bool psf_corr: Sets whether you wish to use a PSF corrected ratemap or not.
        :param str psf_model: If the ratemap you want to use is PSF corrected, this is the PSF model used.
        :param int psf_bins: If the ratemap you want to use is PSF corrected, this is the number of PSFs per
            side in the PSF grid.
        :param str psf_algo: If the ratemap you want to use is PSF corrected, this is the algorithm used.
        :param int psf_iter: If the ratemap you want to use is PSF corrected, this is the number of iterations.
        :return: The background subtracted counts.
        :rtype: Quantity
        """
        # Checking if the user passed any energy limits of their own
        if lo_en is None:
            lo_en = self._peak_lo_en
        if hi_en is None:
            hi_en = self._peak_hi_en

        # Parsing the ObsID and instrument options, see if they want to use a specific ratemap
        if all([obs_id is None, inst is None]):
            # Here the user hasn't set ObsID or instrument, so we use the combined data
            rt = self.get_combined_ratemaps(lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)

        elif all([obs_id is not None, inst is not None]):
            # Both ObsID and instrument have been set by the user
            rt = self.get_ratemaps(obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter)
        else:
            raise ValueError("If you wish to use a specific ratemap for {s}'s signal to noise calculation, please "
                             " pass both obs_id and inst.".format(s=self.name))

        if isinstance(outer_radius, str):
            # Grabs the interloper removed source and background region masks. If the ObsID is None the get_mask
            #  method understands that means it should return the mask for the combined data
            src_mask, bck_mask = self.get_mask(outer_radius, obs_id, central_coord)
        else:
            # Here we have the case where the user has passed a custom outer radius, so we need to generate a
            #  custom mask for it
            src_mask = self.get_custom_mask(outer_radius, obs_id=obs_id, central_coord=central_coord)
            bck_mask = self.get_custom_mask(outer_radius*self._back_out_factor, outer_radius*self._back_inn_factor,
                                            obs_id=obs_id, central_coord=central_coord)

        # We use the ratemap's built in background subtracted counts calculation method
        cnts = rt.background_subtracted_counts(src_mask, bck_mask)

        return cnts

    def regions_within_radii(self, inner_radius: Quantity, outer_radius: Quantity, deg_central_coord: Quantity,
                             regions_to_search: Union[np.ndarray, list] = None) -> np.ndarray:
        """
        This function finds and returns any interloper regions (by default) that have any part of their boundary
        within the specified radii, centered on the specified central coordinate. Users may also pass their own
        array of regions to check.

        :param Quantity inner_radius: The inner radius of the area to search for interlopers in.
        :param Quantity outer_radius: The outer radius of the area to search for interlopers in.
        :param Quantity deg_central_coord: The central coordinate (IN DEGREES) of the area to search for
            interlopers in.
        :param np.ndarray/list regions_to_search: An optional parameter that allows the user to pass a specific
            list of regions to check. Default is None, in which case the interloper_regions internal list
            will be used.
        :return: A numpy array of the interloper regions (or user passed regions) within the specified area.
        :rtype: np.ndarray
        """
        def perimeter_points(reg_cen_x: float, reg_cen_y: float, reg_major_rad: float, reg_minor_rad: float,
                             rotation: float) -> np.ndarray:
            """
            An internal function to generate thirty x-y positions on the boundary of a particular region.

            :param float reg_cen_x: The x position of the centre of the region, in degrees.
            :param float reg_cen_y: The y position of the centre of the region, in degrees
            :param float reg_major_rad: The semi-major axis of the region, in degrees.
            :param float reg_minor_rad: The semi-minor axis of the region, in degrees.
            :param float rotation: The rotation of the region, in radians.
            :return: An array of thirty x-y coordinates on the boundary of the region.
            :rtype: np.ndarray
            """
            # Just the numpy array of angles (in radians) to find the x-y points of
            angs = np.linspace(0, 2 * np.pi, 30)

            # This is just the parametric equation of an ellipse - I only include the displacement to the
            #  central coordinates of the region AFTER it has been rotated
            x = reg_major_rad * np.cos(angs)
            y = reg_minor_rad * np.sin(angs)

            # Sets of the rotation matrix
            rot_mat = np.array([[np.cos(rotation), -1 * np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]])

            # Just rotates the edge coordinates to match the known rotation of this particular region
            edge_coords = (rot_mat @ np.vstack([x, y])).T

            # Now I re-centre the region
            edge_coords[:, 0] += reg_cen_x
            edge_coords[:, 1] += reg_cen_y

            return edge_coords

        if deg_central_coord.unit != deg:
            raise UnitConversionError("The central coordinate must be in degrees for this function.")

        # If no custom regions array was passed, we use the internal array of interloper regions
        if regions_to_search is None:
            regions_to_search = self._interloper_regions.copy()

        inner_radius = self.convert_radius(inner_radius, 'deg')
        outer_radius = self.convert_radius(outer_radius, 'deg')

        # Then we can check to make sure that the outer radius is larger than the inner radius
        if inner_radius >= outer_radius:
            raise ValueError("inner_radius cannot be larger than or equal to outer_radius".format(s=self.name))

        # I think my last attempt at this type of function was made really slow by something to with the regions
        #  module, so I'm going to try and move away from that here
        # This is horrible I know, but it basically generates points on the boundary of each interloper, and then
        #  calculates their distance from the central coordinate. So you end up with an Nx30 (because 30 is
        #  how many points I generate) and N is the number of potential interlopers
        int_dists = np.array([np.sqrt(np.sum((perimeter_points(r.center.ra.value, r.center.dec.value,
                                                               r.width.to('deg').value/2,
                                                               r.height.to('deg').value/2, r.angle.to('rad').value)
                                              - deg_central_coord.value) ** 2, axis=1))
                              for r in regions_to_search])

        # Finds which of the possible interlopers have any part of their boundary within the annulus in consideration
        int_within = np.unique(np.where((int_dists < outer_radius.value) & (int_dists > inner_radius.value))[0])

        return np.array(regions_to_search)[int_within]

    @staticmethod
    def _interloper_sas_string(reg: EllipseSkyRegion, im: Image, output_unit: Union[UnitBase, str]) -> str:
        """
        Converts ellipse sky regions into SAS region strings for use in SAS tasks.

        :param EllipseSkyRegion reg: The interloper region to generate a SAS string for
        :param Image im: The XGA image to use for coordinate conversion.
        :param UnitBase/str output_unit: The output unit for this SAS region, either xmm_sky or xmm_det.
        :return: The SAS string region for this interloper
        :rtype: str
        """

        if output_unit == xmm_det:
            c_str = "DETX,DETY"
            raise NotImplementedError("This coordinate system is not yet supported, and isn't a priority. Please "
                                      "submit an issue on https://github.com/DavidT3/XGA/issues if you particularly "
                                      "want this.")
        elif output_unit == xmm_sky:
            c_str = "X,Y"
        else:
            raise NotImplementedError("Only detector and sky coordinates are currently "
                                      "supported for generating SAS region strings.")

        cen = Quantity([reg.center.ra.value, reg.center.dec.value], 'deg')
        sky_to_deg = sky_deg_scale(im, cen).value
        conv_cen = im.coord_conv(cen, output_unit)
        # Have to divide the width by two, I need to know the half-width for SAS regions, then convert
        #  from degrees to XMM sky coordinates using the factor we calculated in the main function
        w = (reg.width.to('deg').value / 2 / sky_to_deg).round(4)
        # We do the same for the height
        h = (reg.height.to('deg').value / 2 / sky_to_deg).round(4)
        if w == h:
            shape_str = "(({t}) IN circle({cx},{cy},{r}))"
            shape_str = shape_str.format(t=c_str, cx=conv_cen[0].round(4).value, cy=conv_cen[1].round(4).value, r=h)
        else:
            # The rotation angle from the region object is in degrees already
            shape_str = "(({t}) IN ellipse({cx},{cy},{w},{h},{rot}))".format(t=c_str, cx=conv_cen[0].round(4).value,
                                                                             cy=conv_cen[1].round(4).value, w=w, h=h,
                                                                             rot=reg.angle.round(4).value)
        return shape_str

    def get_annular_sas_region(self, inner_radius: Quantity, outer_radius: Quantity, obs_id: str, inst: str,
                               output_unit: Union[UnitBase, str] = xmm_sky, rot_angle: Quantity = Quantity(0, 'deg'),
                               interloper_regions: np.ndarray = None, central_coord: Quantity = None) -> str:
        """
        A method to generate a SAS region string for an arbitrary circular or elliptical annular region, with
        interloper sources removed.

        :param Quantity inner_radius: The inner radius/radii of the region you wish to generate in SAS, if the
            quantity has multiple elements then an elliptical region will be generated, with the first element
            being the inner radius on the semi-major axis, and the second on the semi-minor axis.
        :param Quantity outer_radius: The inner outer_radius/radii of the region you wish to generate in SAS, if the
            quantity has multiple elements then an elliptical region will be generated, with the first element
            being the outer radius on the semi-major axis, and the second on the semi-minor axis.
        :param str obs_id: The ObsID of the observation you wish to generate the SAS region for.
        :param str inst: The instrument of the observation you to generate the SAS region for.
        :param UnitBase/str output_unit: The output unit for this SAS region, either xmm_sky or xmm_det.
        :param np.ndarray interloper_regions: The interloper regions to remove from the source region,
            default is None, in which case the function will run self.regions_within_radii.
        :param Quantity rot_angle: The rotation angle of the source region, default is zero degrees.
        :param Quantity central_coord: The coordinate on which to centre the source region, default is
            None in which case the function will use the default_coord of the source object.
        :return: A string for use in a SAS routine that describes the source region, and the regions
            to cut out of it.
        :rtype: str
        """

        if central_coord is None:
            central_coord = self._default_coord

        # These checks/conversions are already done by the evselect_spectrum command, but I don't
        #  mind doing them again
        inner_radius = self.convert_radius(inner_radius, 'deg')
        outer_radius = self.convert_radius(outer_radius, 'deg')

        # Then we can check to make sure that the outer radius is larger than the inner radius
        if inner_radius.isscalar and inner_radius >= outer_radius:
            raise ValueError("A SAS circular region for {s} cannot have an inner_radius larger than or equal to its "
                             "outer_radius".format(s=self.name))
        elif not inner_radius.isscalar and (inner_radius[0] >= outer_radius[0] or inner_radius[1] >= outer_radius[1]):
            raise ValueError("A SAS elliptical region for {s} cannot have inner radii larger than or equal to its "
                             "outer radii".format(s=self.name))

        if output_unit == xmm_det:
            c_str = "DETX,DETY"
            raise NotImplementedError("This coordinate system is not yet supported, and isn't a priority. Please "
                                      "submit an issue on https://github.com/DavidT3/XGA/issues if you particularly "
                                      "want this.")
        elif output_unit == xmm_sky:
            c_str = "X,Y"
        else:
            raise NotImplementedError("Only detector and sky coordinates are currently "
                                      "supported for generating SAS region strings.")

        # We need a matching image to perform the coordinate conversion we require
        rel_im = self.get_products("image", obs_id, inst)[0]
        # We can set our own offset value when we call this function, but I don't think I need to
        sky_to_deg = sky_deg_scale(rel_im, central_coord).value

        # We need our chosen central coordinates in the right units of course
        xmm_central_coord = rel_im.coord_conv(central_coord, output_unit)
        # And just to make sure the central coordinates are in degrees
        deg_central_coord = rel_im.coord_conv(central_coord, deg)

        # If the user doesn't pass any regions, then we have to find them ourselves. I decided to allow this
        #  so that within_radii can just be called once externally for a set of ObsID-instrument combinations,
        #  like in evselect_spectrum for instance.
        if interloper_regions is None and inner_radius.isscalar:
            interloper_regions = self.regions_within_radii(inner_radius, outer_radius, deg_central_coord)
        elif interloper_regions is None and not inner_radius.isscalar:
            interloper_regions = self.regions_within_radii(min(inner_radius), max(outer_radius), deg_central_coord)

        # So now we convert our interloper regions into their SAS equivalents
        sas_interloper = [self._interloper_sas_string(i, rel_im, output_unit) for i in interloper_regions]

        if inner_radius.isscalar and inner_radius.value != 0:
            # And we need to define a SAS string for the actual region of interest
            sas_source_area = "(({t}) IN annulus({cx},{cy},{ri},{ro}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].round(4).value,
                                                     cy=xmm_central_coord[1].round(4).value,
                                                     ri=(inner_radius.value/sky_to_deg).round(4),
                                                     ro=(outer_radius.value/sky_to_deg).round(4))
        # If the inner radius is zero then we write a circle region, because it seems that's a LOT faster in SAS
        elif inner_radius.isscalar and inner_radius.value == 0:
            sas_source_area = "(({t}) IN circle({cx},{cy},{r}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].round(4).value,
                                                     cy=xmm_central_coord[1].round(4).value,
                                                     r=(outer_radius.value/sky_to_deg).round(4))
        elif not inner_radius.isscalar and inner_radius[0].value != 0:
            sas_source_area = "(({t}) IN elliptannulus({cx},{cy},{wi},{hi},{wo},{ho},{rot},{rot}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].round(4).value,
                                                     cy=xmm_central_coord[1].round(4).value,
                                                     wi=(inner_radius[0].value/sky_to_deg).round(4),
                                                     hi=(inner_radius[1].value/sky_to_deg).round(4),
                                                     wo=(outer_radius[0].value/sky_to_deg).round(4),
                                                     ho=(outer_radius[1].value/sky_to_deg).round(4),
                                                     rot=rot_angle.to('deg').round(4).value)
        elif not inner_radius.isscalar and inner_radius[0].value == 0:
            sas_source_area = "(({t}) IN ellipse({cx},{cy},{w},{h},{rot}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].round(4).value,
                                                     cy=xmm_central_coord[1].round(4).value,
                                                     w=(outer_radius[0].value / sky_to_deg).round(4),
                                                     h=(outer_radius[1].value / sky_to_deg).round(4),
                                                     rot=rot_angle.to('deg').round(4).value)

        # Combining the source region with the regions we need to cut out
        if len(sas_interloper) == 0:
            final_src = sas_source_area
        else:
            final_src = sas_source_area + " &&! " + " &&! ".join(sas_interloper)

        return final_src

    @property
    def nH(self) -> Quantity:
        """
        Property getter for neutral hydrogen column attribute.

        :return: Neutral hydrogen column surface density.
        :rtype: Quantity
        """
        return self._nH

    @property
    def redshift(self) -> float:
        """
        Property getter for the redshift of this source object.

        :return: Redshift value
        :rtype: float
        """
        return self._redshift

    @property
    def on_axis_obs_ids(self) -> list:
        """
        This method returns an array of ObsIDs that this source is approximately on axis in.

        :return: ObsIDs for which the source is approximately on axis.
        :rtype: list
        """
        return self._onaxis

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

    @property
    def fitted_models(self) -> dict:
        """
        A property that gets the list of spectral models that have been fit to each set of spectra that were
        generated for this source.

        :return: A dict with keys being storage identifiers for spectra, and values being lists of models fit to
            this spectrum.
        :rtype: dict
        """
        return {s_ident: list(self._fit_results[s_ident].keys()) for s_ident in self._fit_results}

    @property
    def fitted_model_configurations(self) -> dict:
        """
        Property that returns a dictionary with spectrum storage identifiers as top level keys, model names as lower
        level keys, and lists of fit configuration identifiers as values.

        :return: Dictionary with model names as keys, and lists of model configuration identifiers as values.
        :rtype: dict
        """
        return {s_ident: {m: list(self._fit_results[s_ident][m].keys()) for m in self.fitted_models[s_ident]}
                for s_ident in self._fit_results}

    @property
    def fitted_model_failures(self) -> dict:
        """
        Property that returns a dictionary with spectrum storage identifiers as top level keys, model names as lower
        level keys, and lists of fit configuration identifiers that correspond to FAILED fits.

        :return: Dictionary with model names as keys, and lists of model configuration identifiers as values.
        :rtype: dict
        """
        return self._failed_fits

    def add_fit_failure(self, model: str, spec_storage_key: str, fit_conf: str):
        """
        A method that keeps a record of when a model fit to a set of spectra, with particular spectrum generation
        and fit configuration, was not successful. It is helpful to keep track of these in order to avoid wasting
        compute time re-running previously failed fits.

        :param str model: The XSPEC definition of the model used to perform the fit. e.g. constant*tbabs*apec
        :param str spec_storage_key: The storage key of any spectrum that was used in this particular fit. The
            ObsID and instrument used don't matter, as the storage key will be the same and is based off of the
            settings when the spectra were generated.
        :param str fit_conf: In order to be able to store results for different fit configurations (e.g. different
            starting pars, abundance tables, all that), we need to have a key that identifies the configuration. We
            do not expect the user to be adding fit data, so this will be a key generated by the fit function.
        """
        # Make sure that the _failed_fits attribute has the structure we want (top level keys spectrum storage
        #  identifiers, lower level keys model names, values lists of fit configurations
        self._failed_fits.setdefault(spec_storage_key, {}).setdefault(model, [])
        # Then we add the failed fit configuration to the storage structure
        self._failed_fits[spec_storage_key][model].append(fit_conf)

    def add_fit_data(self, model: str, tab_line, lums: dict, spec_storage_key: str, fit_conf: str):
        """
        A method that stores fit results and global information for a set of spectra in a source object.
        Any variable parameters in the fit are stored in an internal dictionary structure, as are any luminosities
        calculated. Other parameters of interest are store in other internal attributes. This probably shouldn't
        ever be used by the user, just other parts of XGA, hence why I've asked for a spec_storage_key to be passed
        in rather than all the spectrum configuration options individually.

        This method should not need to be directly called by the user - only by other XGA functions.

        :param str model: The XSPEC definition of the model used to perform the fit. e.g. constant*tbabs*apec
        :param tab_line: The table line with the fit data.
        :param dict lums: The various luminosities measured during the fit.
        :param str spec_storage_key: The storage key of any spectrum that was used in this particular fit. The
            ObsID and instrument used don't matter, as the storage key will be the same and is based off of the
            settings when the spectra were generated.
        :param str fit_conf: In order to be able to store results for different fit configurations (e.g. different
            starting pars, abundance tables, all that), we need to have a key that identifies the configuration. We
            do not expect the user to be adding fit data, so this will be a key generated by the fit function.
        """
        # Just headers that will always be present in tab_line that are not fit parameters
        not_par = ['MODEL', 'TOTAL_EXPOSURE', 'TOTAL_COUNT_RATE', 'TOTAL_COUNT_RATE_ERR',
                   'NUM_UNLINKED_THAWED_VARS', 'FIT_STATISTIC', 'TEST_STATISTIC', 'DOF']

        # Various global values of interest - setting up the dictionary storage structure
        self._total_exp[spec_storage_key] = float(tab_line["TOTAL_EXPOSURE"])
        self._total_count_rate.setdefault(spec_storage_key, {}).setdefault(model, {})
        self._test_stat.setdefault(spec_storage_key, {}).setdefault(model, {})
        self._fit_stat.setdefault(spec_storage_key, {}).setdefault(model, {})
        self._dof.setdefault(spec_storage_key, {}).setdefault(model, {})
        self._fit_results.setdefault(spec_storage_key, {}).setdefault(model, {})
        self._luminosities.setdefault(spec_storage_key, {}).setdefault(model, {})

        # Now we start actually storing things
        self._total_count_rate[spec_storage_key][model][fit_conf] = [float(tab_line["TOTAL_COUNT_RATE"]),
                                                                     float(tab_line["TOTAL_COUNT_RATE_ERR"])]
        self._test_stat[spec_storage_key][model][fit_conf] = float(tab_line["TEST_STATISTIC"])
        self._fit_stat[spec_storage_key][model][fit_conf] = float(tab_line["FIT_STATISTIC"])
        self._dof[spec_storage_key][model][fit_conf] = float(tab_line["DOF"])

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
        self._fit_results[spec_storage_key][model][fit_conf] = mod_res

        # And now storing the luminosity results
        self._luminosities[spec_storage_key][model][fit_conf] = lums

    def _get_fit_checks(self, spec_storage_key: str, model: str = None, par: str = None,
                        fit_conf: Union[str, dict] = None) -> Tuple[str, str]:
        """
        An internal function to perform input checks and pre-processing for get methods that access fit results, or
        other related information such as fit statistic.

        :param str spec_storage_key: The XGA product storage key of the spectrum for which we're checking
            spectral fit configurations.
        :param str model: The name of the fitted model that you're requesting the results from
            (e.g. constant*tbabs*apec).
        :param str par: The name of the parameter you want a result for.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the spectrum
            fitting function and values being the changed values (only values changed-from-default need be included)
            or a full string representation of the fit configuration that is being requested.
        :return: The model name and fit configuration.
        :rtype: Tuple[str, str]
        """
        from ..xspec.fit import FIT_FUNC_MODEL_NAMES
        from ..xspec.fitconfgen import fit_conf_from_function

        # It is possible to pass a null value for the 'model' parameter, but we'll only accept that if a single model
        #  has been fit to this spectrum - otherwise how are we to know which model they want?
        if spec_storage_key not in self.fitted_models:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with the specified Spectrum object.")
        elif model is None and len(self.fitted_models[spec_storage_key]) != 1:
            av_mods = ", ".join(self.fitted_models[spec_storage_key])
            raise ValueError("Multiple models have been fit to the specified spectrum, so model=None is not "
                             "valid; available models are {a}".format(m=model, a=av_mods))
        elif model is None:
            # In this case there is ONE model fit, and the user didn't pass a model parameter value, so we'll just
            #  automatically select it for them
            model = self.fitted_models[spec_storage_key][0]
        elif model is not None and model not in self.fitted_models[spec_storage_key]:
            av_mods = ", ".join(self.fitted_models[spec_storage_key])
            raise ModelNotAssociatedError("{m} has not been fitted to the specified spectrum; available "
                                          "models are {a}".format(m=model, a=av_mods))

        # Checks the input fit configuration values - if they are completely illegal we throw an error
        if fit_conf is not None and not isinstance(fit_conf, (str, dict)):
            raise TypeError("'fit_conf' must be a string fit configuration key, or a dictionary with "
                            "changed-from-default fit function arguments as keys and changed values as items.")
        # If the input is a dictionary then we need to construct the key, as opposed to it being passed in whole
        #  as a string
        elif isinstance(fit_conf, dict):
            fit_conf = fit_conf_from_function(FIT_FUNC_MODEL_NAMES[model], fit_conf)
        # In this case the user passed no fit configuration key, but there are multiple fit configurations stored here
        elif fit_conf is None and len(self.fitted_model_configurations[spec_storage_key][model]) != 1:
            av_fconfs = ", ".join(self.fitted_model_configurations[spec_storage_key][model])
            raise ValueError("The {m} model has been fit to the specified spectrum with multiple configuration, so "
                             "fit_conf=None is not valid; available fit configurations are "
                             "{a}".format(m=model, a=av_fconfs))
        # However here they passed no fit configuration, and only one has been used for the model, so we're all good
        #  and will select it for them
        elif fit_conf is None and len(self.fitted_model_configurations[spec_storage_key][model]) == 1:
            fit_conf = self.fitted_model_configurations[spec_storage_key][model][0]

        # Check to make sure the requested results actually exist
        if fit_conf not in self._fit_results[spec_storage_key][model]:
            av_fconfs = ", ".join(self.fitted_model_configurations[spec_storage_key][model])
            raise FitConfNotAssociatedError("The {fc} fit configuration has not been used for any {m} fit to the "
                                            "specified spectrum; available fit configurations are "
                                            "{a}".format(fc=fit_conf, m=model, a=av_fconfs))
        elif par is not None and par not in self._fit_results[spec_storage_key][model][fit_conf]:
            av_pars = ", ".join(self._fit_results[spec_storage_key][model][fit_conf].keys())
            raise ParameterNotAssociatedError("{p} was not a free parameter in the {m} fit to the specified spectra; "
                                              "available parameters are {a}".format(p=par, m=model, a=av_pars))

        return model, fit_conf

    def get_results(self, outer_radius: Union[str, Quantity], model: str = None,
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), par: str = None,
                    group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                    fit_conf: Union[str, dict] = None) -> Union[dict, Quantity]:
        """
        Important method that will retrieve fit results from the source object. Either for a specific
        parameter of a given region-model combination, or for all of them. If a specific parameter is requested,
        all matching values from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+). If no parameter is specified, the return will be a dictionary
        of such numpy arrays, with the keys corresponding to parameter names.

        If no model name is supplied, but only one model was fit to the spectrum of interest, then that model
        will be automatically selected - this behavior also applies to the fit configuration (fit_conf) parameter; if
        a model was only fit with one fit configuration then that will be automatically selected.

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
        :param float min_sn: The minimum signal-to-noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal-to-noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the
            spectrum fitting function and values being the changed values (only values changed-from-default need
            be included) or a full string representation of the fit configuration that is being requested.
        :return: The requested result value, and uncertainties.
        :rtype: Union[dict, Quantity]
        """
        # First I want to retrieve the spectra that were fitted to produce the result they're looking for,
        #  because then I can just grab the storage key from one of them
        specs = self.get_spectra(outer_radius, None, None, inner_radius, group_spec, min_counts, min_sn, over_sample)
        # I just take the first spectrum in the list because the storage key will be the same for all of them
        if isinstance(specs, list):
            # This goes through the selected spectra and just finds the one with
            storage_key = specs[0].storage_key
        else:
            storage_key = specs.storage_key

        model, fit_conf = self._get_fit_checks(storage_key, model, par, fit_conf)

        # Read out into variable for readabilities sake
        fit_data = self._fit_results[storage_key][model][fit_conf]
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

    def get_fit_statistic(self, outer_radius: Union[str, Quantity], model: str = None,
                          inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                          min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                          fit_conf: Union[str, dict] = None) -> float:
        """
        Method that will retrieve fit statistic from the specified spectrum object. If no model name is supplied, but
        only one model has been fit to the spectrum of interest, then that model will be automatically selected - this
        behavior also applies to the fit configuration (fit_conf) parameter; if a model was only fit with one fit
        configuration then that will be automatically selected.

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
        :param float min_sn: The minimum signal-to-noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal-to-noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the
            spectrum fitting function and values being the changed values (only values changed-from-default need
            be included) or a full string representation of the fit configuration that is being requested.
        :return: The requested fit statistic.
        :rtype: float
        """
        specs = self.get_spectra(outer_radius, None, None, inner_radius, group_spec, min_counts, min_sn, over_sample)
        # I just take the first spectrum in the list because the storage key will be the same for all of them
        if isinstance(specs, list):
            # This goes through the selected spectra and just finds the one with
            storage_key = specs[0].storage_key
        else:
            storage_key = specs.storage_key

        model, fit_conf = self._get_fit_checks(storage_key, model, None, fit_conf)

        return self._fit_stat[storage_key][model][fit_conf]

    def get_test_statistic(self, outer_radius: Union[str, Quantity], model: str = None,
                           inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                           min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                           fit_conf: Union[str, dict] = None) -> float:
        """
        Method that will retrieve test statistic from the specified spectrum object. If no model name is supplied, but
        only one model has been fit to the spectrum of interest, then that model will be automatically selected - this
        behavior also applies to the fit configuration (fit_conf) parameter; if a model was only fit with one fit
        configuration then that will be automatically selected.

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
        :param float min_sn: The minimum signal-to-noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal-to-noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the
            spectrum fitting function and values being the changed values (only values changed-from-default need
            be included) or a full string representation of the fit configuration that is being requested.
        :return: The requested fit statistic.
        :rtype: float
        """
        specs = self.get_spectra(outer_radius, None, None, inner_radius, group_spec, min_counts, min_sn, over_sample)
        # I just take the first spectrum in the list because the storage key will be the same for all of them
        if isinstance(specs, list):
            # This goes through the selected spectra and just finds the one with
            storage_key = specs[0].storage_key
        else:
            storage_key = specs.storage_key

        model, fit_conf = self._get_fit_checks(storage_key, model, None, fit_conf)

        return self._test_stat[storage_key][model][fit_conf]

    def get_fit_dof(self, outer_radius: Union[str, Quantity], model: str = None,
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                    min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                    fit_conf: Union[str, dict] = None) -> int:
        """
        Method that will retrieve DOF from the specified spectrum object. If no model name is supplied, but
        only one model has been fit to the spectrum of interest, then that model will be automatically selected - this
        behavior also applies to the fit configuration (fit_conf) parameter; if a model was only fit with one fit
        configuration then that will be automatically selected.

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
        :param float min_sn: The minimum signal-to-noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal-to-noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the
            spectrum fitting function and values being the changed values (only values changed-from-default need
            be included) or a full string representation of the fit configuration that is being requested.
        :return: The requested fit statistic.
        :rtype: float
        """
        specs = self.get_spectra(outer_radius, None, None, inner_radius, group_spec, min_counts, min_sn, over_sample)
        # I just take the first spectrum in the list because the storage key will be the same for all of them
        if isinstance(specs, list):
            # This goes through the selected spectra and just finds the one with
            storage_key = specs[0].storage_key
        else:
            storage_key = specs.storage_key

        model, fit_conf = self._get_fit_checks(storage_key, model, None, fit_conf)

        return self._dof[storage_key][model][fit_conf]

    def get_luminosities(self, outer_radius: Union[str, Quantity], model: str,
                         inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = None,
                         hi_en: Quantity = None, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                         over_sample: float = None, fit_conf: Union[str, dict] = None) -> Union[dict, Quantity]:
        """
        Get method for luminosities calculated from model fits to spectra associated with this source.
        Either for given energy limits (that must have been specified when the fit was first performed), or
        for all luminosities associated with that model. Luminosities are returned as a 3 column numpy array;
        the 0th column is the value, the 1st column is the err-, and the 2nd is err+.

        If no model name is supplied, but only one model was fit to the spectrum of interest, then that model
        will be automatically selected - this behavior also applies to the fit configuration (fit_conf) parameter; if
        a model was only fit with one fit configuration then that will be automatically selected.

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
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the
            spectrum fitting function and values being the changed values (only values changed-from-default need
            be included) or a full string representation of the fit configuration that is being requested.
        :return: The requested luminosity value, and uncertainties.
        :rtype: Union[dict, Quantity]
        """
        # First I want to retrieve the spectra that were fitted to produce the result they're looking for,
        #  because then I can just grab the storage key from one of them
        specs = self.get_spectra(outer_radius, None, None, inner_radius, group_spec, min_counts, min_sn, over_sample)
        # I just take the first spectrum in the list because the storage key will be the same for all of them
        if isinstance(specs, list):
            storage_key = specs[0].storage_key
        else:
            storage_key = specs.storage_key

        model, fit_conf = self._get_fit_checks(storage_key, model, None, fit_conf)

        # Checking the input energy limits are valid, and assembles the key to look for lums in those energy
        #  bounds. If the limits are none then so is the energy key
        if lo_en is not None and hi_en is not None and lo_en > hi_en:
            raise ValueError("The low energy limit cannot be greater than the high energy limit")
        elif lo_en is not None and hi_en is not None:
            en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        else:
            en_key = None

        # Checks the energy band actually exists
        if en_key is not None and en_key not in self._luminosities[storage_key][model][fit_conf]:
            av_bands = ", ".join([en.split("_")[-1] + "keV"
                                  for en in self._luminosities[storage_key][model][fit_conf].keys()])
            raise ParameterNotAssociatedError("{l}-{u}keV was not an energy band for the fit with {m}; available "
                                              "energy bands are {b}".format(l=lo_en.to("keV").value,
                                                                            u=hi_en.to("keV").value,
                                                                            m=model, b=av_bands))

        # If no limits specified,the user gets all the luminosities, otherwise they get the one they asked for
        if en_key is None:
            parsed_lums = {}
            for lum_key in self._luminosities[storage_key][model][fit_conf]:
                lum_value = self._luminosities[storage_key][model][fit_conf][lum_key]
                parsed_lum = Quantity([lum.value for lum in lum_value], lum_value[0].unit)
                parsed_lums[lum_key] = parsed_lum
            return parsed_lums
        else:
            lum_value = self._luminosities[storage_key][model][fit_conf][en_key]
            parsed_lum = Quantity([lum.value for lum in lum_value], lum_value[0].unit)
            return parsed_lum

    def convert_radius(self, radius: Quantity, out_unit: Union[Unit, str] = 'deg') -> Quantity:
        """
        A simple method to convert radii between different distance units, it automatically checks whether
        the requested conversion is possible, given available information. For instance it would fail if you
        requested a conversion from arcseconds to a proper distance if no redshift information were available.

        :param Quantity radius: The radius to convert to a new unit.
        :param Unit/str out_unit: The unit to convert the input radius to.
        :return: The converted radius
        :rtype: Quantity
        """
        # If a string representation was passed, we make it an astropy unit
        if isinstance(out_unit, str):
            out_unit = Unit(out_unit)

        if out_unit.is_equivalent('kpc') and self._redshift is None:
            raise UnitConversionError("You cannot convert to this unit without redshift information.")

        if radius.unit.is_equivalent('deg') and out_unit.is_equivalent('deg'):
            out_rad = radius.to(out_unit)
        elif radius.unit.is_equivalent('deg') and out_unit.is_equivalent('kpc'):
            out_rad = ang_to_rad(radius, self._redshift, self._cosmo).to(out_unit)
        elif radius.unit.is_equivalent('kpc') and out_unit.is_equivalent('kpc'):
            out_rad = radius.to(out_unit)
        elif radius.unit.is_equivalent('kpc') and out_unit.is_equivalent('deg'):
            out_rad = rad_to_ang(radius, self._redshift, self._cosmo).to(out_unit)
        else:
            raise UnitConversionError("Cannot understand {} as a distance unit".format(str(out_unit)))

        return out_rad

    def get_radius(self, rad_name: str, out_unit: Union[Unit, str] = 'deg') -> Quantity:
        """
        Allows a radius associated with this source to be retrieved in specified distance units. Note
        that physical distance units such as kiloparsecs may only be used if the source has
        redshift information.

        :param str rad_name: The name of the desired radius, r200 for instance.
        :param Unit/str out_unit: An astropy unit, either a Unit instance or a string
            representation. Default is degrees.
        :return: The desired radius in the desired units.
        :rtype: Quantity
        """

        # In case somebody types in R500 rather than r500 for instance.
        rad_name = rad_name.lower()
        if rad_name not in self._radii:
            raise ValueError("There is no {r} radius associated with this object.".format(r=rad_name))

        out_rad = self.convert_radius(self._radii[rad_name], out_unit)

        return out_rad

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

    def disassociate_obs(self, to_remove: Union[dict, str, list]):
        """
        Method that uses the supplied dictionary to safely remove data from the source. This data will no longer
        be used in any analyses, and would typically be removed because it is of poor quality, or doesn't contribute
        enough to justify its presence.

        :param dict/str/list to_remove: Either a dictionary of observations to remove, (in the style of
            the source.instruments dictionary with the top level keys being ObsIDs, and the lower levels
            being instrument names), a string containing an ObsID, or a list of ObsIDs.
        """
        # Users can pass just an ObsID string, but we then need to convert it to the form
        #  that the rest of the function requires
        if isinstance(to_remove, str):
            to_remove = {to_remove: deepcopy(self.instruments[to_remove])}
        # Here is where they have just passed a list of ObsIDs, and we need to fill in the blanks with the instruments
        #  currently loaded for those ObsIDs
        elif isinstance(to_remove, list):
            to_remove = {o: deepcopy(self.instruments[o]) for o in to_remove}
        # Here deals with when someone might have passed a dictionary where there is a single instrument, and
        #  they haven't put it in a list; e.g. {'0201903501': 'pn'}. This detects instances like that and then
        #  puts the individual instrument in a list as is expected by the rest of the function
        elif isinstance(to_remove, dict) and not all([isinstance(v, list) for v in to_remove.values()]):
            new_to_remove = {}
            for o in to_remove:
                if not isinstance(to_remove[o], list):
                    new_to_remove[o] = [deepcopy(to_remove[o])]
                else:
                    new_to_remove[o] = deepcopy(to_remove[o])

            # I use deepcopy again because there have been issues with this function still pointing to old memory
            #  addresses, so I'm quite paranoid in this bit of code
            to_remove = deepcopy(new_to_remove)

        # We also check to make sure that the data we're being asked to remove actually is associated with the
        #  source. We shall be forgiving if it isn't, and just issue a warning to let the user know that they are
        #  assuming data was here that actually isn't present
        # Iterating through the keys (ObsIDs) in to_remove
        for o in to_remove:
            if o not in self.obs_ids:
                warn("{o} data cannot be removed from {s} as they are not associated with "
                              "it.".format(o=o, s=self.name), stacklevel=2)
            # Check to see whether any of the instruments for o are not actually associated with the source
            elif any([i not in self.instruments[o] for i in to_remove[o]]):
                bad_list = [i for i in to_remove[o] if i not in self.instruments[o]]
                bad_str = "/".join(bad_list)
                warn("{o}-{ib} data cannot be removed from {s} as they are not associated "
                              "with it.".format(o=o, ib=bad_str, s=self.name), stacklevel=2)

        # Sets the attribute that tells us whether any data has been removed
        if not self._disassociated:
            self._disassociated = True

        # We want to store knowledge of what data has been removed, if there hasn't been anything taken away yet
        #  then we can just set it equal to the to_remove dictionary
        if len(self._disassociated_obs) == 0:
            self._disassociated_obs = to_remove
        # Otherwise we have to add the data to the existing dictionary structure
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
                del self._instruments[o][self._instruments[o].index(i)]

            if len(self._instruments[o]) == 0:
                del self._products[o]
                del self._detected[o]
                del self._initial_regions[o]
                del self._initial_region_matches[o]
                del self._regions[o]
                del self._other_regions[o]
                del self._alt_match_regions[o]
                # These are made on demand, so need to check if its actually present first
                if o in self._interloper_masks:
                    del self._interloper_masks[o]
                if self._peaks is not None:
                    del self._peaks[o]

                del self._obs[self._obs.index(o)]
                if o in self._onaxis:
                    del self._onaxis[self._onaxis.index(o)]
                del self._instruments[o]

        if len(self._obs) == 0:
            raise NoValidObservationsError("No observations remain associated with {} after cleaning".format(self.name))

        # We attempt to load in matching XGA products if that was the behaviour set by load_products on init
        if self._load_products:
            self._existing_xga_products(self._load_fits)

    @property
    def luminosity_distance(self) -> Quantity:
        """
        Tells the user the luminosity distance to the source if a redshift was supplied, if not returns None.

        :return: The luminosity distance to the source, calculated using the cosmology associated with this source.
        :rtype: Quantity
        """
        return self._lum_dist

    @property
    def angular_diameter_distance(self) -> Quantity:
        """
        Tells the user the angular diameter distance to the source if a redshift was supplied, if not returns None.

        :return: The angular diameter distance to the source, calculated using the cosmology
            associated with this source.
        :rtype: Quantity
        """
        return self._ang_diam_dist

    @property
    def background_radius_factors(self) -> ndarray:
        """
        The factors by which to multiply outer radius by to get inner and outer radii for background regions.

        :return: An array of the two factors.
        :rtype: ndarray
        """
        return np.array([self._back_inn_factor, self._back_out_factor])

    def obs_check(self, reg_type: str, threshold_fraction: float = 0.5) -> Dict:
        """
        This method uses exposure maps and region masks to determine which ObsID/instrument combinations
        are not contributing to the analysis. It calculates the area intersection of the mask and exposure
        maps, and if (for a given ObsID-Instrument) the ratio of that area to the full area of the region
        calculated is less than the threshold fraction, that ObsID-instrument will be included in the returned
        rejection dictionary.

        :param str reg_type: The region type for which to calculate the area intersection.
        :param float threshold_fraction: Intersection area/ full region area ratios below this value will mean an
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

        area = {o: {} for o in self.obs_ids}
        full_area = {o: {} for o in self.obs_ids}
        for o in self.obs_ids:
            # Exposure maps of the peak finding energy range for this ObsID
            exp_maps = self.get_products("expmap", o, extra_key=extra_key)
            m = self.get_source_mask(reg_type, o, central_coord=self._default_coord)[0]
            full_area[o] = m.sum()

            for ex in exp_maps:
                # Grabs exposure map data, then alters it so anything that isn't zero is a one
                ex_data = ex.data.copy()
                ex_data[ex_data > 0] = 1
                # We do this because it then becomes very easy to calculate the intersection area of the mask
                #  with the XMM chips. Just mask the modified expmap, then sum.
                area[o][ex.instrument] = (ex_data * m).sum()

        if max(list(full_area.values())) == 0:
            # Everything has to be rejected in this case
            reject_dict = deepcopy(self._instruments)
        else:
            reject_dict = {}
            for o in area:
                for i in area[o]:
                    if full_area[o] != 0:
                        frac = (area[o][i] / full_area[o])
                    else:
                        frac = 0
                    if frac <= threshold_fraction and o not in reject_dict:
                        reject_dict[o] = [i]
                    elif frac <= threshold_fraction and o in reject_dict:
                        reject_dict[o].append(i)

        return reject_dict

    # And here I'm adding a bunch of get methods that should mean the user never has to use get_products, for
    #  individual product types. It will also mean that they will never have to figure out extra keys themselves
    #  and I can make lists of 1 product return just as the product without being a breaking change
    def get_spectra(self, outer_radius: Union[str, Quantity], obs_id: str = None, inst: str = None,
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                    min_counts: int = 5, min_sn: float = None,
                    over_sample: float = None) -> Union[Spectrum, List[Spectrum]]:
        """
        A useful method that wraps the get_products function to allow you to easily retrieve XGA Spectrum objects.
        Simply pass the desired ObsID/instrument, and the same settings you used to generate the spectrum
        in evselect_spectrum, and the spectra(um) will be provided to you. If no match is found then a
        NoProductAvailableError will be raised.

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
            'region' is chosen (to use the regions in region files), then any inner radius will be ignored.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all spectra matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all spectra matching the other criteria will be returned.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
            default this is zero arcseconds, resulting in a circular spectrum.
        :param bool group_spec: Was the spectrum you wish to retrieve grouped?
        :param float min_counts: If the spectrum you wish to retrieve was grouped on minimum counts, what was
            the minimum number of counts?
        :param float min_sn: If the spectrum you wish to retrieve was grouped on minimum signal to noise, what was
            the minimum signal to noise.
        :param float over_sample: If the spectrum you wish to retrieve was over sampled, what was the level of
            over sampling used?
        :return: An XGA Spectrum object (if there is an exact match), or a list of XGA Spectrum objects (if there
            were multiple matching products).
        :rtype: Union[Spectrum, List[Spectrum]]
        """
        if isinstance(inner_radius, Quantity):
            inn_rad_num = self.convert_radius(inner_radius, 'deg')
        elif isinstance(inner_radius, str):
            inn_rad_num = self.get_radius(inner_radius, 'deg')
        else:
            raise TypeError("You may only pass a quantity or a string as inner_radius")

        if isinstance(outer_radius, Quantity):
            out_rad_num = self.convert_radius(outer_radius, 'deg')
        elif isinstance(outer_radius, str):
            out_rad_num = self.get_radius(outer_radius, 'deg')
        else:
            raise TypeError("You may only pass a quantity or a string as outer_radius")

        if over_sample is not None:
            over_sample = int(over_sample)
        if min_counts is not None:
            min_counts = int(min_counts)
        if min_sn is not None:
            min_sn = float(min_sn)

        # Sets up the extra part of the storage key name depending on if grouping is enabled
        if group_spec and min_counts is not None:
            extra_name = "_mincnt{}".format(min_counts)
        elif group_spec and min_sn is not None:
            extra_name = "_minsn{}".format(min_sn)
        else:
            extra_name = ''

        # And if it was oversampled during generation then we need to include that as well
        if over_sample is not None:
            extra_name += "_ovsamp{ov}".format(ov=over_sample)

        if outer_radius != 'region':
            # The key under which these spectra will be stored
            spec_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}"
            spec_storage_name = spec_storage_name.format(ra=self.default_coord[0].value,
                                                         dec=self.default_coord[1].value,
                                                         ri=inn_rad_num.value, ro=out_rad_num.value,
                                                         gr=group_spec)
        else:
            spec_storage_name = "region_grp{gr}".format(gr=group_spec)

        # Adds on the extra information about grouping to the storage key
        spec_storage_name += extra_name
        matched_prods = self.get_products('spectrum', obs_id=obs_id, inst=inst, extra_key=spec_storage_name)
        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any spectra matching your input.")

        return matched_prods

    def get_annular_spectra(self, radii: Quantity = None, group_spec: bool = True, min_counts: int = 5,
                            min_sn: float = None, over_sample: float = None, set_id: int = None) -> AnnularSpectra:
        """
        Another useful method that wraps the get_products function, though this one gets you AnnularSpectra.
        Pass the radii used to generate the annuli, and the same settings you used to generate the spectrum
        in spectrum_set, and the AnnularSpectra will be returned (if it exists). If no match is found then
        a NoProductAvailableError will be raised. This method has an additional way of looking for a matching
        spectrum, if the set ID is known then that can be passed by the user and used to find an exact match.

        :param Quantity radii: The annulus boundary radii that were used to generate the annular spectra set
            that you wish to retrieve. By default this is None, which means the method will return annular
            spectra with any radii.
        :param bool group_spec: Was the spectrum set you wish to retrieve grouped?
        :param float min_counts: If the spectrum set you wish to retrieve was grouped on minimum counts, what was
            the minimum number of counts?
        :param float min_sn: If the spectrum set you wish to retrieve was grouped on minimum signal to
            noise, what was the minimum signal to noise.
        :param float over_sample: If the spectrum set you wish to retrieve was over sampled, what was the level of
            over sampling used?
        :param int set_id: The unique identifier of the annular spectrum set. Passing a value for this parameter
            will override any other information that you have given this method.
        :return: An XGA AnnularSpectra object if there is an exact match.
        :rtype: AnnularSpectra
        """
        if group_spec and min_counts is not None:
            extra_name = "_mincnt{}".format(min_counts)
        elif group_spec and min_sn is not None:
            extra_name = "_minsn{}".format(min_sn)
        else:
            extra_name = ''

        # If set_id is passed then we make sure that it is an integer
        if set_id is not None:
            set_id = int(set_id)

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
            matched_prods = self.get_products('combined_spectrum', extra_key=spec_storage_name)
        # But if the user hasn't passed an ID AND the radii are None then we look for partial matches
        elif set_id is None and radii is None:
            matched_prods = [p for p in self.get_products('combined_spectrum')
                             if spec_storage_name[0] in p.storage_key and spec_storage_name[1] in p.storage_key]
        # However if they have passed a setID then this over-rides everything else
        else:
            # With the set ID we fetch ALL annular spectra, then use their set_id property to match against
            #  whatever the user passed in
            matched_prods = [p for p in self.get_products('combined_spectrum') if p.set_ident == set_id]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("No matching AnnularSpectra can be found.")

        return matched_prods

    def _get_phot_prod(self, prod_type: str, obs_id: str = None, inst: str = None, lo_en: Quantity = None,
                       hi_en: Quantity = None, psf_corr: bool = False, psf_model: str = "ELLBETA",
                       psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15) \
            -> Union[Image, ExpMap, RateMap, List[Image], List[ExpMap], List[RateMap]]:
        """
        An internal method which is the basis of the get_images, get_expmaps, and get_ratemaps methods.

        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all images/expmaps/ratemaps matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all images/expmaps/ratemaps matching the other criteria will be returned.
        :param Quantity lo_en: The lower energy limit of the images/expmaps/ratemaps you wish to
            retrieve, the default is None (which will retrieve all images/expmaps/ratemaps regardless of
            energy limit).
        :param Quantity hi_en: The upper energy limit of the images/expmaps/ratemaps you wish to
            retrieve, the default is None (which will retrieve all images/expmaps/ratemaps regardless of
            energy limit).
        :param bool psf_corr: Sets whether you wish to retrieve a PSF corrected images/ratemaps or not.
        :param str psf_model: If the images/ratemaps you want are PSF corrected, this is the PSF model used.
        :param int psf_bins: If the images/ratemaps you want are PSF corrected, this is the number of PSFs per
            side in the PSF grid.
        :param str psf_algo: If the images/ratemaps you want are PSF corrected, this is the algorithm used.
        :param int psf_iter: If the images/ratemaps you want are PSF corrected, this is the number of iterations.
        :return: An XGA Image/RateMap/ExpMap object (if there is an exact match), or a list of XGA
            Image/RateMap/ExpMap objects (if there were multiple matching products).
        :rtype: Union[Image, ExpMap, RateMap, List[Image], List[ExpMap], List[RateMap]]
        """
        # Checks to make sure that an allowed combination of lo_en and hi_en has been passed.
        if all([lo_en is None, hi_en is None]):
            # Sets a flag to tell the rest of the method whether we have energy lims or not
            with_lims = False
            energy_key = None
        elif all([lo_en is not None, hi_en is not None]):
            with_lims = True
            # We have energy limits here so we assemble the key that describes the energy range
            energy_key = "bound_{l}-{h}".format(l=lo_en.to('keV').value, h=hi_en.to('keV').value)
        else:
            raise ValueError("lo_en and hi_en must be either BOTH None or BOTH an Astropy quantity.")

        # If we are looking for a PSF corrected image/ratemap then we assemble the extra key with PSF details
        if psf_corr and prod_type in ["image", "ratemap"]:
            extra_key = "_" + psf_model + "_" + str(psf_bins) + "_" + psf_algo + str(psf_iter)

        if not psf_corr and with_lims:
            # Simplest case, just calling get_products and passing in our information
            matched_prods = self.get_products(prod_type, obs_id, inst, extra_key=energy_key)
        elif not psf_corr and not with_lims:
            broad_matches = self.get_products(prod_type, obs_id, inst)
            matched_prods = [p for p in broad_matches if not p.psf_corrected]
        elif psf_corr and with_lims:
            # Here we need to add the extra key to the energy key
            matched_prods = self.get_products(prod_type, obs_id, inst, extra_key=energy_key + extra_key)
        elif psf_corr and not with_lims:
            # Here we don't know the energy key, so we have to look for partial matches in the get_products return
            broad_matches = self.get_products(prod_type, obs_id, inst, extra_key=None, just_obj=False)
            matched_prods = [p[-1] for p in broad_matches if extra_key in p[-2]]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any {p}s matching your input.".format(p=prod_type))

        return matched_prods

    def get_images(self, obs_id: str = None, inst: str = None, lo_en: Quantity = None, hi_en: Quantity = None,
                   psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                   psf_iter: int = 15) -> Union[Image, List[Image]]:
        """
        A method to retrieve XGA Image objects. This supports the retrieval of both PSF corrected and non-PSF
        corrected images, as well as setting the energy limits of the specific image you would like. A
        NoProductAvailableError error will be raised if no matches are found.

        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all images matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all images matching the other criteria will be returned.
        :param Quantity lo_en: The lower energy limit of the image you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the image you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param bool psf_corr: Sets whether you wish to retrieve a PSF corrected image or not.
        :param str psf_model: If the image you want is PSF corrected, this is the PSF model used.
        :param int psf_bins: If the image you want is PSF corrected, this is the number of PSFs per
            side in the PSF grid.
        :param str psf_algo: If the image you want is PSF corrected, this is the algorithm used.
        :param int psf_iter: If the image you want is PSF corrected, this is the number of iterations.
        :return: An XGA Image object (if there is an exact match), or a list of XGA Image objects (if there
            were multiple matching products).
        :rtype: Union[Image, List[Image]]
        """
        return self._get_phot_prod("image", obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                                   psf_iter)

    def get_expmaps(self, obs_id: str = None, inst: str = None, lo_en: Quantity = None, hi_en: Quantity = None) \
            -> Union[ExpMap, List[ExpMap]]:
        """
        A method to retrieve XGA ExpMap objects. This supports setting the energy limits of the specific
        exposure maps you would like. A NoProductAvailableError error will be raised if no matches are found.

        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all exposure maps matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all exposure maps matching the other criteria will be returned.
        :param Quantity lo_en: The lower energy limit of the exposure maps you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the exposure maps you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :return: An XGA ExpMap object (if there is an exact match), or a list of XGA ExpMap objects (if there
            were multiple matching products).
        :rtype: Union[ExpMap, List[ExpMap]]
        """
        return self._get_phot_prod("expmap", obs_id, inst, lo_en, hi_en, False)

    def get_ratemaps(self, obs_id: str = None, inst: str = None, lo_en: Quantity = None, hi_en: Quantity = None,
                     psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                     psf_iter: int = 15) -> Union[RateMap, List[RateMap]]:
        """
        A method to retrieve XGA RateMap objects. This supports the retrieval of both PSF corrected and non-PSF
        corrected ratemaps, as well as setting the energy limits of the specific ratemap you would like. A
        NoProductAvailableError error will be raised if no matches are found.

        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all ratemaps matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all ratemaps matching the other criteria will be returned.
        :param Quantity lo_en: The lower energy limit of the ratemaps you wish to retrieve, the default
            is None (which will retrieve all ratemaps regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the ratemaps you wish to retrieve, the default
            is None (which will retrieve all ratemaps regardless of energy limit).
        :param bool psf_corr: Sets whether you wish to retrieve a PSF corrected ratemap or not.
        :param str psf_model: If the ratemap you want is PSF corrected, this is the PSF model used.
        :param int psf_bins: If the ratemap you want is PSF corrected, this is the number of PSFs per
            side in the PSF grid.
        :param str psf_algo: If the ratemap you want is PSF corrected, this is the algorithm used.
        :param int psf_iter: If the ratemap you want is PSF corrected, this is the number of iterations.
        :return: An XGA RateMap object (if there is an exact match), or a list of XGA RateMap objects (if there
            were multiple matching products).
        :rtype: Union[RateMap, List[RateMap]]
        """
        return self._get_phot_prod("ratemap", obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                                   psf_iter)

    def _get_lc_prod(self, outer_radius: Union[str, Quantity] = None, obs_id: str = None, inst: str = None,
                     inner_radius: Union[str, Quantity] = None, lo_en: Quantity = None, hi_en: Quantity = None,
                     time_bin_size: Quantity = None) \
            -> Union[LightCurve, List[LightCurve], AggregateLightCurve, List[AggregateLightCurve]]:
        """
        A protected method to retrieve XGA LightCurve objects, the user should never interact with this directly.

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the lightcurve (for instance 'point' would be acceptable for a PointSource, or Quantity(100, 'kpc')).
            Default is None, meaning all lightcurves will be retrieved.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all lightcurves matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all lightcurves matching the other criteria will be returned.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the lightcurve (for instance 'point' would be acceptable for a PointSource, or Quantity(0, 'kpc')).
            Default is None, meaning all lightcurves will be retrieved.
        :param Quantity lo_en: The lower energy limit of the lightcurves you wish to retrieve, the default
            is None (which will retrieve all lightcurves regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the lightcurves you wish to retrieve, the default
            is None (which will retrieve all lightcurves regardless of energy limit).
        :param Quantity time_bin_size: The time bin size used to generate the desired lightcurve. The default value
            is None, in which case all lightcurves matching other criteria will be retrieved.
        :return: An XGA LightCurve object (if there is an exact match), or a list of XGA LightCurve objects (if there
            were multiple matching products), or a single/list of AggregateLightCurve objects.
        :rtype: Union[LightCurve, List[LightCurve], AggregateLightCurve, List[AggregateLightCurve]]
        """
        # Set up search strings (for the product storage keys) for the inner and outer radii here. The default None
        #  value just creates a key that looks for the 'ri' or 'ro' precursor to the value in the key, i.e. it doesn't
        #  do anything - we also make sure that any radii passed by the user are converted properly
        if inner_radius is not None and isinstance(inner_radius, Quantity):
            inn_rad_search = '_ri{}_'.format(self.convert_radius(inner_radius, 'deg').value)
        elif inner_radius is not None and isinstance(inner_radius, str):
            inn_rad_search = '_ri{}_'.format(self.get_radius(inner_radius, 'deg').value)
        elif inner_radius is None:
            inn_rad_search = "_ri"
        else:
            raise TypeError("You may only pass a quantity or a string as inner_radius")

        if outer_radius is not None and isinstance(outer_radius, Quantity):
            out_rad_search = '_ro{}_'.format(self.convert_radius(outer_radius, 'deg').value)
        elif outer_radius is not None and isinstance(outer_radius, str):
            out_rad_search = '_ro{}_'.format(self.get_radius(outer_radius, 'deg').value)
        elif outer_radius is None:
            out_rad_search = "_ro"
        else:
            raise TypeError("You may only pass a quantity or a string as outer_radius")

        # Check to make sure that the time bin size is a legal value, and set up a search string for the time bin
        #  size in order to narrow down the lightcurves to just the ones that the user wants
        if time_bin_size is not None and not time_bin_size.unit.is_equivalent('s'):
            raise UnitConversionError("The 'time_bin_size' argument must be convertible to seconds.")
        elif time_bin_size is None:
            time_bin_search = '_timebin'
        else:
            time_bin_search = '_timebin{}'.format(time_bin_size.to('s').value)

        # Setting up the energy band search string - if one bound is specified then the other has to be as well, I
        #  didn't think it made sense otherwise
        if any([lo_en is not None, hi_en is not None]) and not all([lo_en is not None, hi_en is not None]):
            raise ValueError("The 'lo_en' and 'hi_en' values must either both be None, or both be an energy value.")
        if (lo_en is not None and not lo_en.unit.is_equivalent('keV')) or \
                (hi_en is not None and not hi_en.unit.is_equivalent('keV')):
            raise UnitConversionError("The 'lo_en' and 'hi_en' arguments must be convertible to keV.")
        # If either is None then we know both are because we checked earlier
        elif lo_en is None:
            en_search = 'bound_'
        elif lo_en is not None:
            en_search = 'bound_{l}-{u}'.format(l=lo_en.to('keV').value, u=hi_en.to('keV').value)

        if obs_id == 'combined':
            search_key = 'combined_lightcurve'
        else:
            search_key = 'lightcurve'
        # Grabbing every single lightcurve that matches ObsID and inst passed by the user (remember they could be
        #  None values, indeed they are by default) - we'll then sweep through whatever list is returned and
        #  narrow them down
        all_lcs = self.get_products(search_key, obs_id, inst)
        # It was getting to the point where a list comprehension was less readable than a for loop, particularly
        #  with the pattern logic, so I changed it to this
        matched_prods = []
        for lc in all_lcs:
            if out_rad_search in lc.storage_key and inn_rad_search in lc.storage_key and \
                    time_bin_search in lc.storage_key and en_search in lc.storage_key:
                matched_prods.append(lc)

        if len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any lightcurves matching your input.")

        return matched_prods

    def get_lightcurves(self, outer_radius: Union[str, Quantity] = None, obs_id: str = None, inst: str = None,
                        inner_radius: Union[str, Quantity] = None, lo_en: Quantity = None,
                        hi_en: Quantity = None, time_bin_size: Quantity = None,
                        pattern: Union[dict, str] = 'default') -> Union[LightCurve, List[LightCurve]]:
        """
        A method to retrieve XGA LightCurve objects.

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the lightcurve (for instance 'point' would be acceptable for a PointSource, or Quantity(100, 'kpc')).
            Default is None, meaning all lightcurves will be retrieved.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all lightcurves matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all lightcurves matching the other criteria will be returned.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the lightcurve (for instance 'point' would be acceptable for a PointSource, or Quantity(0, 'kpc')).
            Default is None, meaning all lightcurves will be retrieved.
        :param Quantity lo_en: The lower energy limit of the lightcurves you wish to retrieve, the default
            is None (which will retrieve all lightcurves regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the lightcurves you wish to retrieve, the default
            is None (which will retrieve all lightcurves regardless of energy limit).
        :param Quantity time_bin_size: The time bin size used to generate the desired lightcurve. The default value
            is None, in which case all lightcurves matching other criteria will be retrieved.
        :param dict pattern: Event selection patterns used to create lightcurves of interest. The default value is
            'default' which uses the default values for generating lightcurves for different instruments, or you
            can pass a dictionary with patterns in; e.g. {'pn': '<=4', 'mos': '<=12'}. You can also pass None, which
            means all light curves matching other search terms will be returned.
        :return: An XGA LightCurve object (if there is an exact match), or a list of XGA LightCurve objects (if there
            were multiple matching products).
        :rtype: Union[LightCurve, List[LightCurve]]
        """
        from xga.sas import check_pattern

        # TODO This is XMM specific because of the patterns currently
        # This is where we set up the search string for the patterns specified by the user.
        if pattern is None:
            patt_search = "_pattern"
        elif isinstance(pattern, str):
            pattern = {'pn': '<=4', 'mos': '<=12'}
            patt_search = {inst: "_pattern" + check_pattern(patt)[1] for inst, patt in pattern.items()}
        elif isinstance(pattern, dict):
            if 'mos1' in list(pattern.keys()) or 'mos2' in list(pattern.keys()):
                raise ValueError("Specific MOS instruments do not need to be specified for 'pattern'; i.e. there "
                                 "should be one entry for 'mos'.")
            pattern = {inst: patt.replace(' ', '') for inst, patt in pattern.items()}
            patt_search = {inst: "_pattern" + check_pattern(patt)[1] for inst, patt in pattern.items()}
        else:
            raise TypeError("The 'pattern' argument must be either 'default', or a dictionary where the keys are "
                            "instrument names and values are string patterns.")

        # Just makes the search easier down the line
        if 'mos' in patt_search:
            patt_search.update({'mos1': patt_search['mos'], 'mos2': patt_search['mos']})

        some_lcs = self._get_lc_prod(outer_radius, obs_id, inst, inner_radius, lo_en, hi_en, time_bin_size)
        matched_prods = []
        for lc in some_lcs:
            if isinstance(patt_search, str):
                rel_patt_search = patt_search
            else:
                rel_patt_search = patt_search[lc.instrument]

            if rel_patt_search in lc.storage_key:
                matched_prods.append(lc)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any lightcurves matching your input.")

        return matched_prods

    def get_combined_lightcurves(self, outer_radius: Union[str, Quantity] = None,
                                 inner_radius: Union[str, Quantity] = None, lo_en: Quantity = None,
                                 hi_en: Quantity = None, time_bin_size: Quantity = None,
                                 pattern: Union[dict, str] = "default") \
            -> Union[AggregateLightCurve, List[AggregateLightCurve]]:
        """
        A method to retrieve XGA AggregateLightCurve objects (i.e. lightcurves for this object that were generated at
        the same time and have been packaged together).

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the aggregate lightcurve (for instance 'point' would be acceptable for a PointSource, or
            Quantity(100, 'kpc')). Default is None, meaning all aggregate lightcurves will be retrieved.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the aggregate lightcurve (for instance 'point' would be acceptable for a PointSource, or
            Quantity(0, 'kpc')). Default is None, meaning all aggregate lightcurves will be retrieved.
        :param Quantity lo_en: The lower energy limit of the aggregate lightcurves you wish to retrieve, the default
            is None (which will retrieve all aggregate lightcurves regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the aggregate lightcurves you wish to retrieve, the default
            is None (which will retrieve all aggregate lightcurves regardless of energy limit).
        :param Quantity time_bin_size: The time bin size used to generate the desired aggregate lightcurve. The
            default value is None, in which case all aggregate lightcurves matching other criteria will be retrieved.
        :param dict pattern: Event selection patterns used to create aggregate lightcurves of interest. The default
            is 'default' which uses the default values for generating lightcurves for different instruments, or you
            can pass a dictionary with patterns in; e.g. {'pn': '<=4', 'mos': '<=12'}. You can also pass None, which
            means all aggregate  light curves matching other search terms will be returned.
        :return: An XGA AggregateLightCurve object (if there is an exact match), or a list of XGA AggregateLightCurve
            objects (if there were multiple matching products).
        :rtype: Union[AggregateLightCurve, List[AggregateLightCurve]]
        """
        from xga.sas import check_pattern

        # TODO This is XMM specific because of the patterns currently
        # This is where we set up the search string for the patterns specified by the user.
        if pattern is None:
            patt_search = "pattern"
        elif isinstance(pattern, str):
            pattern = {'pn': '<=4', 'mos': '<=12'}
            patt_search = {inst: "_{i}pattern".format(i=inst) + check_pattern(patt)[1]
                           for inst, patt in pattern.items()}
        elif isinstance(pattern, dict):
            if 'mos1' in list(pattern.keys()) or 'mos2' in list(pattern.keys()):
                raise ValueError("Specific MOS instruments do not need to be specified for 'pattern'; i.e. there "
                                 "should be one entry for 'mos'.")
            pattern = {inst: patt.replace(' ', '') for inst, patt in pattern.items()}
            patt_search = {inst: "_{i}pattern".format(i=inst) + check_pattern(patt)[1]
                           for inst, patt in pattern.items()}
        else:
            raise TypeError("The 'pattern' argument must be either 'default', or a dictionary where the keys are "
                            "instrument names and values are string patterns.")

        # Use the internal function to find the combined light curves, then apply pattern checks after
        some_lcs = self._get_lc_prod(outer_radius, 'combined', None, inner_radius, lo_en, hi_en, time_bin_size)
        matched_prods = []
        for lc in some_lcs:
            if isinstance(patt_search, str):
                rel_patt_search = [patt_search]
            else:
                rel_patt_search = [patt for inst, patt in patt_search.items()]

            if all([rps in lc.storage_key for rps in rel_patt_search]):
                matched_prods.append(lc)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any lightcurves matching your input.")

        return matched_prods

    # The combined photometric products don't really NEED their own get methods, but I figured I would just for
    #  clarity's sake
    def get_combined_images(self, lo_en: Quantity = None, hi_en: Quantity = None, psf_corr: bool = False,
                            psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                            psf_iter: int = 15) -> Union[Image, List[Image]]:
        """
        A method to retrieve combined XGA Image objects, as in those images that have been created by
        merging all available data for this source. This supports the retrieval of both PSF corrected and non-PSF
        corrected images, as well as setting the energy limits of the specific image you would like. A
        NoProductAvailableError error will be raised if no matches are found.

        :param Quantity lo_en: The lower energy limit of the image you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the image you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param bool psf_corr: Sets whether you wish to retrieve a PSF corrected image or not.
        :param str psf_model: If the image you want is PSF corrected, this is the PSF model used.
        :param int psf_bins: If the image you want is PSF corrected, this is the number of PSFs per
            side in the PSF grid.
        :param str psf_algo: If the image you want is PSF corrected, this is the algorithm used.
        :param int psf_iter: If the image you want is PSF corrected, this is the number of iterations.
        :return: An XGA Image object (if there is an exact match), or a list of XGA Image objects (if there
            were multiple matching products).
        :rtype: Union[Image, List[Image]]
        """

        # Checks to make sure that an allowed combination of lo_en and hi_en has been passed.
        if all([lo_en is None, hi_en is None]):
            # Sets a flag to tell the rest of the method whether we have energy lims or not
            with_lims = False
            energy_key = None
        elif all([lo_en is not None, hi_en is not None]):
            with_lims = True
            # We have energy limits here so we assemble the key that describes the energy range
            energy_key = "bound_{l}-{h}".format(l=lo_en.to('keV').value, h=hi_en.to('keV').value)
        else:
            raise ValueError("lo_en and hi_en must be either BOTH None or BOTH an Astropy quantity.")

        # If we are looking for a PSF corrected image then we assemble the extra key with PSF details
        if psf_corr:
            extra_key = "_" + psf_model + "_" + str(psf_bins) + "_" + psf_algo + str(psf_iter)

        if not psf_corr and with_lims:
            # Simplest case, just calling get_products and passing in our information
            matched_prods = self.get_products('combined_image', extra_key=energy_key)
        elif not psf_corr and not with_lims:
            broad_matches = self.get_products("combined_image")
            matched_prods = [p for p in broad_matches if not p.psf_corrected]
        elif psf_corr and with_lims:
            # Here we need to add the extra key to the energy key
            matched_prods = self.get_products('combined_image', extra_key=energy_key + extra_key)
        elif psf_corr and not with_lims:
            # Here we don't know the energy key, so we have to look for partial matches in the get_products return
            broad_matches = self.get_products('combined_image', extra_key=None, just_obj=False)
            matched_prods = [p[-1] for p in broad_matches if extra_key in p[-2]]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined images matching your input.")

        return matched_prods

    def get_combined_expmaps(self, lo_en: Quantity = None, hi_en: Quantity = None) -> Union[ExpMap, List[ExpMap]]:
        """
        A method to retrieve combined XGA ExpMap objects, as in those exposure maps that have been created by
        merging all available data for this source. This supports setting the energy limits of the specific
        exposure maps you would like. A NoProductAvailableError error will be raised if no matches are found.

        :param Quantity lo_en: The lower energy limit of the exposure maps you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the exposure maps you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :return: An XGA ExpMap object (if there is an exact match), or a list of XGA Image objects (if there
            were multiple matching products).
        :rtype: Union[ExpMap, List[ExpMap]]
        """
        if all([lo_en is None, hi_en is None]):
            energy_key = None
        elif all([lo_en is not None, hi_en is not None]):
            energy_key = "bound_{l}-{h}".format(l=lo_en.to('keV').value, h=hi_en.to('keV').value)
        else:
            raise ValueError("lo_en and hi_en must be either BOTH None or BOTH an Astropy quantity.")

        matched_prods = self.get_products('combined_expmap', extra_key=energy_key)
        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined exposure maps matching your input.")

        return matched_prods

    def get_combined_ratemaps(self, lo_en: Quantity = None, hi_en: Quantity = None,  psf_corr: bool = False,
                              psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                              psf_iter: int = 15) -> Union[RateMap, List[RateMap]]:
        """
        A method to retrieve combined XGA RateMap objects, as in those ratemap that have been created by
        merging all available data for this source. This supports the retrieval of both PSF corrected and non-PSF
        corrected ratemaps, as well as setting the energy limits of the specific ratemap you would like. A
        NoProductAvailableError error will be raised if no matches are found.

        :param Quantity lo_en: The lower energy limit of the ratemaps you wish to retrieve, the default
            is None (which will retrieve all ratemaps regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the ratemaps you wish to retrieve, the default
            is None (which will retrieve all ratemaps regardless of energy limit).
        :param bool psf_corr: Sets whether you wish to retrieve a PSF corrected ratemap or not.
        :param str psf_model: If the ratemap you want is PSF corrected, this is the PSF model used.
        :param int psf_bins: If the ratemap you want is PSF corrected, this is the number of PSFs per
            side in the PSF grid.
        :param str psf_algo: If the ratemap you want is PSF corrected, this is the algorithm used.
        :param int psf_iter: If the ratemap you want is PSF corrected, this is the number of iterations.
        :return: An XGA RateMap object (if there is an exact match), or a list of XGA RateMap objects (if there
            were multiple matching products).
        :rtype: Union[RateMap, List[RateMap]]
        """
        # This function is essentially identical to get_images, but I'm going to be lazy and not write
        #  a separate internal function to do both.

        # Checks to make sure that an allowed combination of lo_en and hi_en has been passed.
        if all([lo_en is None, hi_en is None]):
            # Sets a flag to tell the rest of the method whether we have energy lims or not
            with_lims = False
            energy_key = None
        elif all([lo_en is not None, hi_en is not None]):
            with_lims = True
            # We have energy limits here so we assemble the key that describes the energy range
            energy_key = "bound_{l}-{h}".format(l=lo_en.to('keV').value, h=hi_en.to('keV').value)
        else:
            raise ValueError("lo_en and hi_en must be either BOTH None or BOTH an Astropy quantity.")

        # If we are looking for a PSF corrected ratemap then we assemble the extra key with PSF details
        if psf_corr:
            extra_key = "_" + psf_model + "_" + str(psf_bins) + "_" + psf_algo + str(psf_iter)

        if not psf_corr and with_lims:
            # Simplest case, just calling get_products and passing in our information
            matched_prods = self.get_products('combined_ratemap', extra_key=energy_key)
        elif not psf_corr and not with_lims:
            broad_matches = self.get_products("combined_ratemap")
            matched_prods = [p for p in broad_matches if not p.psf_corrected]
        elif psf_corr and with_lims:
            # Here we need to add the extra key to the energy key
            matched_prods = self.get_products('combined_ratemap', extra_key=energy_key + extra_key)
        elif psf_corr and not with_lims:
            # Here we don't know the energy key, so we have to look for partial matches in the get_products return
            broad_matches = self.get_products('combined_ratemap', extra_key=None, just_obj=False)
            matched_prods = [p[-1] for p in broad_matches if extra_key in p[-2]]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined ratemaps matching your input.")

        return matched_prods

    def _get_prof_prod(self, search_key: str, obs_id: str = None, inst: str = None,
                       central_coord: Quantity = None, radii: Quantity = None, annuli_bound_radii: Quantity = None,
                       lo_en: Quantity = None, hi_en: Quantity = None, spec_model: str = None,
                       spec_fit_conf: Union[str, dict] = None) -> Union[BaseProfile1D, List[BaseProfile1D]]:
        """
        The internal method which is the guts of get_profiles and get_combined_profiles. It parses the input and
        searches for full and partial matches in this source's product storage structure.

        :param str search_key: The exact search key which defined profile type, and whether it is combined or not.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all profiles matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all profiles matching the other criteria will be returned.
        :param Quantity central_coord: The central coordinate of the profile you wish to retrieve, the default
            is None which means the method will use the default coordinate of this source.
        :param Quantity radii: The central radii of the profile points, it is not likely that this option will be
            used often as you likely won't know the radial values a priori.
        :param Quantity annuli_bound_radii: The radial boundaries of the annuli of the profile you wish to
            retrieve, the inner and outer radii of the annuli (the centres of which can instead be passed to
            the 'radii' argument). The default is None, in which no matching on annuli radii will be performed.
        :param Quantity lo_en: The lower energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed hi_en must be too.
        :param Quantity hi_en: The higher energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed lo_en must be too.
        :param str spec_model: The name of the spectral model from which the profile originates.
        :param str/dict spec_fit_conf: Only relevant to profiles that were generated from annular spectra, this
            uniquely identifies the configuration (start parameters, abundance tables, settings, etc.) of the
            spectral model fit to measure the properties used in this profile. Either a dictionary with keys being
            the names of parameters passed to the spectrum fitting function and values being the changed values (only
            values changed-from-default need be included) or a full string representation of the fit configuration.
        :return: An XGA profile object (if there is an exact match), or a list of XGA profile objects (if there
            were multiple matching products).
        :rtype: Union[BaseProfile1D, List[BaseProfile1D]]
        """
        # Checking the energy bound input parameters
        if any([lo_en is None, hi_en is None]) and not all([lo_en is None, hi_en is None]):
            raise ValueError("The 'lo_en' and 'hi_en' arguments must both be None, or both be an astropy quantity.")
        elif lo_en is not None and not all([lo_en.unit.is_equivalent('keV'), hi_en.unit.is_equivalent('keV')]):
            raise UnitConversionError("The 'lo_en' and 'hi_en' arguments must be convertible to keV.")
        elif lo_en is not None:
            # Make sure they're in the units we want
            lo_en = lo_en.to('keV')
            hi_en = hi_en.to('keV')

        # If no specific central coordinate has been passed, we fetch the default central coordinate - don't love
        #  this in hindsight, but I won't change it now
        # TODO Consider that this base level get method shouldn't impose a default coordinate?
        if central_coord is None:
            central_coord = self.default_coord

        # Now we convert the input radii to degrees for future comparisons to profile annuli radii - whether those
        #  radii be annular bounds or central radii
        if all([radii is not None, annuli_bound_radii is not None]):
            raise ValueError("Both the 'radii' and 'annuli_bound_radii' arguments are not None - please supply one"
                             " or the other.")
        elif radii is not None:
            radii = self.convert_radius(radii, 'deg')
        elif annuli_bound_radii is not None:
            annuli_bound_radii = self.convert_radius(annuli_bound_radii, 'deg')

        broad_prods = self.get_products(search_key, obs_id, inst, just_obj=True)
        matched_prods = []
        for p in broad_prods:
            p: BaseProfile1D
            # We can now start checking any supplied matching criteria against the current profile - if we got
            #  given energy bounds then we compare them to the profile's bounds. If they don't match then we
            #  trigger a continue statement
            if lo_en is not None and (p.energy_bounds[0] != lo_en or p.energy_bounds[1] != hi_en):
                continue

            # Nice simple one, do the coordinates supplied to the get method match the central coordinates of
            #  the current profile - if they don't then we aren't interested
            if (p.centre != central_coord).any():
                continue

            # If we received information on the inner and outer radii of the annular bins, we'll
            #  check it against the current profile
            if (annuli_bound_radii is not None and
                    (len(annuli_bound_radii) != len(p.annulus_bounds) or
                     (self.convert_radius(p.annulus_bounds, 'deg') != annuli_bound_radii).any())):
                continue

            # The same as above, but for central radii
            if (radii is not None and (len(radii) != len(p.radii) or
                                       (self.convert_radius(p.radii, 'deg') != radii).any())):
                continue

            # If we get here, then the current profile matches our search criteria
            matched_prods.append(p)

        # At this point, we might have to impose more checks on the keys of the identified products - if the
        #  user has passed information that indicates the profile originated from an annular spectrum, then the
        #  profile key will have an additional component that identifies the spectral model and fit configuration
        #  that it originates from.
        if spec_model is not None:
            matched_prods = [p for p in matched_prods if p.spec_model == spec_model]

        # Then the fit configuration
        if spec_fit_conf is not None:
            from ..xspec.fit import PROF_FIT_FUNC_MODEL_NAMES
            from ..xspec.fitconfgen import fit_conf_from_function

            # At this point we've already applied any constraints on the spectral model name, so we just
            #  cycle through the profiles, and see if any of their stored spectral fit configuration match the
            #  fit configuration that was passed to this method. We give the passed spec_fit_conf to the
            #  fit_conf_from_function function to ensure that fit configuration dictionaries are converted
            #  to full fit configuration keys
            new_matched_prods = []
            for p in matched_prods:
                try:
                    cur_gen_fit_conf = fit_conf_from_function(PROF_FIT_FUNC_MODEL_NAMES[p.spec_model], spec_fit_conf)
                    if cur_gen_fit_conf == p.spec_fit_conf:
                        new_matched_prods.append(p)

                # If there is a KeyError, that means that the passed spec_fit_conf isn't compatible with the
                #  model of the current profile, so we skip right on by
                except KeyError as err:
                    print(err)
                    pass
            matched_prods = new_matched_prods

        return matched_prods

    def get_profiles(self, profile_type: str, obs_id: str = None, inst: str = None, central_coord: Quantity = None,
                     radii: Quantity = None, annuli_bound_radii: Quantity = None, lo_en: Quantity = None,
                     hi_en: Quantity = None, spec_model: str = None,
                     spec_fit_conf: Union[str, dict] = None) -> Union[BaseProfile1D, List[BaseProfile1D]]:
        """
        This is the generic get method for XGA profile objects stored in this source. You still must remember
        the profile type value to use it, but once entered it will return a list of all matching profiles (or a
        single object if only one match is found).

        :param str profile_type: The string profile type of the profile(s) you wish to retrieve.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied. The default is None,
            which means all profiles matching the other criteria will be returned.
        :param str inst: Optionally, a specific instrument to search for can be supplied. The default is None,
            which means all profiles matching the other criteria will be returned.
        :param Quantity central_coord: The central coordinate of the profile you wish to retrieve, the default
            is None which means the method will use the default coordinate of this source.
        :param Quantity radii: The central radii of the profile points, it is not likely that this option will be
            used often as you likely won't know the radial values a priori.
        :param Quantity annuli_bound_radii: The radial boundaries of the annuli of the profile you wish to
            retrieve, the inner and outer radii of the annuli (the centres of which can instead be passed to
            the 'radii' argument). The default is None, in which no matching on annuli radii will be performed.
        :param Quantity lo_en: The lower energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed hi_en must be too.
        :param Quantity hi_en: The higher energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed lo_en must be too.
        :param str spec_model: The name of the spectral model from which the profile originates.
        :param str/dict spec_fit_conf: Only relevant to profiles that were generated from annular spectra, this
            uniquely identifies the configuration (start parameters, abundance tables, settings, etc.) of the
            spectral model fit to measure the properties used in this profile. Either a dictionary with keys being
            the names of parameters passed to the spectrum fitting function and values being the changed values (only
            values changed-from-default need be included) or a full string representation of the fit configuration.
        :return: An XGA profile object (if there is an exact match), or a list of XGA profile objects (if there
            were multiple matching products).
        :rtype: Union[BaseProfile1D, List[BaseProfile1D]]
        """
        if "profile" in profile_type:
            warn("The profile_type you passed contains the word 'profile', which is appended onto "
                 "a profile type by XGA, you need to try this again without profile on the end, unless"
                 " you gave a generic profile a type with 'profile' in.", stacklevel=2)

        search_key = profile_type + "_profile"
        if search_key not in ALLOWED_PRODUCTS:
            warn("{} seems to be a custom profile, not an XGA default type. If this is not "
                 "true then you have passed an invalid profile type.".format(search_key), stacklevel=2)

        matched_prods = self._get_prof_prod(search_key, obs_id, inst, central_coord, radii, annuli_bound_radii,
                                            lo_en, hi_en, spec_model, spec_fit_conf)
        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any {p} profiles matching your input.".format(p=profile_type))

        return matched_prods

    def get_combined_profiles(self, profile_type: str, central_coord: Quantity = None, radii: Quantity = None,
                              annuli_bound_radii: Quantity = None, lo_en: Quantity = None, hi_en: Quantity = None,
                              spec_model: str = None, spec_fit_conf: Union[str, dict] = None) \
            -> Union[BaseProfile1D, List[BaseProfile1D]]:
        """
        The generic get method for XGA profiles made using all available data which are stored in this source.
        You still must remember the profile type value to use it, but once entered it will return a list
        of all matching profiles (or a single object if only one match is found).

        :param str profile_type: The string profile type of the profile(s) you wish to retrieve.
        :param Quantity central_coord: The central coordinate of the profile you wish to retrieve, the default
            is None which means the method will use the default coordinate of this source.
        :param Quantity radii: The central radii of the profile points, it is not likely that this option will be
            used often as you likely won't know the radial values a priori.
        :param Quantity annuli_bound_radii: The radial boundaries of the annuli of the profile you wish to
            retrieve, the inner and outer radii of the annuli (the centres of which can instead be passed to
            the 'radii' argument). The default is None, in which no matching on annuli radii will be performed.
        :param Quantity lo_en: The lower energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed hi_en must be too.
        :param Quantity hi_en: The higher energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed lo_en must be too.
        :param str spec_model: The name of the spectral model from which the profile originates.
        :param str/dict spec_fit_conf: Only relevant to profiles that were generated from annular spectra, this
            uniquely identifies the configuration (start parameters, abundance tables, settings, etc.) of the
            spectral model fit to measure the properties used in this profile. Either a dictionary with keys being
            the names of parameters passed to the spectrum fitting function and values being the changed values (only
            values changed-from-default need be included) or a full string representation of the fit configuration.
        :return: An XGA profile object (if there is an exact match), or a list of XGA profile objects (if there
            were multiple matching products).
        :rtype: Union[BaseProfile1D, List[BaseProfile1D]]
        """
        if "profile" in profile_type:
            warn("The profile_type you passed contains the word 'profile', which is appended onto "
                          "a profile type by XGA, you need to try this again without profile on the end, unless"
                          " you gave a generic profile a type with 'profile' in.", stacklevel=2)

        search_key = "combined_" + profile_type + "_profile"

        if search_key not in ALLOWED_PRODUCTS:
            warn("That profile type seems to be a custom profile, not an XGA default type. If this is not "
                          "true then you have passed an invalid profile type.", stacklevel=2)

        matched_prods = self._get_prof_prod(search_key, None, None, central_coord, radii, annuli_bound_radii,
                                            lo_en, hi_en, spec_model, spec_fit_conf)
        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined {p} profiles matching your "
                                          "input.".format(p=profile_type))

        return matched_prods

    @property
    def all_fitted_models(self) -> List[str]:
        """
        This property cycles through all the available fit results, and finds the unique names of XSPEC models
        that have been fitted to this source.

        :return: A list of model names.
        :rtype: List[str]
        """
        models = []
        for s_key in self._fit_results:
            models += list(self._fit_results[s_key].keys())

        return models

    def snr_ranking(self, outer_radius: Union[Quantity, str], lo_en: Quantity = None, hi_en: Quantity = None,
                    allow_negative: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        This method generates a list of ObsID-Instrument pairs, ordered by the signal to noise measured for the
        given region, with element zero being the lowest SNR, and element N being the highest.

        :param Quantity/str outer_radius: The radius that SNR should be calculated within, this can either be a
            named radius such as r500, or an astropy Quantity.
        :param Quantity lo_en: The lower energy bound of the ratemap to use to calculate the SNR. Default is None,
            in which case the lower energy bound for peak finding will be used (default is 0.5keV).
        :param Quantity hi_en: The upper energy bound of the ratemap to use to calculate the SNR. Default is None,
            in which case the upper energy bound for peak finding will be used (default is 2.0keV).
        :param bool allow_negative: Should pixels in the background subtracted count map be allowed to go below
            zero, which results in a lower signal to noise (and can result in a negative signal to noise).
        :return: Two arrays, the first an N by 2 array, with the ObsID, Instrument combinations in order
            of ascending signal to noise, then an array containing the order SNR ratios.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Set up some lists for the ObsID-Instrument combos and their SNRs respectively
        obs_inst = []
        snrs = []
        # We loop through the ObsIDs associated with this source and the instruments associated with those ObsIDs
        for obs_id in self.instruments:
            for inst in self.instruments[obs_id]:
                # Use our handy get_snr method to calculate the SNRs we want, then add that and the
                #  ObsID-inst combo into their respective lists
                snrs.append(self.get_snr(outer_radius, self.default_coord, lo_en, hi_en, obs_id, inst,
                                         allow_negative))
                obs_inst.append([obs_id, inst])

        # Make our storage lists into arrays, easier to work with that way
        obs_inst = np.array(obs_inst)
        snrs = np.array(snrs)

        # We want to order the output by SNR, with the lowest being first and the highest being last, so we
        #  use a numpy function to output the index order needed to re-order our two arrays
        reorder_snrs = np.argsort(snrs)
        # Then we use that to re-order them
        snrs = snrs[reorder_snrs]
        obs_inst = obs_inst[reorder_snrs]

        # And return our ordered dictionaries
        return obs_inst, snrs

    def count_ranking(self, outer_radius: Union[Quantity, str], lo_en: Quantity = None,
                      hi_en: Quantity = None) -> Tuple[np.ndarray, Quantity]:
        """
        This method generates a list of ObsID-Instrument pairs, ordered by the counts measured for the
        given region, with element zero being the lowest counts, and element N being the highest.

        :param Quantity/str outer_radius: The radius that counts should be calculated within, this can either be a
            named radius such as r500, or an astropy Quantity.
        :param Quantity lo_en: The lower energy bound of the ratemap to use to calculate the counts. Default is None,
            in which case the lower energy bound for peak finding will be used (default is 0.5keV).
        :param Quantity hi_en: The upper energy bound of the ratemap to use to calculate the counts. Default is None,
            in which case the upper energy bound for peak finding will be used (default is 2.0keV).
        :return: Two arrays, the first an N by 2 array, with the ObsID, Instrument combinations in order
            of ascending counts, then an array containing the order counts ratios.
        :rtype: Tuple[np.ndarray, Quantity]
        """
        # Set up some lists for the ObsID-Instrument combos and their cnts respectively
        obs_inst = []
        cnts = []
        # We loop through the ObsIDs associated with this source and the instruments associated with those ObsIDs
        for obs_id in self.instruments:
            for inst in self.instruments[obs_id]:
                cnts.append(self.get_counts(outer_radius, self.default_coord, lo_en, hi_en, obs_id, inst))
                obs_inst.append([obs_id, inst])

        # Make our storage lists into arrays, easier to work with that way
        obs_inst = np.array(obs_inst)
        cnts = Quantity(cnts)

        # We want to order the output by counts, with the lowest being first and the highest being last, so we
        #  use a numpy function to output the index order needed to re-order our two arrays
        reorder_cnts = np.argsort(cnts)
        # Then we use that to re-order them
        cnts = cnts[reorder_cnts]
        obs_inst = obs_inst[reorder_cnts]

        # And return our ordered dictionaries'
        return obs_inst, cnts

    def offset(self, off_unit: Union[Unit, str] = "arcmin") -> Quantity:
        """
        This method calculates the Haversine separation between the user supplied ra_dec coordinates, and the peak
        coordinates in the requested off_unit. If there is no peak attribute and error will be thrown, and if no
        peak has been calculated then the result will be 0.

        :param Unit/str off_unit: The unit that the offset should be in.
        :return: The offset between ra_dec and peak, in the requested unit.
        :rtype: Quantity
        """
        # Check that the source has a peak attribute to fetch, otherwise throw error
        if not hasattr(self, 'peak'):
            raise AttributeError("This source does not have a peak attribute, and so an offset cannot be calculated.")

        # Calculate the Haversine distance between ra_dec and peak
        hav_sep = 2 * np.arcsin(np.sqrt((np.sin((self.peak[1] - self.ra_dec[1]) / 2) ** 2) +
                                        np.cos(self.ra_dec[1]) * np.cos(self.peak[1]) *
                                        np.sin((self.peak[0] - self.ra_dec[0]) / 2) ** 2))

        # Convert the separation to the requested unit - this will throw an error if the unit is stupid
        conv_sep = self.convert_radius(hav_sep, off_unit)

        # Return the converted separation
        return conv_sep

    @property
    def peak_lo_en(self) -> Quantity:
        """
        This property returns the lower energy bound of the image used for peak finding.

        :return: A quantity containing the lower energy limit used for peak finding.
        :rtype: Quantity
        """
        return self._peak_lo_en

    @property
    def peak_hi_en(self) -> Quantity:
        """
        This property returns the upper energy bound of the image used for peak finding.

        :return: A quantity containing the upper energy limit used for peak finding.
        :rtype: Quantity
        """
        return self._peak_hi_en

    @property
    def use_peak(self) -> bool:
        """
        This property shows whether a particular XGA source object has been setup to use peak coordinates
        or not. The property is either True, False, or None (if its a BaseSource).

        :return: If the source is set to use peaks, True, otherwise False.
        :rtype: bool
        """
        return self._use_peak

    @property
    def suppressed_warnings(self) -> List[str]:
        """
        A property getter (with no setter) for the warnings that have been suppressed from display to the user as
        the source was declared as a member of a sample.

        :return: The list of suppressed warnings for this source.
        :rtype: List[str]
        """
        if not self._samp_member:
            raise NotSampleMemberError("The source for {n} is not a member of a sample, and as such warnings have "
                                       "not been suppressed.".format(n=self.name))
        else:
            return self._supp_warn

    def info(self):
        """
        Very simple function that just prints a summary of important information related to the source object..
        """
        print("\n-----------------------------------------------------")
        print("Source Name - {}".format(self._name))
        print("User Coordinates - ({0}, {1}) degrees".format(*self._ra_dec))
        if self._use_peak is not None and self._use_peak:
            print("X-ray Peak - ({0}, {1}) degrees".format(*self._peaks["combined"].value))
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
        print("Obs with 1 detection - {}".format(sum([1 for o in self._initial_region_matches if
                                                    self._initial_region_matches[o].sum() == 1])))
        print("Obs with >1 matches - {}".format(sum([1 for o in self._initial_region_matches if
                                                     self._initial_region_matches[o].sum() > 1])))
        # If a combined exposure map exists, we'll use it to give the user an idea of the total exposure
        try:
            ex = self.get_combined_expmaps()
            if isinstance(ex, list):
                ex = ex[0]
            print("Total exposure - {}".format(ex.get_exp(self.ra_dec).to('ks').round(2)))
        except NoProductAvailableError:
            pass
        print("Images associated - {}".format(len(self.get_products("image"))))
        print("Exposure maps associated - {}".format(len(self.get_products("expmap"))))
        print("Combined Ratemaps associated - {}".format(len(self.get_products("combined_ratemap"))))
        print("Spectra associated - {}".format(len(self.get_products("spectrum"))))

        if len(self._fit_results) != 0:
            print("Fitted Models - {}".format(" | ".join(self.all_fitted_models)))

        if self._regions is not None and "custom" in self._radii:
            if self._redshift is not None:
                region_radius = ang_to_rad(self._custom_region_radius, self._redshift, cosmo=self._cosmo)
            else:
                region_radius = self._custom_region_radius.to("deg")
            print("Custom Region Radius - {}".format(region_radius.round(2)))
            if len(self.get_products('combined_image')) != 0:
                print("Custom Region SNR - {}".format(self.get_snr("custom", self._default_coord).round(2)))

        if self._r200 is not None:
            print("R200 - {}".format(self._r200.round(2)))
            if len(self.get_products('combined_image')) != 0:
                print("R200 SNR - {}".format(self.get_snr("r200", self._default_coord).round(2)))
        if self._r500 is not None:
            print("R500 - {}".format(self._r500.round(2)))
            if len(self.get_products('combined_image')) != 0:
                print("R500 SNR - {}".format(self.get_snr("r500", self._default_coord).round(2)))
        if self._r2500 is not None:
            print("R2500 - {}".format(self._r2500.round(2)))
            if len(self.get_products('combined_image')) != 0:
                print("R2500 SNR - {}".format(self.get_snr("r2500", self._default_coord).round(2)))

        # There's probably a neater way of doing the observables - maybe a formatting function?
        if self._richness is not None and self._richness_err is not None \
                and not isinstance(self._richness_err, (list, tuple, ndarray)):
            print("Richness - {0}±{1}".format(self._richness.round(2), self._richness_err.round(2)))
        elif self._richness is not None and self._richness_err is not None \
                and isinstance(self._richness_err, (list, tuple, ndarray)):
            print("Richness - {0} -{1}+{2}".format(self._richness.round(2), self._richness_err[0].round(2),
                                                   self._richness_err[1].round(2)))
        elif self._richness is not None and self._richness_err is None:
            print("Richness - {0}".format(self._richness.round(2)))

        if self._wl_mass is not None and self._wl_mass_err is not None \
                and not isinstance(self._wl_mass_err, (list, tuple, ndarray)):
            print("Weak Lensing Mass - {0}±{1}".format(self._wl_mass, self._richness_err))
        elif self._wl_mass is not None and self._wl_mass_err is not None \
                and isinstance(self._wl_mass_err, (list, tuple, ndarray)):
            print("Weak Lensing Mass - {0} -{1}+{2}".format(self._wl_mass, self._wl_mass_err[0],
                                                            self._wl_mass_err[1]))
        elif self._wl_mass is not None and self._wl_mass_err is None:
            print("Weak Lensing Mass - {0}".format(self._wl_mass))

        if 'get_temperature' in dir(self):
            try:
                tx = self.get_temperature('r500', 'constant*tbabs*apec').value.round(2)
                # Just average the uncertainty for this
                print("R500 Tx - {0}±{1}[keV]".format(tx[0], tx[1:].mean().round(2)))
            except (ModelNotAssociatedError, NoProductAvailableError, ValueError):
                pass

            try:
                lx = self.get_luminosities('r500', 'constant*tbabs*apec', lo_en=Quantity(0.5, 'keV'),
                                           hi_en=Quantity(2.0, 'keV')).to('10^44 erg/s').value.round(2)
                print("R500 0.5-2.0keV Lx - {0}±{1}[e+44 erg/s]".format(lx[0], lx[1:].mean().round(2)))

            except (ModelNotAssociatedError, NoProductAvailableError, ValueError):
                pass
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
    """
    A useful, but very limited, source class. By default this source class will include all ObsIDs present in the
    XGA census, and as such can be used for bulk generation of SAS products. It can also be made to only include
    certain ObsIDs.


    :param List obs: An optional list of ObsIDs to include in the NullSource, otherwise all available ObsIDs
        will be included.
    """
    def __init__(self, obs: List[str] = None):
        """
        The method used to initiate the NullSource class.
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

        # Here we check to make sure that there is at least one valid ObsID remaining
        if len(self._obs) == 0:
            raise NoValidObservationsError("After checks using the XGA census, all ObsIDs associated with this "
                                           "NullSource are considered unusable.")

        # The SAS generation routine might need this information
        self._att_files = {o: xga_conf["XMM_FILES"]["attitude_file"].format(obs_id=o) for o in self._obs}

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
            raise NotAssociatedError("{o} is not associated with {s}".format(o=obs_id, s=self.name))
        else:
            return self._att_files[obs_id]

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
        This method will ONLY store images and exposure maps. Ideally I wouldn't store them as product objects
        at all, but unfortunately exposure maps require an image to be generated. Unlike all other source classes,
        ratemaps will not be generated when matching images and exposure maps are added.

        :param BaseProduct prod_obj: The new product object to be added to the source object.
        """
        if not isinstance(prod_obj, (BaseProduct, BaseAggregateProduct)) and prod_obj is not None:
            raise TypeError("Only product objects can be assigned to sources.")
        elif prod_obj.type != 'image' and prod_obj.type != 'expmap':
            raise TypeError("Only images and exposure maps can be stored in a NullSource, {} objects "
                            "cannot".format(prod_obj.type))

        if prod_obj is not None:
            en_bnds = prod_obj.energy_bounds
            extra_key = "bound_{l}-{u}".format(l=float(en_bnds[0].value), u=float(en_bnds[1].value))
            # As the extra_key variable can be altered if the Image is PSF corrected, I'll also make
            #  this variable with just the energy key
            en_key = "bound_{l}-{u}".format(l=float(en_bnds[0].value), u=float(en_bnds[1].value))

            # Secondary checking step now I've added PSF correction
            if type(prod_obj) == Image and prod_obj.psf_corrected:
                extra_key += "_" + prod_obj.psf_model + "_" + str(prod_obj.psf_bins) + "_" + prod_obj.psf_algorithm + \
                             str(prod_obj.psf_iterations)

            # All information about where to place it in our storage hierarchy can be pulled from the product
            # object itself
            obs_id = prod_obj.obs_id
            inst = prod_obj.instrument
            p_type = prod_obj.type

            # Double check that something is trying to add products from another source to the current one.
            if obs_id != "combined" and obs_id not in self._products:
                raise NotAssociatedError("{o} is not associated with this null source.".format(o=obs_id))
            elif inst != "combined" and inst not in self._products[obs_id]:
                raise NotAssociatedError(
                    "{i} is not associated with XMM observation {o}".format(i=inst, o=obs_id))

            if extra_key not in self._products[obs_id][inst]:
                self._products[obs_id][inst][extra_key] = {}
            self._products[obs_id][inst][extra_key][p_type] = prod_obj

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

        if obs_id not in self._products and obs_id is not None:
            raise NotAssociatedError("{o} is not associated with {s}.".format(o=obs_id, s=self.name))
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
