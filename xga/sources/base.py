#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 03/07/2024, 08:43. Copyright (c) The Contributors

import os
import pickle
from copy import deepcopy
from shutil import move, copyfile
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
from regions import SkyRegion, EllipseSkyRegion, CircleSkyRegion, EllipsePixelRegion, CirclePixelRegion
from regions import read_ds9, PixelRegion

from .. import xga_conf, BLACKLIST
from ..exceptions import NotAssociatedError, NoValidObservationsError, NoProductAvailableError, ModelNotAssociatedError, \
    ParameterNotAssociatedError, \
    NotSampleMemberError, TelescopeNotAssociatedError, PeakConvergenceFailedError
from ..imagetools.misc import pix_deg_scale
from ..imagetools.misc import sky_deg_scale
from ..imagetools.profile import annular_mask
from ..products import PROD_MAP, EventList, BaseProduct, BaseAggregateProduct, Image, Spectrum, ExpMap, \
    RateMap, PSFGrid, BaseProfile1D, AnnularSpectra
from ..products.lightcurve import LightCurve, AggregateLightCurve
from ..sourcetools import separation_match, nh_lookup, ang_to_rad, rad_to_ang
from ..sourcetools.match import _dist_from_source, census_match
from ..sourcetools.misc import coord_to_name
from ..utils import ALLOWED_PRODUCTS, dict_search, xmm_det, xmm_sky, OUTPUT, SRC_REGION_COLOURS, \
    DEFAULT_COSMO, ALLOWED_INST, COMBINED_INSTS, obs_id_test, PRETTY_TELESCOPE_NAMES

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
    :param str/List[str] telescope: The telescope(s) to be used in analyses of the source. If specified here, and
        set up with this installation of XGA, then relevant data (if it exists) will be located and used. The
        default is None, in which case all available telescopes will be used. The user can pass a single name
        (see xga.TELESCOPES for a list of supported telescopes, and xga.USABLE for a list of currently usable
        telescopes), or a list of telescope names.
    :param List[str] sel_null_obs: If a NullSource is being declared, this argument controls the ObsIDs that are to
        be selected, in any other circumstances it has no effect. This argument should either be None (in which
        case all ObsIDs will be eligible) or a list of ObsIDs.
    :param Union[Quantity, dict] search_distance: The distance to search for observations within, the default
        is None in which case standard search distances for different telescopes are used. The user may pass a
        single Quantity to use for all telescopes, a dictionary with keys corresponding to ALL or SOME of the
        telescopes specified by the 'telescope' argument. In the case where only SOME of the telescopes are
        specified in a distance dictionary, the default XGA values will be used for any that are missing.
    :param List[str] sel_null_obs: If a NullSource is being declared, this argument controls the ObsIDs that are to
        be selected, in any other circumstances it has no effect. This argument should either be None (in which
        case all ObsIDs will be eligible) or a list of ObsIDs.
    :param bool null_load_products: Controls whether the image and exposure maps that may be specified in the
        configuration file are loaded. This can cause slow-down with very large NullSources, so by default is
        set to False.
    :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
            radius for the background region. Default is 1.05.
    :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
        radius for the background region. Default is 1.5.
    :param bool load_regions: Whether this BaseSource should load in region files corresponding to the associated
        ObsIDs or not. Default is True.
    :param bool load_spectra: Whether this BaseSource should load in previously generated spectra, which can be
        time-consuming. This adds more nuance to the 'load_products' argument. If 'load_products' is False, then
        this will also be treated as False. If 'load_products' is True and this is False, then images, exposure
        maps, and lightcurves will be loaded, but spectra will not. Default is True.
    """
    def __init__(self, ra: float, dec: float, redshift: float = None, name: str = None,
                 cosmology: Cosmology = DEFAULT_COSMO, load_products: bool = True, load_fits: bool = False,
                 in_sample: bool = False, telescope: Union[str, List[str]] = None,
                 search_distance: Union[Quantity, dict] = None, sel_null_obs: List[str] = None,
                 null_load_products: bool = False, back_inn_rad_factor: float = 1.05, back_out_rad_factor: float = 1.5,
                 load_regions: bool = True, load_spectra: bool = True):
        """
        The init method for the BaseSource, the most general type of XGA source which acts as a superclass for all
        others. The overlord of all XGA classes, the superclass for all source classes. This contains a huge amount of
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
        :param bool in_sample: A boolean argument that tells the source whether it is part of a sample or
            not, setting to True suppresses some warnings so that they can be displayed at the end of the sample
            progress bar. Default is False. User should only set to True to remove warnings.
        :param str/List[str] telescope: The telescope(s) to be used in analyses of the source. If specified here, and
            set up with this installation of XGA, then relevant data (if it exists) will be located and used. The
            default is None, in which case all available telescopes will be used. The user can pass a single name
            (see xga.TELESCOPES for a list of supported telescopes, and xga.USABLE for a list of currently usable
            telescopes), or a list of telescope names.
        :param Union[Quantity, dict] search_distance: The distance to search for observations within, the default
            is None in which case standard search distances for different telescopes are used. The user may pass a
            single Quantity to use for all telescopes, a dictionary with keys corresponding to ALL or SOME of the
            telescopes specified by the 'telescope' argument. In the case where only SOME of the telescopes are
            specified in a distance dictionary, the default XGA values will be used for any that are missing.
        :param List[str] sel_null_obs: If a NullSource is being declared, this argument controls the ObsIDs that are to
            be selected, in any other circumstances it has no effect. This argument should either be None (in which
            case all ObsIDs will be eligible) or a list of ObsIDs.
        :param bool null_load_products: Controls whether the image and exposure maps that may be specified in the
            configuration file are loaded. This can cause slow-down with very large NullSources, so by default is
            set to False.
        :param float back_inn_rad_factor: This factor is multiplied by an analysis region radius, and gives the inner
            radius for the background region. Default is 1.05.
        :param float back_out_rad_factor: This factor is multiplied by an analysis region radius, and gives the outer
            radius for the background region. Default is 1.5.
        :param bool load_regions: Whether this BaseSource should load in region files corresponding to the associated
            ObsIDs or not. Default is True.
        :param bool load_spectra: Whether this BaseSource should load in previously generated spectra, which can be
            time-consuming. This adds more nuance to the 'load_products' argument. If 'load_products' is False, then
            this will also be treated as False. If 'load_products' is True and this is False, then images, exposure
            maps, and lightcurves will be loaded, but spectra will not. Default is True.
        """

        # This checks whether the overall source being declared is a NullSource - if it is that will affect the
        #  behaviour of this init in some significant ways
        if type(self) == NullSource:
            # In this case a NullSource is being declared - won't be that common as specific source classes
            #  are far more useful
            null_source = True
        else:
            # This will be the case in the vast majority of cases
            null_source = False

        # If load_spectra is True and load_products is False, then load_products overrides it
        if load_spectra and not load_products:
            load_spectra = False

        # This tells the source that it is a part of a sample, which we will check to see whether to
        #  suppress (i.e. store in an attribute rather than display) warnings
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

        # ---------------------------------- Identifying relevant data ----------------------------------
        # Firstly, we use the separation match function to find data relevant to this source, searching within a
        #  telescope dependant radius. This function also validates the input that was given for 'telescope'. If
        #  no named telescopes are valid, or no data is found, then an error is thrown.
        # The returns are dictionaries, where the key is the telescope name, and the values are dataframes of
        #  matching ObsIDs (for the first return, matches), or completely blacklisted (observations with SOME
        #  blacklisted instruments aren't included in this) ObsIDs (the second return).
        if not null_source:
            matches, excluded = separation_match(ra, dec, search_distance, telescope)
        else:
            # If we are declaring a NullSource, then the RA and Dec are going to be NaN - and we want to use the
            #  entire census of the telescopes specified by the user
            matches, excluded = census_match(telescope, sel_null_obs)

        # Observations that seem like they can be used are stored in this dictionary - though some of them may be
        #  removed later after some checks. The top level keys will be telescopes
        obs = {}
        # This will store information on the observations that were never included in analysis (so it's distinct from
        #  the disassociated_obs information) - I don't know if this is the solution I'll stick with, but it'll do
        blacklisted_obs = {}
        # Each telescope will have a key in this dictionary, even if there were no observations that were excluded
        for tel in excluded:
            # Add empty dictionary entries for the current telescope, both for the observations and the blacklisted
            #  observations dictionaries
            obs[tel] = {}
            blacklisted_obs[tel] = {}
            for row_ind, row in excluded[tel].iterrows():
                # Just blacklist all instruments for that telescope for that ObsID because for an ObsID to be in
                #  the excluded return from separation_match this has to be the case
                blacklisted_obs[tel][row['ObsID']] = ALLOWED_INST[tel]

            # We can safely use the 'tel' from the keys of 'excluded' to access 'matches' as well, because they
            #  will both have entries regardless of whether there is any data associated with them
            for row_ind, row in matches[tel].iterrows():
                # It is possible to blacklist only SOME of the instruments for a given ObsID, in which case that
                #  ObsID would not appear in the 'excluded' dictionary's dataframes. As such we will check each
                #  ObsID to see whether we need to add SOME of the instruments to the blacklisted observations
                #  attribute that we started further up
                if row['ObsID'] in BLACKLIST[tel]['ObsID'].values:
                    # Extract the exact row of the current telescope's blacklist that is relevant to the current
                    #  ObsID
                    bl_row = BLACKLIST[tel][BLACKLIST[tel]['ObsID'] == row['ObsID']].iloc[0]
                    # Find the instruments that we need to exclude because the 'EXCLUDE_{INST}' column was marked 'T'
                    excl_inst = [col.split('_')[-1].lower() for col in BLACKLIST[tel].columns
                                 if 'EXCLUDE' in col and bl_row[col] == 'T']
                    # Add the partially excluded observation to the blacklist, with the excluded instrument
                    blacklisted_obs[tel][row['ObsID']] = excl_inst

                # This vaguely unpleasant looking list comprehension is actually simple - it locates the columns of
                #  the match dataframe that tell use whether the CENSUS for this telescope says we should use the
                #  different instruments of an observation. If the CENSUS says yes for the current ObsID (row['ObsID']),
                #  and that particular ObsID-instrument has not appeared in the blacklist, then it gets added to the
                #  list which will be included in the 'obs' dictionary under the current telescope and ObsID
                acc_insts = [col.split('_')[-1].lower() for col in matches[tel].columns if 'USE' in col and row[col] and
                             (row['ObsID'] not in blacklisted_obs[tel] or
                              col.split('_')[-1].lower() not in blacklisted_obs[tel][row['ObsID']])]
                # We check that there are actually some instruments that got past the above checks, and if yes then
                #  we add an entry to the obs dictionary
                if len(acc_insts) != 0:
                    obs[tel][row['ObsID']] = acc_insts

        # Perform a check to make sure that there are some observations left after the usable (from CENSUS) and
        #  blacklist checks have been performed
        cur_obs_nums = {tel: len(obs[tel]) for tel in obs}
        if sum(cur_obs_nums.values()) == 0:
            raise NoValidObservationsError("All {t} observations identified for {s} are either unusable or "
                                           "blacklisted.".format(s=self.name, t=', '.join(telescope)))
        # In this case one of the telescopes has no observations that are relevant, so we must remove the key
        #  in 'obs' that refers to it
        elif 0 in cur_obs_nums.values():
            new_obs = {tel: obs[tel] for tel, num in cur_obs_nums.items() if num != 0}
            obs = new_obs

        # If this is a null source declaration then there is absolutely no point loading region files - as such we
        #  pass this to self._initial_products, and it will set all region paths to None, which we know the subsequent
        #  region-based methods of BaseSource can handle
        if null_source:
            # We're overriding the passed value here
            load_regions = False
        else:
            # Load regions has been passed in, so we'll just use whatever has been specified
            null_load_products = True

        # Now we run the method which takes those initially identified observations and goes looking for their
        #  actual event list/image/expmap/region files - those initial products are loaded into XGA products
        self._products, region_dict, self._att_files = self._initial_products(obs, load_regions, null_load_products)

        # Now we do ANOTHER check just like the one above, but on the products attribute, as it is possible that
        #  all those files cannot be found
        cur_obs_nums = {tel: len(self._products[tel]) for tel in self._products}
        if sum(cur_obs_nums.values()) == 0:
            raise NoValidObservationsError("None of the {t} observations identified for this {s} have valid event "
                                           "lists associated with them.".format(s=self.name, t='/'.join(telescope)))
        elif 0 in cur_obs_nums.values():
            # Cut out any mention of a telescope with no loaded files
            new_obs = {tel: obs[tel] for tel, num in cur_obs_nums.items() if num != 0}
            new_prods = {tel: self._products[tel] for tel, num in cur_obs_nums.items() if num != 0}
            new_regs = {tel: region_dict[tel] for tel, num in cur_obs_nums.items() if num != 0}
            new_atts = {tel: self._att_files[tel] for tel, num in cur_obs_nums.items() if num != 0}
            # Then assign the new cut down dictionaries to their original names
            obs = new_obs
            self._products = new_prods
            self._att_files = new_atts
            region_dict = new_regs

        # This is somewhat inelegant, but oh well; it ensures that there are individual instrument entries for each
        #  ObsID in the products dictionary, for cases where the data are shipped with all instruments combined
        for tel in self._products:
            if COMBINED_INSTS[tel]:
                for oi in obs[tel]:
                    self._products[tel][oi].update({i: {} for i in obs[tel][oi]})

        # We now have the final set of initial observations, so we'll store them in an attribute - note that they
        #  may change later as other source classes have different cleaning steps, but any observations will be
        #  removed through the 'disassociation' mechanism
        # NOTE that this attribute has changed considerably since the pre-multi mission version of XGA, as the
        #  instruments attribute has been consolidated into it - plus there is an extra level for telescope names
        self._obs = {t: {o: obs[t][o] if COMBINED_INSTS[t] else [i for i in self._products[t][o]
                                                                 if len(self._products[t][o][i]) != 0]
                         for o in self._products[t]} for t in self._products}

        # Set the blacklisted observation attribute with our dictionary - if all has gone well then this will be a
        #  dictionary of empty dictionaries
        self._blacklisted_obs = blacklisted_obs

        # Pre-multimission XGA had an attribute that described whether a particular observation was 'on-axis' or
        #  not, but that has less relevance for an all-sky survey (which we intend to support), so now we will
        #  store the separation of the centre of each ObsID region from the user defined coordinate
        self._obs_sep = {tel: {o: Quantity(matches[tel][matches[tel]['ObsID'] == o].iloc[0]['dist'], 'deg')
                               for o in self.obs_ids[tel]} for tel in self.obs_ids}
        # -----------------------------------------------------------------------------------------------

        # ---------------------------------- Creating directory structure ----------------------------------
        # This will ensure that the directories in the output directory (default is xga_output, but it can be set
        #  by the user) necessary to store the XGA output products have been created.

        # If the old way (i.e. when XGA was just for XMM) of structuring the output directory has been detected, then
        #  we need to rename and move things to make it compatible - there will eventually be XSPEC and combined
        #  directories at this level again, but that will only be once we start doing combined telescope analyses
        #  and that is a long way off
        if os.path.exists(OUTPUT) and 'combined' in os.listdir(OUTPUT):
            # This is the directory to move the content to - we know that we need to move to a directory named XMM
            #  because that is the only telescope that XGA supported previously
            xmm_dir = OUTPUT + 'xmm/'
            # Make sure that directory has been created - don't check to see if it already exists because there
            #  shouldn't be any way that it could already exist
            os.makedirs(xmm_dir)
            # Move the directories we know exist
            move(OUTPUT + '/combined', xmm_dir + 'combined')
            move(OUTPUT + '/profiles', xmm_dir + 'profiles')
            move(OUTPUT + '/regions', xmm_dir + 'regions')

            # Reads out the current contents of the OUTPUT dir - mostly useful to get a list of ObsID directories
            #  to move
            contents = os.listdir(OUTPUT)
            # I think it is possible for XSPEC not to exist - though honestly I don't remember - either way we
            #  just check to see if it is there
            if 'XSPEC' in contents:
                move(OUTPUT + '/XSPEC', xmm_dir + 'XSPEC')
            # Finally we iterate through the remaining directories, using a regex checker for XMM ObsIDs to see
            #  which match the pattern
            for oi_cand in contents:
                if obs_id_test('xmm', oi_cand):
                    move(OUTPUT + '/' + oi_cand, OUTPUT + 'xmm/{}'.format(oi_cand))

        # This part of the init sets up the directory structure in the output directory specified in the
        #  configuration file - some of this was in xga.utils in a past version of XGA, and a lot of this is now
        #  more resilient to things like inventory files being deleted (though they will hopefully be replaced soon)
        for tel in self.obs_ids:
            # By iterating through self.obs_ids we can create the directories only for those telescopes/ObsIDs that
            #  are relevant to the current source.
            # This is the current telescope's output path
            cur_pth = OUTPUT + tel + '/'
            # Iterate through the relevant ObsIDs, if they don't have a directory then make one. These are for
            #  products generated by XGA to be stored in, they also get an inventory file to store information
            #  about them - largely because some of the informative file names I was using were longer than
            #  256 characters which my OS does not support
            for oi in self.obs_ids[tel]:
                if not os.path.exists(cur_pth + '/' + oi):
                    os.makedirs(cur_pth + '/' + oi)
                # We also make an inventory file for each ObsID inventory, if we can't find a pre-existing one
                if not os.path.exists(cur_pth + '/{}/inventory.csv'.format(oi)):
                    with open(cur_pth + '/{}/inventory.csv'.format(oi), 'w') as inven:
                        inven.writelines(['file_name,obs_id,inst,info_key,src_name,type'])

            # Now we follow the same process but for the profiles, combined, regions, and XSPEC directories
            if not os.path.exists(cur_pth + 'profiles/{}'.format(self.name)):
                os.makedirs(cur_pth + 'profiles/{}'.format(self.name))
            if not os.path.exists(cur_pth + "profiles/{}/inventory.csv".format(self.name)):
                with open(cur_pth + "/profiles/{}/inventory.csv".format(self.name), 'w') as inven:
                    inven.writelines(["file_name,obs_ids,insts,info_key,src_name,type"])

            # There is currently no inventory file for the XSPEC directory
            if not os.path.exists(cur_pth + 'XSPEC/'):
                os.makedirs(cur_pth + 'XSPEC/')

            if not os.path.exists(cur_pth + 'combined/'):
                os.makedirs(cur_pth + 'combined/')
            # And create an inventory file for that directory
            if not os.path.exists(cur_pth + "/combined/inventory.csv"):
                with open(cur_pth + "/combined/inventory.csv", 'w') as inven:
                    inven.writelines(["file_name,obs_ids,insts,info_key,src_name,type"])

            if not os.path.exists(cur_pth + 'regions/'):
                os.makedirs(cur_pth + 'regions/')
            # We now create a directory for custom region files for the source to be stored in
            if not os.path.exists(cur_pth + "/regions/{0}/{0}_custom.reg".format(self.name)):
                os.makedirs(cur_pth + "/regions/{}".format(self.name))
                # And a start to the custom file itself, with white (custom source) as the default colour
                with open(cur_pth + "/regions/{0}/{0}_custom.reg".format(self.name), 'w') as reggo:
                    reggo.write("global color=white\n")
        # --------------------------------------------------------------------------------------------------

        # ---------------------------------- Loading initial region lists ----------------------------------
        # This method takes our vetted (as in we've checked that region files exist) set of region files,
        #  loads them in and starts actually getting the region objects where they need to be for this source
        self._initial_regions, self._initial_region_matches = self._load_regions(region_dict)
        # --------------------------------------------------------------------------------------------------

        # ---------------------------------- Setting up general attributes ---------------------------------
        # The nh_lookup function returns average and weighted average values, so just take the first. If this is a
        #  BaseSource and part of a sample then we're going to avoid the call to nh_lookup, for efficiency
        if in_sample and type(self) == BaseSource:
            self._nH = Quantity(np.NaN, 'cm^-2')
        else:
            self._nH = nh_lookup(self.ra_dec)[0]

        self._redshift = redshift
        self._cosmo = cosmology
        if redshift is not None:
            self._lum_dist = self._cosmo.luminosity_distance(self._redshift)
            self._ang_diam_dist = self._cosmo.angular_diameter_distance(self._redshift)
        else:
            self._lum_dist = None
            self._ang_diam_dist = None

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
        self._interloper_regions = {}
        # This dictionary is used to store the masks generated to remove contaminating sources from XGA images and
        #  ratemaps, in order to perform photometric analyses. We used to generate masks for all ObsIDs during
        #  definition of an Extended or Point source, regardless of whether they would be used or not, but this
        #  behaviour was altered when eROSITA support was added, due to the time-cost of generating masks for
        #  all the detected sources in an eROSITA ObsID.
        self._interloper_masks = {tel: {o: {} for o in self.obs_ids[tel] + ['combined']} for tel in self.telescopes}

        # Set up an attribute where a default central coordinate will live
        self._default_coord = self.ra_dec

        # Initialisation of fit result attributes
        self._fit_results = {tel: {} for tel in self.telescopes}
        self._test_stat = {tel: {} for tel in self.telescopes}
        self._dof = {tel: {} for tel in self.telescopes}
        self._total_count_rate = {tel: {} for tel in self.telescopes}
        self._total_exp = {tel: {} for tel in self.telescopes}
        self._luminosities = {tel: {} for tel in self.telescopes}

        # Initialisation of attributes related to Extended and GalaxyCluster sources
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
        # Here we set up storage structures for peak coordinates - though they may never be filled.
        self._peaks = {tel: {o: {} for o in self.obs_ids[tel]} for tel in self.telescopes}
        self._peaks_near_edge = {tel: {o: {} for o in self.obs_ids[tel]} for tel in self.telescopes}
        for tel in self.telescopes:
            self._peaks[tel]['combined'] = None
            self._peaks_near_edge[tel]['combined'] = None

        self._chosen_peak_cluster = {}
        self._other_peak_clusters = {}

        # Here we deal with the user defined background region, if an annulus surrounding the source is
        #  to be used. First though, we check whether that the outer radius factor is larger than the inner radius
        #  factor.
        if back_out_rad_factor <= back_inn_rad_factor:
            raise ValueError("The 'back_out_rad_factor' argument must be larger than the 'back_inn_rad_factor' "
                             "argument.")
        self._back_inn_factor = back_inn_rad_factor
        self._back_out_factor = back_out_rad_factor

        # These attributes pertain to the cleaning of observations (as in disassociating them from the source if
        #  they don't include enough of the object we care about).
        self._disassociated = False
        self._disassociated_obs = {}
        # ---------------------------------------------------------------------------------------------------

        # If there is an existing XGA output directory, then it makes sense to search for products that XGA
        #  may have already generated and load them in - saves us wasting time making them again.
        # The user does have control over whether this happens or not though.
        # This goes at the end of init to make sure everything necessary has been declared
        if os.path.exists(OUTPUT) and load_products:
            self._existing_xga_products(load_fits, load_spectra)

        # Now going to save load_fits in an attribute, just because if the observation is cleaned we need to
        #  run _existing_xga_products again, same for load_products
        self._load_fits = load_fits
        self._load_products = load_products
        self._load_spectra = load_spectra

    # Firstly, we have all the properties
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

    @property
    def instruments(self) -> Dict:
        """
        A property of a source that details which instruments have valid data for which observations of
        which telescopes.

        :return: A dictionary where the top level keys are telescopes, the lower level keys are ObsIDs, and
            their values are lists of valid instruments.
        :rtype: Dict
        """
        return self._obs

    @property
    def obs_ids(self) -> Dict:
        """
        Property getter for ObsIDs associated with this source that are confirmed to have events files. This
        provides the ObsIDs and the telescopes that they are associated with.

        :return: A dictionary where the keys are telescope names and the values are lists of ObsIDs
        :rtype: Dict
        """
        return {t: list(self._obs[t].keys()) for t in self._obs}

    @property
    def telescopes(self) -> List[str]:
        """
        Property getter for telescopes that are associated with this source.

        :return: A list of telescope names with valid data related to this source.
        :rtype: List[str]
        """
        return list(self._obs.keys())

    @property
    def blacklisted(self) -> Dict:
        """
        A property getter that returns the dictionary of telescope ObsIDs and their instruments which have been
        blacklisted, and thus not considered for use in any analysis of this source.

        :return: The dictionary of blacklisted data, top level keys are .
        :rtype: Dict
        """
        return self._blacklisted_obs

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
    def obs_separations(self) -> dict:
        """
        A property getter that returns a dictionary with the Haversine separations between the user-defined
        source coordinate and the pointing coordinate of each observation, for each telescope.

        :return: A dictionary where the top-level keys are telescope names, the lower-level keys are ObsIDs and
            the values are astropy quantities that represent the separation between the user-defined coordinate
            and the pointing coordinate of the observation.
        :rtype: dict
        """
        return self._obs_sep

    @property
    def cosmo(self) -> Cosmology:
        """
        This method returns whatever cosmology object is associated with this source object.

        :return: An astropy cosmology object specified for this source on initialization.
        :rtype: Cosmology
        """
        return self._cosmo

    @property
    def name(self) -> str:
        """
        The name of the source, either given at initialisation or generated from the user-supplied coordinates.

        :return: The name of the source.
        :rtype: str
        """
        return self._name

    @property
    def num_inst_obs(self) -> dict:
        """
        A property that returns a dictionary of the total number of each instrument for each telescope associated
        with the source. Top level keys are telescope names, lower level keys are instrument names for the telescope,
        and values are an integer count of the total number of observations for that instrument.

        :return: A dictionary of the total number of each instrument for each telescope associated with the source.
            Top level keys are telescope names, lower level keys are instrument names for the telescope, and values
            are an integer count of the total number of observations for that instrument.
        :rtype: dict
        """
        return {tel: {i: len([i for o in self.instruments[tel] if i in self.instruments[tel][o]])
                      for i in ALLOWED_INST[tel]} for tel in self.instruments}

    # @property
    # def num_pn_obs(self) -> int:
    #     """
    #     Getter method that gives the number of PN observations.
    #
    #     :return: Integer number of PN observations associated with this source
    #     :rtype: int
    #     """
    #     return len([o for o in self.obs_ids if 'pn' in self._products[o]])
    #
    # @property
    # def num_mos1_obs(self) -> int:
    #     """
    #     Getter method that gives the number of MOS1 observations.
    #
    #     :return: Integer number of MOS1 observations associated with this source
    #     :rtype: int
    #     """
    #     return len([o for o in self.obs_ids if 'mos1' in self._products[o]])
    #
    # @property
    # def num_mos2_obs(self) -> int:
    #     """
    #     Getter method that gives the number of MOS2 observations.
    #
    #     :return: Integer number of MOS2 observations associated with this source
    #     :rtype: int
    #     """
    #     return len([o for o in self.obs_ids if 'mos2' in self._products[o]])

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
    def fitted_models(self) -> List[str]:
        """
        This property cycles through all the available fit results, and finds the unique names of XSPEC models
        that have been fitted to this source.

        :return: A list of model names.
        :rtype: List[str]
        """
        models = []
        for tel in self.telescopes:
            for s_key in self._fit_results[tel]:
                models += list(self._fit_results[tel][s_key].keys())

        models = list(set(models))

        return models

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
    def background_radius_factors(self) -> ndarray:
        """
        The factors by which to multiply outer radius by to get inner and outer radii for background regions.

        :return: An array of the two factors.
        :rtype: ndarray
        """
        return np.array([self._back_inn_factor, self._back_out_factor])

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

    @property
    def peak(self) -> Quantity:
        """
        A property getter for the COMBINED X-ray peak coordinates - if multiple telescopes are associated with the
        source then this will return the peak that is likely to be the 'best', though that is based on very simplistic
        assumptions of particular telescopes being 'better' than others for this use. All peak coordinates are made
        available through the 'all_peaks' property.

        :return: The fiducial X-ray peak coordinates.
        :rtype: Quantity
        """
        if any([tel not in ['xmm', 'erosita'] for tel in self.telescopes]):
            raise NotImplementedError("This property will not work if there are any telescopes apart from XMM and "
                                      "eROSITA associated - this should never be seen by a non-developer, but if it "
                                      "please get in touch.")

        if len(self.telescopes) > 1:
            warn_text = ("Multiple telescopes are associated with source {n} - we do not yet support combining "
                         "peak information, so the XMM peak is being returned.").format(n=self.name)
            if not self._samp_member:
                warn(warn_text, stacklevel=2)
            else:
                self._supp_warn.append(warn_text)
            peak = self._peaks['xmm']["combined"]

        else:
            peak = self._peaks[self.telescopes[0]]['combined']

        return peak

    @peak.setter
    def peak(self, new_peak: Quantity):
        """
        Allows the user to update the peak value used during analyses manually.

        :param Quantity new_peak: A new RA-DEC peak coordinate, in degrees.
        """
        if any([tel not in ['xmm', 'erosita'] for tel in self.telescopes]):
            raise NotImplementedError("This property will not work if there are any telescopes apart from XMM and "
                                      "eROSITA associated - this should never be seen by a non-developer, but if it "
                                      "please get in touch.")

        if not new_peak.unit.is_equivalent("deg"):
            raise UnitConversionError("The new peak value must be in RA and DEC coordinates")
        elif len(new_peak) != 2:
            raise ValueError("Please pass an astropy Quantity, in units of degrees, with two entries - "
                             "one for RA and one for DEC.")

        if len(self.telescopes) > 1:
            warn_text = ("Multiple telescopes are associated with source {n} - we do not yet support combining "
                         "peak information, so the passed coordinate was set as the XMM peak.").format(n=self.name)
            if not self._samp_member:
                warn(warn_text, stacklevel=2)
            else:
                self._supp_warn.append(warn_text)

            self._peaks['xmm']["combined"] = new_peak.to("deg")

        else:
            self._peaks[self.telescopes[0]]['combined'] = new_peak.to('deg')

    @property
    def all_peaks(self) -> dict:
        """
        A property that provides access to the dictionary of all peak coordinates measured by this source.

        :return: A dictionary where the top-level keys are telescope names, mid level keys are ObsIDs (including a
            'combined' ObsID, which is the primary peak for a telescope), and coordinates as values. Note that the
            individual ObsID entries will be None unless XGA was specifically told to compute their peaks.
        :rtype: dict
        """
        return self._peaks

    # Next up we define the protected methods of the class
    def _initial_products(self, init_obs: dict, load_regions: bool = True, load_products: bool = True) \
            -> Tuple[dict, dict, dict]:
        """
        Assembles the initial dictionary structure of existing data products, for all selected
        telescopes, associated with this source.

        :param dict init_obs: The dictionary (top level keys are telescopes, the lower level keys are ObsIDs with
            values that are lists of instruments) of observations that have initially been identified as being
            relevant to this source.
        :param bool load_regions: This controls whether we read in regions during the course of this method. The
            default is True.
        :param bool load_products: This controls whether products OTHER THAN EVENT LISTS are declared and stored in
            the XGA source product structure.
        :return: A dictionary structure detailing the data products available at initialisation, another
            dictionary containing paths to region files, and another dictionary containing paths to attitude files.
        :rtype: Tuple[dict, dict, dict]
        """

        def read_default_products(en_lims: tuple) -> Tuple[str, dict]:
            """
            This nested function takes pairs of energy limits defined in the config file and runs
            through the default products for each telescope defined in the config file, filling in the
            energy limits and checking if the file paths exist. Those that do exist are read into the relevant
            product object and returned.

            :param tuple en_lims: A tuple containing a lower and upper energy limit to generate file names for,
                the first entry should be the lower limit, the second the upper limit.
            :return: A dictionary key based on the energy limits for the file paths to be stored under, and the
                dictionary of file paths.
            :rtype: tuple[str, dict]
            """
            not_these = ["root_{}_dir".format(tel), "lo_en", "hi_en", "attitude_file", "region_file"] + \
                        [k for k in rel_sec if 'evts' in k]

            # Define the energy limits as astropy quantities, these have originally been retrieved from the
            #  configuration file
            lo = Quantity(float(en_lims[0]), 'keV')
            hi = Quantity(float(en_lims[1]), 'keV')

            # Depending on whether the current telescope provides combined data by default, or on an instrument by
            #  instrument basis, we have to check for the image/expmap entries in the config file differently.
            if not COMBINED_INSTS[tel]:
                # Formats the generic paths given in the config file for this particular obs and energy range
                files = {k.split('_')[1]: v.format(lo_en=en_lims[0], hi_en=en_lims[1], obs_id=obs_id)
                         for k, v in xga_conf["{}_FILES".format(tel.upper())].items() if k not in not_these
                         and inst in k}
            else:
                files = {k.split('_')[1]: v.format(lo_en=en_lims[0], hi_en=en_lims[1], obs_id=obs_id)
                         for k, v in xga_conf["{}_FILES".format(tel.upper())].items() if k not in not_these}

            # It is not necessary to check that the files exist, as this happens when the product classes
            # are instantiated. So whether the file exists or not, an object WILL exist, and you can check if
            # you should use it for analysis using the .usable attribute
            # This looks up the class which corresponds to the key (which is the product ID in this case
            #  e.g. image), then instantiates an object of that class
            prod_objs = {key: PROD_MAP[key](file, obs_id=obs_id, instrument=inst, stdout_str="", stderr_str="",
                                            gen_cmd="", lo_en=lo, hi_en=hi, telescope=tel)
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
        obs_dict = {tel: {o: {} for o in init_obs[tel]} for tel in init_obs}
        # Regions will get their own dictionary, I don't care about keeping the reg_file paths as
        # an attribute because they get read into memory in the init of this class
        reg_dict = {tel: {} for tel in init_obs}
        # Attitude files also get their own dictionary, they won't be read into memory by XGA
        att_dict = {tel: {} for tel in init_obs}

        for tel in init_obs:
            # Grab the dictionary relevant to the current telescope, for readability purposes
            cur_obs = init_obs[tel]
            # Also define the section of the configuration file that we'll be looking at here, again to make
            #  latter parts of this method more readable
            rel_sec = xga_conf["{}_FILES".format(tel.upper())]

            if not COMBINED_INSTS[tel]:
                to_iter = [(o, i) for o, insts in cur_obs.items() for i in insts]
            else:
                to_iter = [(o, tel) for o in cur_obs]

            # We iterate through the pairs of ObsIDs and instrument defined by the initial dictionary of useful data
            #  passed into this method
            for oi in to_iter:
                # This is purely to make the code easier to read
                obs_id = oi[0]
                # This could be an instrument or a telescope name depending on whether the telescope in question has
                #  combined event lists from multiple instruments as standard
                inst_or_tel = oi[1]
                # This is somewhat roundabout, but at points I need the name of the telescope and sometimes I need
                #  the 'combined' flag (i.e. this is what they are stored under).
                if inst_or_tel == tel:
                    inst = 'combined'
                else:
                    inst = inst_or_tel

                # Produces a list of the combinations of upper and lower energy bounds from the config file. If it
                #  isn't in the loop EVERYTHING breaks
                en_comb = zip(rel_sec["lo_en"], rel_sec["hi_en"])

                # First off, we set up the relevant paths to various important files that we're going to
                #  ingest. The first one is the current relevant event list.
                evt_file = rel_sec["clean_{}_evts".format(inst_or_tel)].format(obs_id=obs_id)
                # Then we do the path to the region file specified in the configuration file. Note that later (in
                #  the_load_regions method) we will make a local copy (if the original file exists) and then use
                #  the copy so that any modifications don't harm the original file.
                reg_file = rel_sec["region_file"].format(obs_id=obs_id)

                # Attitude file is a special type of data product, we shouldn't ever deal with it directly so it
                #  doesn't have a product object. It also isn't guaranteed to be a separate thing for all
                #  telescopes, so we do check that the configuration file actually has an entry for it.
                if 'attitude_file' in rel_sec:
                    att_file = rel_sec["attitude_file"].format(obs_id=obs_id)
                else:
                    att_file = None

                if (att_file is not None and os.path.exists(att_file)) or att_file is None:
                    # An instrument subsection of an observation will ONLY be populated if the events file exists
                    # Otherwise nothing can be done with it.
                    evt_list = EventList(evt_file, obs_id=obs_id, instrument=inst, stdout_str="", stderr_str="",
                                         gen_cmd="",  telescope=tel)
                    if not evt_list.usable:
                        continue

                    obs_dict[tel][obs_id][inst] = {"events": evt_list}
                    att_dict[tel][obs_id] = att_file
                    if load_products:
                        # Dictionary updated with derived product names
                        map_ret = map(read_default_products, en_comb)
                        obs_dict[tel][obs_id][inst].update({gen_return[0]: gen_return[1] for gen_return in map_ret})

                    # The path to the region file, as specified in the configuration file, is added to the returned
                    #  dictionary if it exists - we'll make a copy in _load_regions because the BaseSource init
                    #  hasn't created the directory at this stage.
                    # If the load_regions argument is False, then we aren't going to load regions even if the
                    #  file path does exist - this can be useful for NullSource declarations, which don't care
                    #  about regions
                    if load_regions and os.path.exists(reg_file):
                        # Regions dictionary updated with path to local region file, if it exists
                        reg_dict[tel][obs_id] = reg_file
                    else:
                        reg_dict[tel][obs_id] = None

            # Cleans any observations that don't have at least one instrument associated with them
            obs_dict[tel] = {o: v for o, v in obs_dict[tel].items() if len(v) != 0}

        return obs_dict, reg_dict, att_dict

    def _existing_xga_products(self, read_fits: bool, load_spectra: bool):
        """
        A method specifically for searching an existing XGA output directory for relevant files and loading
        them in as XGA products. This will retrieve images, exposure maps, and spectra; then the source product
        structure is updated. The method also finds previous fit results and loads them in.

        :param bool read_fits: Boolean flag that controls whether past fits are read back in or not.
        :param bool read_fits: Boolean flag that controls whether already-generated spectra are loaded in.
        """

        def parse_image_like(file_path: str, exact_type: str, telescope: str,
                             merged: bool = False) -> Union[Image, ExpMap]:
            """
            Very simple little function that takes the path to an XGA generated image-like product (so either an
            image or an exposure map), parses the file path and makes an XGA object of the correct type by using
            the exact_type variable.

            :param str file_path: Absolute path to an XGA-generated data product.
            :param str exact_type: Either 'image' or 'expmap', the type of product that the file_path leads to.
            :param str telescope: The telescope that this product is from.
            :param bool merged: Whether this is a merged file or not.
            :return: An XGA Image or ExpMap object.
            :rtype: Union[Image, ExpMap]
            """
            # Get rid of the absolute part of the path, then split by _ to get the information from the file name
            im_info = file_path.split("/")[-1].split("_")

            if not merged:
                # I know its hard coded but this will always be the case, these are files I generate with XGA.
                cur_obs_id = im_info[0]
                ins = im_info[1]
            else:
                ins = "combined"
                cur_obs_id = "combined"

            en_str = [entry for entry in im_info if "keV" in entry][0]
            lo_en, hi_en = en_str.split("keV")[0].split("-")

            # Have to be astropy quantities before passing them into the Product declaration
            lo_en = Quantity(float(lo_en), "keV")
            hi_en = Quantity(float(hi_en), "keV")

            # Different types of Product objects, the empty strings are because I don't have the stdout, stderr,
            #  or original commands for these objects.
            if exact_type == "image" and "psfcorr" not in file_path:
                final_obj = Image(file_path, cur_obs_id, ins, "", "", "", lo_en, hi_en, telescope=telescope)
            elif exact_type == "image" and "psfcorr" in file_path:
                final_obj = Image(file_path, cur_obs_id, ins, "", "", "", lo_en, hi_en, telescope=telescope)
                final_obj.psf_corrected = True
                final_obj.psf_bins = int([entry for entry in im_info if "bin" in entry][0].split('bin')[0])
                final_obj.psf_iterations = int([entry for entry in im_info if "iter" in
                                                entry][0].split('iter')[0])
                final_obj.psf_model = [entry for entry in im_info if "mod" in entry][0].split("mod")[0]
                final_obj.psf_algorithm = [entry for entry in im_info if "algo" in entry][0].split("algo")[0]
            elif exact_type == "expmap":
                final_obj = ExpMap(file_path, cur_obs_id, ins, "", "", "", lo_en, hi_en, telescope=telescope)
            else:
                raise TypeError("Only image and expmap are allowed.")

            return final_obj

        def parse_lightcurve(inven_entry: pd.Series, telescope: str) -> LightCurve:
            """
            Very simple little function that takes information on an XGA-generated lightcurve (including a path to
            the file), and sets up a LightCurve product that can be added to the product storage structure
            of the source.

            :param pd.Series inven_entry: The inventory entry from which a LightCurve object should be parsed.
            :param str telescope: The telescope to which this lightcurve belongs.
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
                if (rel_obs_id in self.obs_ids[telescope] and
                        (rel_inst in self.instruments[telescope][rel_obs_id] or rel_inst == "combined")):
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
                                           rel_out_rad, rel_lo_en, rel_hi_en, rel_time_bin, rel_patt, is_back_sub=True,
                                           telescope=telescope)

                else:
                    final_obj = None

            else:
                final_obj = None

            return final_obj

        def parse_spectrum(row: pd.Series, combined_obs: bool):
            """
            Takes information from a row of the inventory csv and sets up a Spectrum product that can be added
            to the product storage structure of the source. If the Spectrum is an annular spectrum
            then relevant information such as the set id and the annulus id is returned.

            :param pd.Series row: The inventory dataframe object:
            :param bool combined_obs: 
            """
            if combined_obs:
                obs_id = 'combined'
            else:
                obs_id = row['obs_id']
            if tel == 'erosita' and not combined_obs:
                obs_id = str(obs_id).zfill(6)
            if combined_obs:
                inst = 'combined'
            else:
                inst = row['inst']

            src_name = row['src_name']

            # we will take the info key from the filename, instead of the actual info key in the 
            # inventory. This is because annular spectrum need to be read in, and their info key
            # follows a different format to regular spectrum, so this is more general
            file_name = str(row['file_name'])

            # the naming scheme for combined spectra is slightly different to not combined
            if not combined_obs:
                info_key = "_".join(file_name.split("/")[-1].split("_spec.fits")[0].split("_")[3:])
            else:
                info_key = "_".join(file_name.split("/")[-1].split("_spec.fits")[0].split("_")[2:])
            
            info_key_parts = info_key.split("_")

            central_coord = Quantity([float(info_key_parts[0].strip('ra')), float(info_key_parts[1].strip('dec'))], 'deg')
            r_inner = Quantity(np.array(info_key_parts[2].strip('ri').split('and')).astype(float), 'deg')
            r_outer = Quantity(np.array(info_key_parts[3].strip('ro').split('and')).astype(float), 'deg')
            # Check if there is only one r_inner and r_outer value each, if so its a circle
            #  (otherwise it's an ellipse)
            if len(r_inner) == 1:
                r_inner = r_inner[0]
                r_outer = r_outer[0]

            if 'grpTrue' in info_key_parts:
                grp_ind = info_key_parts.index('grpTrue')
                grouped = True
            else:
                grouped = False
            # mincnt or minsn information will only be in the filename if the spectrum is grouped
            if grouped and 'mincnt' in info_key:
                min_counts = int(info_key_parts[grp_ind+1].split('mincnt')[-1])
                min_sn = None
            elif grouped and 'minsn' in info_key:
                min_sn = float(info_key_parts[grp_ind+1].split('minsn')[-1])
                min_counts = None
            else:
                # We still need to pass the variables to the spectrum definition, even if it isn't
                #  grouped
                min_sn = None
                min_counts = None

            # Only if oversampling was applied will it appear in the filename
            if 'ovsamp' in info_key:
                over_sample = int(info_key_parts[-2].split('ovsamp')[-1])
            else:
                over_sample = None

            if "region" in info_key:
                region = True
            else:
                region = False
            
            # defining a standard product path that I can just suffixes to 
            if combined_obs:
                indent_no = row['file_name'].split('_')[0]
                # the info key actually needs to be used here
                prod_gen_path = cur_d + indent_no + '_' + str(src_name) + '_' + info_key
            else:
                prod_gen_path = cur_d + obs_id + '_' + inst + '_' +  str(src_name) + '_' + info_key
            
            spec = prod_gen_path + '_spec.fits'
            arf = prod_gen_path + '.arf'
            back = prod_gen_path + '_backspec.fits'

            if tel == 'erosita':
                back_rmf = prod_gen_path + '_backspec.rmf'
                back_arf = prod_gen_path + '_backspec.arf'
                rmf = prod_gen_path + '.rmf'

            else:
                if os.path.exists(prod_gen_path + '.rmf'):
                    rmf = prod_gen_path + '.rmf'
                else:
                    rmf = cur_d + obs_id + '_' + inst + '_' +  str(src_name) + '_universal.rmf'
                back_rmf = ''
                back_arf = ''

            # Defining our XGA spectrum instance
            obj = Spectrum(spec, rmf, arf, back, central_coord, r_inner, r_outer, obs_id, inst,
                                grouped, min_counts, min_sn, over_sample, "", "", "", region, back_rmf,
                                back_arf, telescope=tel)
            
            if "ident" in info_key:
                set_id = int(info_key.split('ident')[-1].split('_')[0])
                ann_id = int(info_key.split('ident')[-1].split('_')[1])
            
            else:
                set_id = None
                ann_id = None
                
            return obj, set_id, ann_id

        # Just figure out where we are in the filesystem, we'll make sure to return to this location after all
        #  the changing directory we're about to do
        og_dir = os.getcwd()

        # We iterate through the associated telescopes - the XGA generated products are stored in telescope/ObsID
        #  subdirectories
        for tel in self.telescopes:
            # Read out the allowed instruments for this particular telescopes - this method won't work like this
            #  eventually, but at the moment I'm just converting what is already here
            all_inst = ALLOWED_INST[tel]

            # This is used for spectra that should be part of an AnnularSpectra object
            ann_spec_constituents = {}
            # This is to store whether all components could be loaded in successfully
            ann_spec_usable = {}
            for obs in self.obs_ids[tel]:
                if os.path.exists(OUTPUT + "{t}/{o}".format(t=tel, o=obs)):
                    os.chdir(OUTPUT + "{t}/{o}".format(t=tel, o=obs))
                    cur_d = os.getcwd() + '/'
                    # Loads in the inventory file for this ObsID
                    inven = pd.read_csv("inventory.csv", dtype=str)

                    # Here we read in instruments and exposure maps which are relevant to this source
                    im_lines = inven[(inven['type'] == 'image') | (inven['type'] == 'expmap')]
                    # Instruments is a dictionary with ObsIDs on the top level and then valid instruments on
                    #  the lower level. As such we can be sure here we're only reading in instruments we decided
                    #  are valid
                    # We add the 'combined' entry to account for the possibility of single ObsID combined instrument
                    #  images being created - # TODO decide whether this would be better off in the 'combined' section
                    for i in self.instruments[tel][obs] + ['combined']:
                        # Fetches lines of the inventory which match the current ObsID and instrument
                        rel_ims = im_lines[(im_lines['obs_id'] == obs) & (im_lines['inst'] == i)]
                        for r_ind, r in rel_ims.iterrows():
                            self.update_products(parse_image_like(cur_d+r['file_name'], r['type'], tel),
                                                 update_inv=False)

                    # TODO THIS NEEDS TO BE UPDATED TO SUPPORT MULTI-MISSION XGA
                    # This finds the lines of the inventory that are lightCurve entries
                    lc_lines = inven[inven['type'] == 'lightcurve']
                    for row_ind, row in lc_lines.iterrows():
                        # The parse lightcurve function does check to see if an inventory entry is relevant to this
                        #  source (using the source name), and if the ObsID and instrument are still associated.
                        self.update_products(parse_lightcurve(row, tel), update_inv=False)

                    if load_spectra:
                        
                        spec_lines = inven[inven['type'] == 'spectrum']
                        for row_ind, row in spec_lines.iterrows():
                            obj, set_id, ann_id = parse_spectrum(row, False)
                            if set_id != None:
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
                            

            os.chdir(og_dir)

            # Here we will load in existing xga profile objects
            os.chdir(OUTPUT + "{t}/profiles/{n}".format(t=tel, n=self.name))
            saved_profs = [pf for pf in os.listdir('.') if '.xga' in pf and 'profile' in pf and self.name in pf]
            for pf in saved_profs:
                # TODO CATCH THE ATTRIBUTE ERRORS WHICH COME FROM LOADING OLD STYLE PROFILES WITHOUT TELESCOPE INFO
                try:
                    with open(pf, 'rb') as reado:
                        temp_prof = pickle.load(reado)
                        try:
                            self.update_products(temp_prof, update_inv=False)
                        except NotAssociatedError:
                            pass
                except (EOFError, pickle.UnpicklingError, AttributeError):
                    warn_text = "A profile save ({}) appears to be corrupted, it has not been " \
                                "loaded; you can safely delete this file".format(os.getcwd() + '/' + pf)
                    if not self._samp_member:
                        # If these errors have been raised then I think that the pickle file has been
                        #  broken (see issue #935)
                        warn(warn_text, stacklevel=2)
                    else:
                        self._supp_warn.append(warn_text)
            os.chdir(og_dir)

            # If spectra that should be a part of annular spectra object(s) have been found, then I need to create
            #  those objects and add them to the storage structure
            if len(ann_spec_constituents) != 0:
                for set_id in ann_spec_constituents:
                    if ann_spec_usable[set_id]:
                        ann_spec_obj = AnnularSpectra(ann_spec_constituents[set_id])
                        if self._redshift is not None:
                            # If we know the redshift we will add the radii to the annular spectra in proper
                            #  distance units
                            ann_spec_obj.proper_radii = self.convert_radius(ann_spec_obj.radii, 'kpc')
                        self.update_products(ann_spec_obj, update_inv=False)

            # Here we load in any combined images and exposure maps that may have been generated
            os.chdir(OUTPUT + '{t}/combined'.format(t=tel))
            cur_d = os.getcwd() + '/'
            # This creates a set of observation-instrument strings that describe the current combinations associated
            #  with this source, for testing against to make sure we're loading in combined images/expmaps that
            #  do belong with this source
            src_oi_set = set([o+i for o in self.instruments[tel] for i in self.instruments[tel][o]])

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
                #  the src_oi_set, and if that is the same length as the original src_oi_set then we know that they
                #  match exactly and the product can be loaded
                if len(src_oi_set) == len(test_oi_set) and len(src_oi_set | test_oi_set) == len(src_oi_set):
                    self.update_products(parse_image_like(cur_d+row['file_name'], row['type'], tel, merged=True),
                                         update_inv=False)
            
            # now assigning combined event lists
            rel_inven = inven[inven['type'] == 'events']
            for row_ind, row in rel_inven.iterrows():
                o_split = row['obs_ids'].split('/')
                i_split = row['insts'].split('/')
                # Assemble a set of observations-instrument strings for the current row, to test against the
                #  src_oi_set we assembled earlier
                test_oi_set = set([o+i_split[o_ind] for o_ind, o in enumerate(o_split)])
                # getting a list of obs_ids to parse to the Eventlist object
                obs_list = list(set(o_split))
                # First we make sure the sets are the same length, if they're not then we know before starting that this
                #  row's file can't be okay for us to load in. Then we compute the union between the test_oi_set and
                #  the src_oi_set, and if that is the same length as the original src_oi_set then we know that they
                #  match exactly and the product can be loaded
                if len(src_oi_set) == len(test_oi_set) and len(src_oi_set | test_oi_set) == len(src_oi_set):
                    evt_list = EventList(cur_d+row['file_name'], 'combined', 'combined', '', '', '', tel, obs_list)
                    self.update_products(evt_list, update_inv=False)
            
            if load_spectra:
                rel_inven = inven[inven['type'] == 'spectrum']
                for row_ind, row in rel_inven.iterrows():
                    obj, set_id, ann_id = parse_spectrum(row, True)
                    try:
                        self.update_products(obj, update_inv=False)
                    except NotAssociatedError:
                        pass

            os.chdir(og_dir)

            # Now loading in previous fits
            if os.path.exists(OUTPUT + "{t}/XSPEC/".format(t=tel) + self.name) and read_fits and load_spectra:
                ann_obs_order = {}
                ann_results = {}
                ann_lums = {}
                prev_fits = [OUTPUT + "{t}/XSPEC/".format(t=tel) + self.name + "/" + f
                             for f in os.listdir(OUTPUT + "{t}/XSPEC/".format(t=tel) + self.name)
                             if ".xcm" not in f and ".fits" in f]
                for fit in prev_fits:
                    fit_name = fit.split("/")[-1]
                    fit_info = fit_name.split("_")
                    # Indexing like this because the last two underscores are for the model name, and then the
                    #  telescope name
                    storage_key = "_".join(fit_info[1:-2])
                    # Load in the results table
                    fit_data = FITS(fit)

                    # This bit is largely copied from xspec.py, sorry for my laziness
                    global_results = fit_data["RESULTS"][0]
                    model = global_results["MODEL"].strip(" ")

                    if "_ident" in storage_key:
                        set_id, ann_id = storage_key.split("_ident")[-1].split("_")
                        set_id = int(set_id)
                        ann_id = int(ann_id)
                        if set_id not in ann_results:
                            ann_results[set_id] = {}
                            ann_lums[set_id] = {}
                            ann_obs_order[set_id] = {}

                        if model not in ann_results[set_id]:
                            ann_results[set_id][model] = {}
                            ann_lums[set_id][model] = {}
                            ann_obs_order[set_id][model] = {}

                    else:
                        set_id = None
                        ann_id = None

                    try:
                        inst_lums = {}
                        obs_order = []
                        for line_ind, line in enumerate(fit_data["SPEC_INFO"]):
                            sp_info = line["SPEC_PATH"].strip(" ").split("/")[-1].split("_")
                            # Want to derive the spectra storage key from the file name, this strips off some
                            #  unnecessary info
                            sp_key = line["SPEC_PATH"].strip(" ").split("/")[-1].split('ra')[-1].split('_spec.fits')[0]

                            # If its not an AnnularSpectra fit then we can just fetch the spectrum from the source
                            #  the normal way
                            if set_id is None:
                                # This adds ra back on, and removes any ident information if it is there
                                sp_key = 'ra' + sp_key
                                # Finds the appropriate matching spectrum object for the current table line
                                spec = self.get_products("spectrum", sp_info[0], sp_info[1], extra_key=sp_key,
                                                         telescope=tel)[0]
                            else:
                                sp_key = 'ra' + sp_key.split('_ident')[0]
                                ann_spec = self.get_annular_spectra(set_id=set_id)
                                spec = ann_spec.get_spectra(ann_id, sp_info[0], sp_info[1])
                                obs_order.append([sp_info[0], sp_info[1]])

                            # Adds information from this fit to the spectrum object.
                            spec.add_fit_data(str(model), line, fit_data["PLOT"+str(line_ind+1)])

                            # The add_fit_data method formats the luminosities nicely, so we grab them back out
                            #  to help grab the luminosity needed to pass to the source object 'add_fit_data' method
                            processed_lums = spec.get_luminosities(model)
                            if spec.instrument not in inst_lums:
                                inst_lums[spec.instrument] = processed_lums

                        if tel == 'xmm':
                            # Ideally the luminosity reported in the source object will be a PN lum, but it's
                            #  not impossible that a PN value won't be available. - it shouldn't matter much, lums
                            #  across the cameras are consistent
                            if "pn" in inst_lums:
                                chosen_lums = inst_lums["pn"]
                            # mos2 generally better than mos1, as mos1 has CCD damage after a certain point in its life
                            elif "mos2" in inst_lums:
                                chosen_lums = inst_lums["mos2"]
                            else:
                                chosen_lums = inst_lums["mos1"]
                        else:
                            # TODO THIS ISN'T NECESSARILY THE WAY I WANT TO DO THIS
                            chosen_lums = processed_lums

                        if set_id is not None:
                            ann_results[set_id][model][spec.annulus_ident] = global_results
                            ann_lums[set_id][model][spec.annulus_ident] = chosen_lums
                            ann_obs_order[set_id][model][spec.annulus_ident] = obs_order
                        else:
                            # Push global fit results, luminosities etc. into the corresponding source object.
                            self.add_fit_data(model, global_results, chosen_lums, sp_key, tel)
                    except (OSError, NoProductAvailableError, IndexError, NotAssociatedError):
                        chosen_lums = {}
                        warn_text = "{src} fit {f} could not be loaded in as there are no matching spectra " \
                                    "available".format(src=self.name, f=fit_name)
                        if not self._samp_member:
                            warn(warn_text, stacklevel=2)
                        else:
                            self._supp_warn.append(warn_text)
                    fit_data.close()

                if len(ann_results) != 0:
                    for set_id in ann_results:
                        try:
                            rel_ann_spec = self.get_annular_spectra(set_id=set_id)
                            for model in ann_results[set_id]:
                                rel_ann_spec.add_fit_data(model, ann_results[set_id][model], ann_lums[set_id][model],
                                                          ann_obs_order[set_id][model])
                                # if model == "constant*tbabs*apec":
                                #     temp_prof = rel_ann_spec.generate_profile(model, 'kT', 'keV')
                                #     self.update_products(temp_prof)
                                #
                                #     # Normalisation profiles can be useful for many things, so we generate them too
                                #     norm_prof = rel_ann_spec.generate_profile(model, 'norm', 'cm^-5')
                                #     self.update_products(norm_prof)
                                #
                                #     if 'Abundanc' in rel_ann_spec.get_results(0, 'constant*tbabs*apec'):
                                #         met_prof = rel_ann_spec.generate_profile(model, 'Abundanc', '')
                                #         self.update_products(met_prof)
                        except (NoProductAvailableError, ValueError):
                            warn_text = "A previous annular spectra profile fit for {src} was not successful, or no " \
                                        "matching spectrum has been loaded, so it cannot be read " \
                                        "in".format(src=self.name)
                            if not self._samp_member:
                                warn(warn_text, stacklevel=2)
                            else:
                                self._supp_warn.append(warn_text)

            os.chdir(og_dir)

            # And finally loading in any conversion factors that have been calculated using XGA's fakeit interface
            if os.path.exists(OUTPUT + "{t}/XSPEC/".format(t=tel) + self.name) and read_fits:
                conv_factors = [OUTPUT + "{t}/XSPEC/".format(t=tel) + self.name + "/" + f
                                for f in os.listdir(OUTPUT + "{t}/XSPEC/".format(t=tel) + self.name)
                                if ".xcm" not in f and "conv_factors" in f]
                for conv_path in conv_factors:
                    res_table = pd.read_csv(conv_path, dtype={"lo_en": str, "hi_en": str})
                    # Gets the model name from the file name of the output results table
                    model = conv_path.split("_")[-4]
                    # We can infer the storage key from the name of the results table, just makes it easier to
                    #  grab the correct spectra
                    storage_key = conv_path.split('/')[-1].split(self.name)[-1][1:].split(model)[0][:-1]

                    # Grabs the ObsID+instrument combinations from the headers of the csv. Makes sure they are unique
                    #  by going to a set (because there will be two columns for each ObsID+Instrument, rate and Lx)
                    # First two columns are skipped because they are energy limits
                    combos = list(set([c.split("_")[1] for c in res_table.columns[2:]]))

                    # Getting the spectra for each column, then assigning rates and luminosities.
                    # Due to the danger of a fit using a piece of data (an ObsID-instrument combo) that isn't currently
                    #  associated with the source, we first fetch the spectra, then in a second loop we assign the
                    #  factors
                    rel_spec = []
                    try:
                        for comb in combos:
                            spec = self.get_products("spectrum", comb[:10], comb[10:], extra_key=storage_key)[0]
                            rel_spec.append(spec)

                        for comb_ind, comb in enumerate(combos):
                            rel_spec[comb_ind].add_conv_factors(res_table["lo_en"].values, res_table["hi_en"].values,
                                                                res_table["rate_{}".format(comb)].values,
                                                                res_table["Lx_{}".format(comb)].values, model, tel)

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

    def _load_regions(self, reg_paths: dict) -> Tuple[dict, dict]:
        """
        An internal method that reads and parses region files found for observations associated with this source.
        Also computes simple matches to find regions likely to be related to the source. This is done for each
        observation of each telescope that has been identified as relevant.

        :param dict reg_paths: A nested dictionary, where top level keys are telescope names, lower level keys
            are ObsIDs, and their values are region file paths (or None if the region file does not exist).
        :return: Two nested dictionaries, the first contains the regions for each of the ObsIDs and the second
            contains the regions that have been very simply matched to the source. These should be ordered from
            closest to furthest from the passed source coordinates. The top level keys are telescope names, the
            lower level keys are ObsIDs, and the values are arrays of either region objects or boolean flags.
        :rtype: Tuple[dict, dict]
        """

        # We set up the dictionaries that are to be returned by this method - one for all the region objects that
        #  have been read in, and one for the basic separation-based initial matches that this does between the
        #  user-defined coordinate and the regions. The top level keys are telescopes with dictionaries as
        #  values, the lower level keys are ObsIDs with lists of region objects (or boolean flags for the
        #  match_dict) as values
        reg_dict = {tel: {} for tel in reg_paths}
        match_dict = {tel: {} for tel in reg_paths}

        # The input reg_dict is a dictionary of telescopes, which each have an entry of a dictionary of ObsIDs, and
        #  the values for those ObsID keys is a path to a region file for that ObsID
        for tel in reg_paths:
            # Iterate through the ObsIDs for the current telescope
            for obs_id in reg_paths[tel]:
                # Read out the original region file path from the dictionary
                reg_file = reg_paths[tel][obs_id]
                # We make a local copy of the region file if the original file path exists and if a local copy
                #  DOESN'T already exist
                reg_copy_path = OUTPUT + tel + "/{o}/{o}_xga_copy.reg".format(o=obs_id)
                # If the reg_file entry in the dictionary for this telescope-ObsID is not None, then we don't need
                #  to check that it exists because that already happened in the initial_products method.
                if reg_file is not None and not os.path.exists(reg_copy_path):
                    # A local copy of the region file is made and used in place of the original - this is because
                    #  parts of XGA can modify it
                    copyfile(reg_file, reg_copy_path)
                    # Regions dictionary updated with path to local region file, if it exists
                    reg_dict[tel][obs_id] = reg_copy_path
                    reg_file = reg_copy_path
                # In the case where there is already a local copy of the region file
                elif reg_file is not None and os.path.exists(reg_copy_path):
                    reg_dict[tel][obs_id] = reg_copy_path
                    reg_file = reg_copy_path

                # Read in the custom region file associated with this source - they are currently on a telescope
                #  level due to the different observational characteristics that might be present for different
                #  telescopes (e.g. different angular resolutions)
                custom_regs = read_ds9(OUTPUT + "{t}/regions/{n}/{n}_custom.reg".format(t=tel, n=self.name))

                if not all([isinstance(reg, SkyRegion) for reg in custom_regs]):
                    # What the error says really - we only want custom sources defined in ra-dec coords, not telescope
                    #  specific coordinates
                    raise TypeError("Custom sources can only be defined in RA-Dec coordinates.")

                if reg_file is not None:
                    ds9_regs = read_ds9(reg_file)
                    # Grab all images for the ObsID, instruments across an ObsID have the same WCS (other than in cases
                    #  where they were generated with different resolutions).
                    #  TODO see issue #908, figure out how to support different resolutions of image
                    try:
                        ims = self.get_images(obs_id, telescope=tel)
                    except NoProductAvailableError:
                        raise NoProductAvailableError("There is no image available for {t}-{o}, associated "
                                                      "with {n}. An image is currently required to check for sky "
                                                      "coordinates being present within a sky region - though "
                                                      "hopefully no-one will ever see this because I'll have fixed "
                                                      "it!".format(t=tel, o=obs_id, n=self.name))
                        w = None
                    # In this case the try statement worked, and so we can extract the WCS from the image
                    else:
                        # It is possible that either a single image or a list of images have been returned
                        if isinstance(ims, list):
                            w = ims[0].radec_wcs
                        else:
                            w = ims.radec_wcs

                # In the case where there is no region file available to us, or the region file has no entries, then
                #  I just set the ds9_regs to [None] because I know the rest of the code can deal with that. It can't
                #  deal with an empty list
                if reg_file is None or len(ds9_regs) == 0:
                    ds9_regs = [None]

                if isinstance(ds9_regs[0], PixelRegion):
                    # If regions exist in pixel coordinates, we need an image WCS to convert them to RA-DEC, so we need
                    #  one of the images supplied in the config file, not anything that XGA generates.
                    #  But as this method is only run once, before XGA generated products are loaded in, it
                    #  should be fine
                    if w is None:
                        raise NoProductAvailableError("There is no image available for observation {t}-{o} "
                                                      "associated with {n}. An image is currently required to "
                                                      "translate pixel regions to "
                                                      "RA-DEC.".format(t=tel, o=obs_id, n=self.name))
                    # We use the WCS that we've made sure is available to convert the pixel regions to RA-Dec
                    #  regions, and then we store them in the reg_dict
                    sky_regs = [reg.to_sky(w) for reg in ds9_regs]
                    # The reg_dict has keys for telescopes, and then lower level keys for ObsIDs
                    reg_dict[tel][obs_id] = np.array(sky_regs)
                elif isinstance(ds9_regs[0], SkyRegion):
                    # If the regions are already in sky coordinates then all is well
                    reg_dict[tel][obs_id] = np.array(ds9_regs)
                else:
                    # So there is an entry in this for EVERY ObsID
                    reg_dict[tel][obs_id] = np.array([None])

                # Here we add the custom sources to the source list, we know they are sky regions as we have
                #  already enforced it. If there was no region list for a particular ObsID (detected by the first
                #  entry in the reg dict being None) and there IS a custom region, we just replace the None with the
                #  custom region
                if reg_dict[tel][obs_id][0] is not None:
                    reg_dict[tel][obs_id] = np.append(reg_dict[tel][obs_id], custom_regs)
                elif reg_dict[tel][obs_id][0] is None and len(custom_regs) != 0:
                    reg_dict[tel][obs_id] = np.array(custom_regs)
                    # Currently if only custom_regions are used the variable w doesn't get defined,
                    # so we need to define it here
                    try:
                        ims = self.get_images(obs_id, telescope=tel)
                    except NoProductAvailableError:
                        raise NoProductAvailableError("There is no image available for {t}-{o}, associated "
                                                      "with {n}. An image is currently required to check for sky "
                                                      "coordinates being present within a sky region - though "
                                                      "hopefully no-one will ever see this because I'll have fixed "
                                                      "it!".format(t=tel, o=obs_id, n=self.name))
                        w = None
                    # In this case the try statement worked, and so we can extract the WCS from the image
                    else:
                        # It is possible that either a single image or a list of images have been returned
                        if isinstance(ims, list):
                            w = ims[0].radec_wcs
                        else:
                            w = ims.radec_wcs

                else:
                    reg_dict[tel][obs_id] = np.array([None])

                # I'm going to ensure that all regions are elliptical, I don't want to hunt through every place in XGA
                #  where I made that assumption
                for reg_ind, reg in enumerate(reg_dict[tel][obs_id]):
                    if isinstance(reg, CircleSkyRegion):
                        # Multiply radii by two because the ellipse based sources want HEIGHT and WIDTH, not RADIUS
                        # Give small angle (though won't make a difference as circular) to avoid problems with angle=0
                        #  that I've noticed previously
                        new_reg = EllipseSkyRegion(reg.center, reg.radius*2, reg.radius*2, Quantity(3, 'deg'))
                        new_reg.visual['color'] = reg.visual['color']
                        reg_dict[tel][obs_id][reg_ind] = new_reg

                # Hopefully this bodge doesn't have any unforeseen consequences
                if reg_dict[tel][obs_id][0] is not None and len(reg_dict[tel][obs_id]) > 1:
                    # Quickly calculating distance between source and center of regions, then sorting
                    # and getting indices. Thus I only match to the closest 5 regions.
                    diff_sort = np.array([_dist_from_source(*self._ra_dec, r)
                                          for r in reg_dict[tel][obs_id]]).argsort()

                    # Unfortunately due to a limitation of the regions module I think you need images
                    #  to do this contains match...
                    within = np.array([reg.contains(SkyCoord(*self._ra_dec, unit='deg'), w)
                                    for reg in reg_dict[tel][obs_id][diff_sort[0:5]]])

                    # Make sure to re-order the region list to match the sorted within array
                    reg_dict[tel][obs_id] = reg_dict[tel][obs_id][diff_sort]

                    # Expands it so it can be used as a mask on the whole set of regions for this observation
                    within = np.pad(within, [0, len(diff_sort) - len(within)])
                    match_dict[tel][obs_id] = within
                # In the case of only one region being in the list, we simplify the above expression
                elif reg_dict[tel][obs_id][0] is not None and len(reg_dict[tel][obs_id]) == 1:
                    if reg_dict[tel][obs_id][0].contains(SkyCoord(*self._ra_dec, unit='deg'), w):
                        match_dict[tel][obs_id] = np.array([True])
                    else:
                        match_dict[tel][obs_id] = np.array([False])
                else:
                    match_dict[tel][obs_id] = np.array([False])

        return reg_dict, match_dict

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
        results_dict = {tel: {} for tel in self.telescopes}
        # And in this one go all the sources that aren't the matched source, we'll need to subtract them.
        anti_results_dict = {tel: {} for tel in self.telescopes}
        # Sources in this dictionary are within the target source region AND matched to initial coordinates,
        # but aren't the chosen source.
        alt_match_dict = {tel: {} for tel in self.telescopes}
        # Goes through all the telescopes and ObsIDs associated with this source, and checks if they have regions
        #  If not then Nones are added to the various dictionaries, otherwise you end up with a list of regions
        #  with missing ObsIDs
        for tel in self.telescopes:
            for obs in self.obs_ids[tel]:
                if obs in self._initial_regions[tel]:
                    # This sets up an array of matched regions, accounting for the problems that can occur when
                    #  there is only one region in the region list (numpy's indexing gets very angry). The array
                    #  of matched region(s) set up here is used in this method.
                    if len(self._initial_regions[tel][obs]) == 1 and not self._initial_region_matches[tel][obs][0]:
                        init_region_matches = np.array([])
                    elif len(self._initial_regions[tel][obs]) == 1 and self._initial_region_matches[tel][obs][0]:
                        init_region_matches = self._initial_regions[tel][obs]
                    elif len(self._initial_regions[tel][obs][self._initial_region_matches[tel][obs]]) == 0:
                        init_region_matches = np.array([])
                    else:
                        init_region_matches = self._initial_regions[tel][obs][self._initial_region_matches[tel][obs]]

                    # If there are no matches then the returned result is just None.
                    if len(init_region_matches) == 0:
                        results_dict[tel][obs] = None
                    else:
                        interim_reg = []
                        # The only solution I could think of is to go by the XCS standard of region files, so green
                        #  is extended, red is point etc. - not ideal but I'll just explain in the documentation
                        # for entry in self._initial_regions[obs][self._initial_region_matches[obs]]:
                        for entry in init_region_matches:
                            if entry.visual["color"] in allowed_colours:
                                interim_reg.append(entry)

                        # Different matching possibilities
                        if len(interim_reg) == 0:
                            results_dict[tel][obs] = None
                        elif len(interim_reg) == 1:
                            results_dict[tel][obs] = interim_reg[0]
                        # Matching to multiple sources would be very problematic, so throw an error
                        elif len(interim_reg) > 1 and source_type == "pnt":
                            # I made the _load_regions method sort the outputted region dictionaries by distance
                            #  from the input coordinates, so I know that the 0th entry will be the closest to the
                            #  source coords. Hence I choose that one for pnt source multi-matches like this, see
                            #  comment 2 of issue #639 for an example.
                            results_dict[tel][obs] = interim_reg[0]
                            warn_text = "{ns} matches for the point source {n} are found in the {t}-{o} region " \
                                        "file. The source nearest to the passed coordinates is accepted, all others " \
                                        "will be placed in the alternate match category and will not be removed " \
                                        "by masks.".format(o=obs, n=self.name, ns=len(interim_reg), t=tel)
                            if not self._samp_member:
                                warn(warn_text, stacklevel=2)
                            else:
                                self._supp_warn.append(warn_text)

                        elif len(interim_reg) > 1 and source_type == "ext":
                            # raise MultipleMatchError("More than one match for {n} is found in the region file "
                            #                          "for observation {t}-{o}, this cannot yet be dealt with "
                            #                          "for extended sources.".format(o=obs, n=self.name, t=tel))

                            results_dict[tel][obs] = interim_reg[0]
                            warn_text = "{ns} matches for the extended source {n} are found in the {t}-{o} region " \
                                        "file. The source nearest to the passed coordinates is accepted, all others " \
                                        "will be placed in the alternate match category and will not be removed " \
                                        "by masks.".format(o=obs, n=self.name, ns=len(interim_reg), t=tel)
                            if not self._samp_member:
                                warn(warn_text, stacklevel=2)
                            else:
                                self._supp_warn.append(warn_text)

                    # Alt match is used for when there is a secondary match to a point source
                    alt_match_reg = [entry for entry in init_region_matches if entry != results_dict[tel][obs]]
                    alt_match_dict[tel][obs] = alt_match_reg

                    # These are all the sources that aren't a match, and so should be removed from any analysis
                    not_source_reg = [reg for reg in self._initial_regions[tel][obs] if reg != results_dict[tel][obs]
                                      and reg not in alt_match_reg]
                    anti_results_dict[tel][obs] = not_source_reg

                else:
                    results_dict[tel][obs] = None
                    alt_match_dict[tel][obs] = []
                    anti_results_dict[tel][obs] = []

        return results_dict, alt_match_dict, anti_results_dict

    def _generate_interloper_mask(self, mask_image: Image, region_distance: Quantity = None) -> ndarray:
        """
        An internal method to create masks for the removal of contaminating sources - the passed image is used both
        to provide dimension/WCS information, but also to identify the telescope used for the image. Regions are
        kept separate across telescopes.

        :param Image mask_image: The image for which to create the interloper mask.
        :param Quantity region_distance: The distance from the central coordinate within which contaminating
            regions will be included in this mask. Introduced as a counter to the very large numbers of regions
            associated with eROSITA observations. Default is None, in which case all contaminating regions will be
            included in the mask.
        :return: A numpy array of 0s and 1s which acts as a mask to remove interloper sources.
        :rtype: ndarray
        """

        # This is the array that the mask gets built in - initially all ones and as we move through the regions we
        #  start to set the bits that need to be excluded to zero
        mask = np.ones(mask_image.shape)

        # If a region inclusion distance has not been set, then we create a mask of all True so that all regions
        #  are included in the final mask
        if region_distance is None:
            for_mask = np.full(len(self._interloper_regions[mask_image.telescope]), True)
        # However, if an inclusion distance has been passed, we use it to create a True-False array to define which
        #  regions we want to include.
        else:
            for_mask = (Quantity(np.array([_dist_from_source(*self._ra_dec, r)
                                           for r in self._interloper_regions[mask_image.telescope]])) < region_distance)

        # TODO Maybe this is a candidate for some intelligent multi-threading?
        for r in np.array(self._interloper_regions[mask_image.telescope])[for_mask]:
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

                    mask[pr.to_mask().to_image(mask_image.shape) != 0] = 0
                except ValueError:
                    pass

        # masks = [reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
        #          for reg in self._interloper_regions if reg is not None]
        # interlopers = sum([m for m in masks if m is not None])
        #
        # mask[interlopers != 0] = 0

        return mask

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

        # TODO METHODS LIKE THIS DO NOT BELONG IN THE SOURCE CLASS - THEY SHOULD ACT ON THE SOURCE CLASS, PARTICULARLY
        #  NOW THAT WE'RE GOING TO SUPPORTING DIFFERENT TELESCOPES
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
        w = reg.width.to('deg').value / 2 / sky_to_deg
        # We do the same for the height
        h = reg.height.to('deg').value / 2 / sky_to_deg
        if w == h:
            shape_str = "(({t}) IN circle({cx},{cy},{r}))"
            shape_str = shape_str.format(t=c_str, cx=conv_cen[0].value, cy=conv_cen[1].value, r=h)
        else:
            # The rotation angle from the region object is in degrees already
            shape_str = "(({t}) IN ellipse({cx},{cy},{w},{h},{rot}))".format(t=c_str, cx=conv_cen[0].value,
                                                                             cy=conv_cen[1].value, w=w, h=h,
                                                                             rot=reg.angle.value)
        return shape_str

    def _get_phot_prod(self, prod_type: str, obs_id: str = None, inst: str = None, lo_en: Quantity = None,
                       hi_en: Quantity = None, psf_corr: bool = False, psf_model: str = "ELLBETA",
                       psf_bins: int = 4, psf_algo: str = "rl", psf_iter: int = 15, telescope: str = None) \
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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is
            None, which means all images/expmaps/ratemaps matching the other criteria will be returned.
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
            matched_prods = self.get_products(prod_type, obs_id, inst, extra_key=energy_key, telescope=telescope)
        elif not psf_corr and not with_lims:
            broad_matches = self.get_products(prod_type, obs_id, inst, telescope=telescope)
            matched_prods = [p for p in broad_matches if not p.psf_corrected]
        elif psf_corr and with_lims:
            # Here we need to add the extra key to the energy key
            matched_prods = self.get_products(prod_type, obs_id, inst, extra_key=energy_key + extra_key,
                                              telescope=telescope)
        elif psf_corr and not with_lims:
            # Here we don't know the energy key, so we have to look for partial matches in the get_products return
            broad_matches = self.get_products(prod_type, obs_id, inst, extra_key=None, just_obj=False,
                                              telescope=telescope)
            matched_prods = [p[-1] for p in broad_matches if extra_key in p[-2]]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any {p}s matching your input.".format(p=prod_type))

        return matched_prods

    def _get_prof_prod(self, search_key: str, obs_id: str = None, inst: str = None, central_coord: Quantity = None,
                       radii: Quantity = None, lo_en: Quantity = None, hi_en: Quantity = None, telescope: str = None) \
            -> Union[BaseProfile1D, List[BaseProfile1D]]:
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
        :param Quantity lo_en: The lower energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed hi_en must be too.
        :param Quantity hi_en: The higher energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed lo_en must be too.
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all profiles matching the other criteria will be returned.
        :return: An XGA profile object (if there is an exact match), or a list of XGA profile objects (if there
            were multiple matching products).
        :rtype: Union[BaseProfile1D, List[BaseProfile1D]]
        """
        if all([lo_en is None, hi_en is None]):
            energy_key = "_"
        elif all([lo_en is not None, hi_en is not None]):
            energy_key = "bound_{l}-{h}_".format(l=lo_en.to('keV').value, h=hi_en.to('keV').value)
        else:
            raise ValueError("lo_en and hi_en must be either BOTH None or BOTH an Astropy quantity.")

        if central_coord is None:
            central_coord = self.default_coord
        cen_chunk = "ra{r}_dec{d}_".format(r=central_coord[0].value, d=central_coord[1].value)

        if radii is not None:
            radii = self.convert_radius(radii, 'deg')
            rad_chunk = "r" + "_".join(radii.value.astype(str))
            rad_info = True
        else:
            rad_info = False

        broad_prods = self.get_products(search_key, obs_id, inst, just_obj=False, telescope=telescope)
        matched_prods = []
        for p in broad_prods:
            rad_str = p[-2].split("_st")[0].split(cen_chunk)[-1]

            if cen_chunk in p[-2] and energy_key in p[-2] and rad_info and rad_str == rad_chunk:
                matched_prods.append(p[-1])
            elif cen_chunk in p[-2] and energy_key in p[-2] and not rad_info:
                matched_prods.append(p[-1])

        return matched_prods

    def _get_lc_prod(self, outer_radius: Union[str, Quantity] = None, obs_id: str = None, inst: str = None,
                     inner_radius: Union[str, Quantity] = None, lo_en: Quantity = None, hi_en: Quantity = None,
                     time_bin_size: Quantity = None, telescope: str = None) \
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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all profiles matching the other criteria will be returned.
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
        all_lcs = self.get_products(search_key, obs_id, inst, telescope=telescope)
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

    def _all_peaks(self, method: str, source_type: str):
        """
        An internal method that finds the X-ray peaks for all the available telescopes, observations, and
        instruments, as well as the combined ratemap. Peak positions for individual ratemap products are allowed
        to not converge, and will just write None to the peak dictionary, but if the peak of the combined ratemap
        fails to converge an error will be thrown. The combined ratemap peak will also be stored by itself in an
        attribute, to allow a property getter easy access.

        :param str method: The method to be used for peak finding.
        :param str source_type: Whether the source is point or extended, as this affects how the find_peak method
            works. For extended it will be iterative, but not for the point source.
        """
        # TODO Make this all more elegant

        if self == BaseSource:
            raise TypeError("This internal method cannot be used from BaseSource!")

        if source_type not in ['extended', 'point']:
            raise ValueError("The 'source_type' argument must either be 'extended' or 'point'.")

        for tel in self.telescopes:
            try:
                comb_rt = self.get_combined_ratemaps(self._peak_lo_en, self._peak_hi_en, telescope=tel)
            except NoProductAvailableError:
                if self._use_peak and tel == 'xmm':
                    # I didn't want to import this here, but otherwise circular imports become a problem
                    from xga.generate.sas import emosaic
                    emosaic(self, "image", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
                    emosaic(self, "expmap", self._peak_lo_en, self._peak_hi_en, disable_progress=True)
                    comb_rt = self.get_combined_ratemaps(self._peak_lo_en, self._peak_hi_en, telescope=tel)
                elif self._use_peak and tel == 'erosita':
                    warn_text = ("Generating the combined images required for this is not supported for eROSITA - we "
                                 "will use the highest exposure ObsID instead - associated with source "
                                 "{n}").format(t=tel, n=self.name)
                    if not self._samp_member:
                        warn(warn_text, stacklevel=2)
                    else:
                        self._supp_warn.append(warn_text)

                    from xga.generate.esass import evtool_image, expmap
                    evtool_image(self, self._peak_lo_en, self._peak_hi_en, disable_progress=True)
                    expmap(self, self._peak_lo_en, self._peak_hi_en, disable_progress=True)

                    rel_rts = self.get_ratemaps(lo_en=self._peak_lo_en, hi_en=self._peak_hi_en, telescope=tel)
                    if not isinstance(rel_rts, list):
                        rel_rts = [rel_rts]
                    comb_rt = np.array(rel_rts)[np.argmax([rt.expmap.get_exp(self.ra_dec).to('s').value
                                                           for rt in rel_rts])]
                elif self._use_peak:
                    warn_text = ("Generating the combined images required for this is not supported for {t} "
                                 "currently - associated with source {n}").format(t=tel, n=self.name)
                    if not self._samp_member:
                        warn(warn_text, stacklevel=2)
                    else:
                        self._supp_warn.append(warn_text)
                    comb_rt = None
                else:
                    comb_rt = None

            # TODO return this to not checking if comb_rt is None once other telescopes fully supported
            if self._use_peak and comb_rt is not None and source_type == 'extended':
                coord, near_edge, converged, cluster_coords, other_coords = self.find_peak(comb_rt, method=method)
                # Updating nH for new coord, probably won't make a difference most of the time
                self._nH = nh_lookup(coord)[0]

            elif self._use_peak and comb_rt is not None and source_type == 'point':
                coord, near_edge = self.find_peak(comb_rt)
                # Updating nH for new coord, probably won't make a difference most of the time
                self._nH = nh_lookup(coord)[0]

            else:
                # If we don't care about peak finding then this is the one to go for
                coord = self.ra_dec
                converged = True
                cluster_coords = np.ndarray([])
                other_coords = []
                if comb_rt is not None:
                    near_edge = comb_rt.near_edge(coord)
                else:
                    # TODO remove this once full other telescope support is implemented
                    near_edge = False

            # Unfortunately if the peak convergence fails for the combined ratemap I have to raise an error
            if source_type == 'extended' and converged:
                self._peaks[tel]["combined"] = coord
                self._peaks_near_edge[tel]["combined"] = near_edge
                # I'm only going to save the point cluster positions for the combined ratemap
                self._chosen_peak_cluster[tel] = cluster_coords
                self._other_peak_clusters[tel] = other_coords
            elif source_type == 'point':
                self._peaks[tel]["combined"] = coord
                self._peaks_near_edge[tel]["combined"] = near_edge
            else:
                raise PeakConvergenceFailedError("Peak finding on the combined {t} ratemap failed to converge within "
                                                 "15kpc for {n} in the {l}-{u} energy "
                                                 "band.".format(n=self.name, l=self._peak_lo_en, u=self._peak_hi_en,
                                                                t=tel))

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

    # This is used to name files and directories so this is not allowed to change.
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
        # TODO frankly I don't know if there is any point to this anymore, but I won't remove it for now
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
        # TODO frankly I don't know if there is any point to this anymore, but I won't remove it for now
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

    def update_products(self, prod_obj: Union[BaseProduct, BaseAggregateProduct, BaseProfile1D, List[BaseProduct],
                                              List[BaseAggregateProduct], List[BaseProfile1D]],
                        update_inv: bool = True):
        """
        Setter method for the products attribute of source objects. Cannot delete existing products, but will
        overwrite existing products. Raises errors if the telescope or ObsID is not associated with this source
        or the instrument is not associated with the ObsID. Lists of products can also be passed
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
                tel = po.telescope
                p_type = po.type

                # Previously, merged images/exposure maps were stored in a separate dictionary, but now everything lives
                #  together - merged products do get a 'combined' prefix on their product type key though
                if obs_id == "combined":
                    p_type = "combined_" + p_type

                if tel not in self.telescopes:
                    raise TelescopeNotAssociatedError("The {t} telescope is not associated with this X-ray "
                                                      "source; the {o}-{i} {pt} cannot be added to "
                                                      "{n}.".format(t=tel, o=obs_id, i=inst, pt=p_type, n=self.name))

                # 'Combined' will effectively be stored as another ObsID
                if "combined" not in self._products[tel]:
                    self._products[tel]["combined"] = {}

                # There can be combined obs ids using only one instrument
                for allowed_inst in ALLOWED_INST[tel]:
                    if allowed_inst not in self._products[tel]["combined"]:
                        self._products[tel]["combined"][allowed_inst] = {}
                
                # This is for combined obs and combined instrument
                if "combined" not in self._products[tel]["combined"]:
                    self._products[tel]["combined"]["combined"] = {}
                                
                # TODO set up combined and all instrument names - needs to be generalised remember!

                # The product gets the name of this source object added to it
                po.src_name = self.name

                # Double check that something is trying to add products from another source to the current one.
                if obs_id != "combined" and obs_id not in self.obs_ids[tel]:
                    raise NotAssociatedError("{t}-{o} is not associated with source {n}.".format(t=tel, o=obs_id,
                                                                                                 n=self.name))
                elif inst != "combined" and inst not in self.instruments[tel][obs_id]:
                    raise NotAssociatedError("{i} is not associated with {t} observation "
                                             "{o} for source {n}.".format(i=inst, o=obs_id,
                                                                          t=PRETTY_TELESCOPE_NAMES[tel],
                                                                          n=self.name))

                if extra_key is not None and obs_id != "combined":
                    # If there is no entry for this 'extra key' (energy band for instance) already, we must make one
                    if extra_key not in self._products[tel][obs_id][inst]:
                        self._products[tel][obs_id][inst][extra_key] = {}
                    self._products[tel][obs_id][inst][extra_key][p_type] = po

                elif extra_key is None and obs_id != "combined":
                    self._products[tel][obs_id][inst][p_type] = po

                # Here we deal with merged products, they live in the same dictionary, but with no instrument entry
                #  and ObsID = 'combined'
                elif extra_key is not None and obs_id == "combined":
                    if extra_key not in self._products[tel][obs_id][inst]:
                        self._products[tel][obs_id][inst][extra_key] = {}
                    self._products[tel][obs_id][inst][extra_key][p_type] = po

                elif extra_key is None and obs_id == "combined":
                    self._products[tel][obs_id][inst][p_type] = po

                # This is for an image being added, so we look for a matching exposure map. If it exists we can
                #  make a ratemap
                if p_type == "image":
                    # No chance of an expmap being PSF corrected, so we just use the energy key to
                    #  look for one that matches our new image
                    exs = [prod for prod in self.get_products("expmap", obs_id, inst, just_obj=False, telescope=tel)
                           if en_key in prod]
                    if len(exs) == 1:
                        new_rt = RateMap(po, exs[0][-1])
                        new_rt.src_name = self.name
                        self._products[tel][obs_id][inst][extra_key]["ratemap"] = new_rt

                # However, if it's an exposure map that's been added, we have to look for matching image(s). There
                #  could be multiple, because there could be a normal image, and a PSF corrected image
                elif p_type == "expmap":
                    # PSF corrected extra keys are built on top of energy keys, so if the en_key is within the extra
                    #  key string it counts as a match
                    ims = [prod for prod in self.get_products("image", obs_id, inst, just_obj=False, telescope=tel)
                           if en_key in prod[-2]]
                    # If there is at least one match, we can go to work
                    if len(ims) != 0:
                        for im in ims:
                            new_rt = RateMap(im[-1], po)
                            new_rt.src_name = self.name
                            self._products[tel][obs_id][inst][im[-2]]["ratemap"] = new_rt

                # The same behaviours hold for combined_image and combined_expmap, but they get
                #  stored in slightly different places
                elif p_type == "combined_image":
                    exs = [prod for prod in self.get_products("combined_expmap", just_obj=False, telescope=tel)
                           if en_key in prod]
                    if len(exs) == 1:
                        new_rt = RateMap(po, exs[0][-1])
                        new_rt.src_name = self.name
                        # Remember obs_id for combined products is just 'combined'
                        self._products[tel][obs_id][inst][extra_key]["combined_ratemap"] = new_rt

                elif p_type == "combined_expmap":
                    ims = [prod for prod in self.get_products("combined_image", just_obj=False, telescope=tel)
                           if en_key in prod[-2]]
                    if len(ims) != 0:
                        for im in ims:
                            new_rt = RateMap(im[-1], po)
                            new_rt.src_name = self.name
                            self._products[tel][obs_id][inst][im[-2]]["combined_ratemap"] = new_rt
                

                if isinstance(po, BaseProfile1D) and not os.path.exists(po.save_path):
                    po.save()
                # Here we make sure to store a record of the added product in the relevant inventory file
                if isinstance(po, (BaseProduct)) and po.obs_id != 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "{t}/{o}/inventory.csv".format(t=tel, o=obs_id), dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    # I want only the name of the file as it is in the storage directory, I don't want an
                    #  absolute path, so I remove the leading information about the absolute location in
                    #  the .path string
                    f_name = po.path.split(OUTPUT + "{t}/{o}/".format(t=tel, o=obs_id))[-1]

                    # Images, exposure maps, and other such things are not source specific, so I don't want
                    #  the inventory file to assign them a specific source
                    if isinstance(po, Image):
                        s_name = ''
                    else:
                        s_name = po.src_name

                    # Creates new pandas series to be appended to the inventory dataframe
                    new_line = pd.Series([f_name, obs_id, inst, info_key, s_name, po.type],
                                         ['file_name', 'obs_id', 'inst', 'info_key', 'src_name', 'type'], dtype=str)
                    # Concatenates the series with the inventory dataframe
                    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)

                    # Checks for rows that are exact duplicates, this should never happen as far as I can tell, but
                    #  if it did I think it would cause problems so better to be safe and add this.
                    inven.drop_duplicates(subset=None, keep='first', inplace=True)
                    # Saves the updated inventory file
                    inven.to_csv(OUTPUT + "{t}/{o}/inventory.csv".format(t=tel, o=obs_id), index=False)
                
                elif isinstance(po, (BaseProduct)) and obs_id == 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "{t}/combined/inventory.csv".format(t=tel), dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    # TODO AT LEAST SOME COMBINED PRODUCTS NOW DO HAVE THIS INFORMATION STORED IN THEM, IT WOULD
                    #  PROBABLY BE A GOOD IDEA TO UPDATE HOW THIS WORKS AT SOME POINT
                    # We know that this particular product is a combination of multiple ObsIDs, and those ObsIDs
                    #  are not stored explicitly within the product object. However we are currently within the
                    #  source object that they were generated from, thus we do have that information available
                    # Using the _instruments attribute also gives us access to inst information
                    i_str = "/".join([i for o in self.instruments[tel] for i in self.instruments[tel][o]])
                    o_str = "/".join([o for o in self.instruments[tel] for i in self.instruments[tel][o]])
                    # They cannot be stored as lists for a single column entry in a csv though, so I am smushing
                    #  them into strings

                    f_name = po.path.split(OUTPUT + "{t}/combined/".format(t=tel))[-1]
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
                    inven.to_csv(OUTPUT + "{t}/combined/inventory.csv".format(t=tel), index=False)
                
                # Here we make sure to store a record of the added product in the relevant inventory file
                # TODO update this for all BaseAggregateProducts - I think the iteration method is acting strangley
                elif isinstance(po, (AnnularSpectra)) and update_inv:                         # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key                    
                    # Adding each component product to the inventory
                    for comp_po in po.all_spectra:
                        if comp_po.obs_id == "combined":
                            inven = pd.read_csv(OUTPUT + "{t}/combined/inventory.csv".format(t=tel), dtype=str)

                            # TODO AT LEAST SOME COMBINED PRODUCTS NOW DO HAVE THIS INFORMATION STORED IN THEM, IT WOULD
                            #  PROBABLY BE A GOOD IDEA TO UPDATE HOW THIS WORKS AT SOME POINT
                            # We know that this particular product is a combination of multiple ObsIDs, and those ObsIDs
                            #  are not stored explicitly within the product object. However we are currently within the
                            #  source object that they were generated from, thus we do have that information available
                            # Using the _instruments attribute also gives us access to inst information
                            i_str = "/".join([i for o in self.instruments[tel] for i in self.instruments[tel][o]])
                            o_str = "/".join([o for o in self.instruments[tel] for i in self.instruments[tel][o]])
                            # They cannot be stored as lists for a single column entry in a csv though, so I am smushing
                            #  them into strings

                            f_name = comp_po.path.split(OUTPUT + "{t}/combined/".format(t=tel))[-1]
                            if isinstance(comp_po, Image):
                                s_name = ''
                            else:
                                s_name = comp_po.src_name

                            # Creates new pandas series to be appended to the inventory dataframe
                            new_line = pd.Series([f_name, o_str, i_str, info_key, s_name, comp_po.type],
                                                ['file_name', 'obs_ids', 'insts', 'info_key', 'src_name', 'type'], dtype=str)
                            # Concatenates the series with the inventory dataframe
                            inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)
                            inven.drop_duplicates(subset=None, keep='first', inplace=True)
                            inven.to_csv(OUTPUT + "{t}/combined/inventory.csv".format(t=tel), index=False)

                        else:
                            inven = pd.read_csv(OUTPUT + "{t}/{o}/inventory.csv".format(t=tel, o=comp_po.obs_id), dtype=str)

                            # I want only the name of the file as it is in the storage directory, I don't want an
                            #  absolute path, so I remove the leading information about the absolute location in
                            #  the .path string
                            f_name = comp_po.path.split(OUTPUT + "{t}/{o}/".format(t=tel, o=comp_po.obs_id))[-1]

                            # Images, exposure maps, and other such things are not source specific, so I don't want
                            #  the inventory file to assign them a specific source
                            if isinstance(comp_po, Image):
                                s_name = ''
                            else:
                                s_name = comp_po.src_name

                            # Creates new pandas series to be appended to the inventory dataframe
                            new_line = pd.Series([f_name, comp_po.obs_id, comp_po.instrument, info_key, s_name, comp_po.type],
                                                ['file_name', 'obs_id', 'inst', 'info_key', 'src_name', 'type'], dtype=str)
                            # Concatenates the series with the inventory dataframe
                            inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)

                            # Checks for rows that are exact duplicates, this should never happen as far as I can tell, but
                            #  if it did I think it would cause problems so better to be safe and add this.
                            inven.drop_duplicates(subset=None, keep='first', inplace=True)
                            # Saves the updated inventory file
                            inven.to_csv(OUTPUT + "{t}/{o}/inventory.csv".format(t=tel, o=comp_po.obs_id), index=False)
                    

                elif isinstance(po, BaseProfile1D) and obs_id != 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "{t}/profiles/{n}/inventory.csv".format(n=self.name, t=tel),
                                        dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    f_name = po.save_path.split(OUTPUT + "{t}/profiles/{n}/".format(t=tel, n=self.name))[-1]
                    i_str = po.instrument
                    o_str = po.obs_id
                    # Creates new pandas series to be appended to the inventory dataframe
                    new_line = pd.Series([f_name, o_str, i_str, info_key, po.src_name, po.type],
                                         ['file_name', 'obs_ids', 'insts', 'info_key', 'src_name', 'type'], dtype=str)
                    # Concatenates the series with the inventory dataframe
                    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)
                    inven.drop_duplicates(subset=None, keep='first', inplace=True)
                    inven.to_csv(OUTPUT + "{t}/profiles/{n}/inventory.csv".format(t=tel, n=self.name), index=False)

                elif isinstance(po, BaseProfile1D) and obs_id == 'combined' and update_inv:
                    inven = pd.read_csv(OUTPUT + "{t}/profiles/{n}/inventory.csv".format(t=tel, n=self.name),
                                        dtype=str)

                    # Don't want to store a None value as a string for the info_key
                    if extra_key is None:
                        info_key = ''
                    else:
                        info_key = extra_key

                    f_name = po.save_path.split(OUTPUT + "{t}/profiles/{n}/".format(t=tel, n=self.name))[-1]
                    i_str = "/".join([i for o in self.instruments[tel] for i in self.instruments[tel][o]])
                    o_str = "/".join([o for o in self.instruments[tel] for i in self.instruments[tel][o]])
                    # Creates new pandas series to be appended to the inventory dataframe
                    new_line = pd.Series([f_name, o_str, i_str, info_key, po.src_name, po.type],
                                         ['file_name', 'obs_ids', 'insts', 'info_key', 'src_name', 'type'], dtype=str)
                    # Concatenates the series with the inventory dataframe
                    inven = pd.concat([inven, new_line.to_frame().T], ignore_index=True)
                    inven.drop_duplicates(subset=None, keep='first', inplace=True)
                    inven.to_csv(OUTPUT + "{t}/profiles/{n}/inventory.csv".format(t=tel, n=self.name), index=False)

    def get_products(self, p_type: str, obs_id: str = None, inst: str = None,
                     extra_key: str = None, just_obj: bool = True, telescope: str = None) -> List[BaseProduct]:
        """
        This is the getter for the products data structure of Source objects. Passing a product type
        such as 'events' or 'images' will return every matching entry in the products data structure.

        :param str p_type: Product type identifier. e.g. image or expmap.
        :param str obs_id: Optionally, a specific obs_id to search can be supplied.
        :param str inst: Optionally, a specific instrument to search can be supplied.
        :param str extra_key: Optionally, an extra key (like an energy bound) can be supplied.
        :param bool just_obj: A boolean flag that controls whether this method returns just the product objects,
            or the other information that goes with it like ObsID and instrument.
        :param str telescope: Optionally, a specific telescope to search can be supplied.
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

        # We check to see if the telescope the user has passed (assuming it isn't just None) is actually relevant
        #  to this source
        if telescope is not None and telescope not in self.telescopes:
            raise NotAssociatedError("The {t} telescope is not associated with {n}.".format(t=telescope, n=self.name))
        # If the telescope IS associated and the ObsID has been supplied then we can check that it is valid
        elif (telescope is not None and telescope in self.telescopes) and \
                (obs_id is not None and obs_id != 'combined' and obs_id not in self.obs_ids[telescope]):
            raise NotAssociatedError("{o} is not associated with the {t} telescope for "
                                     "{n}.".format(o=obs_id, n=self.name, t=telescope))
        # Finally we can see if supplied instruments are valid, if telescope and ObsID are supplied
        elif (telescope is not None and telescope in self.telescopes) and \
                (obs_id is not None and obs_id in self.obs_ids[telescope]) and \
                (inst is not None and inst != 'combined' and inst not in self.instruments[telescope][obs_id]):
            raise NotAssociatedError("{t}-{o} is associated with {n}, but {i} is not associated with that "
                                     "observation".format(t=telescope, o=obs_id, n=self.name, i=inst))

        # This dictionary is used to store the matching products that are located
        matches = []

        # Iterates through the dict search return, but each match is likely to be a very nested list,
        # with the degree of nesting dependent on product type (as event lists live a level up from
        # images for instance
        for match in dict_search(p_type, self._products):
            out = []
            unpack_list(match)
            # Only adds to matches dict if this particular match is for the obs_id and instrument passed to this method
            # Though all matches will be returned if no obs_id/inst is passed
            if (telescope == out[0] or telescope is None) and (obs_id == out[1] or obs_id is None) and \
                    (inst == out[2] or inst is None) and (extra_key in out or extra_key is None) and not just_obj:
                matches.append(out)
            elif (telescope == out[0] or telescope is None) and (obs_id == out[1] or obs_id is None) and \
                    (inst == out[2] or inst is None) and (extra_key in out or extra_key is None) and just_obj:
                matches.append(out[-1])

        return matches

    # And here I'm adding a bunch of get methods that should mean the user never has to use get_products, for
    #  individual product types. It will also mean that they will never have to figure out extra keys themselves
    #  and I can make lists of 1 product return just as the product without being a breaking change
    def get_images(self, obs_id: str = None, inst: str = None, lo_en: Quantity = None, hi_en: Quantity = None,
                   psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                   psf_iter: int = 15, telescope: str = None) -> Union[Image, List[Image]]:
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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all images matching the other criteria will be returned.
        :return: An XGA Image object (if there is an exact match), or a list of XGA Image objects (if there
            were multiple matching products).
        :rtype: Union[Image, List[Image]]
        """
        return self._get_phot_prod("image", obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                                   psf_iter, telescope)

    def get_expmaps(self, obs_id: str = None, inst: str = None, lo_en: Quantity = None, hi_en: Quantity = None,
                    telescope: str = None) -> Union[ExpMap, List[ExpMap]]:
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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all exposure maps matching the other criteria will be returned.
        :return: An XGA ExpMap object (if there is an exact match), or a list of XGA ExpMap objects (if there
            were multiple matching products).
        :rtype: Union[ExpMap, List[ExpMap]]
        """
        return self._get_phot_prod("expmap", obs_id, inst, lo_en, hi_en, False, telescope=telescope)

    def get_ratemaps(self, obs_id: str = None, inst: str = None, lo_en: Quantity = None, hi_en: Quantity = None,
                     psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                     psf_iter: int = 15, telescope: str = None) -> Union[RateMap, List[RateMap]]:
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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all ratemaps matching the other criteria will be returned.
        :return: An XGA RateMap object (if there is an exact match), or a list of XGA RateMap objects (if there
            were multiple matching products).
        :rtype: Union[RateMap, List[RateMap]]
        """
        return self._get_phot_prod("ratemap", obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo,
                                   psf_iter, telescope)

    def get_combined_images(self, lo_en: Quantity = None, hi_en: Quantity = None, psf_corr: bool = False,
                            psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                            psf_iter: int = 15, telescope: str = None) -> Union[Image, List[Image]]:
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
        :param str telescope: Optionally, a specific telescope to search for combined images can be supplied. The
            default is None, which means all combined images matching the other criteria will be returned.
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
            matched_prods = self.get_products('combined_image', extra_key=energy_key, telescope=telescope)
        elif not psf_corr and not with_lims:
            broad_matches = self.get_products("combined_image", telescope=telescope)
            matched_prods = [p for p in broad_matches if not p.psf_corrected]
        elif psf_corr and with_lims:
            # Here we need to add the extra key to the energy key
            matched_prods = self.get_products('combined_image', extra_key=energy_key + extra_key, telescope=telescope)
        elif psf_corr and not with_lims:
            # Here we don't know the energy key, so we have to look for partial matches in the get_products return
            broad_matches = self.get_products('combined_image', extra_key=None, just_obj=False, telescope=telescope)
            matched_prods = [p[-1] for p in broad_matches if extra_key in p[-2]]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined images matching your input.")

        return matched_prods

    def get_combined_expmaps(self, lo_en: Quantity = None, hi_en: Quantity = None,
                             telescope: str = None) -> Union[ExpMap, List[ExpMap]]:
        """
        A method to retrieve combined XGA ExpMap objects, as in those exposure maps that have been created by
        merging all available data for this source. This supports setting the energy limits of the specific
        exposure maps you would like. A NoProductAvailableError error will be raised if no matches are found.

        :param Quantity lo_en: The lower energy limit of the exposure maps you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param Quantity hi_en: The upper energy limit of the exposure maps you wish to retrieve, the default
            is None (which will retrieve all images regardless of energy limit).
        :param str telescope: Optionally, a specific telescope to search for combined exposure maps can be
            supplied. The default is None, which means all combined exposure maps matching the other criteria
            will be returned.
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

        matched_prods = self.get_products('combined_expmap', extra_key=energy_key, telescope=telescope)
        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined exposure maps matching your input.")

        return matched_prods

    def get_combined_ratemaps(self, lo_en: Quantity = None, hi_en: Quantity = None,  psf_corr: bool = False,
                              psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                              psf_iter: int = 15, telescope: str = None) -> Union[RateMap, List[RateMap]]:
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
        :param str telescope: Optionally, a specific telescope to search for combined ratemaps can be supplied. The
            default is None, which means all combined ratemaps matching the other criteria will be returned.
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
            matched_prods = self.get_products('combined_ratemap', extra_key=energy_key, telescope=telescope)
        elif not psf_corr and not with_lims:
            broad_matches = self.get_products("combined_ratemap", telescope=telescope)
            matched_prods = [p for p in broad_matches if not p.psf_corrected]
        elif psf_corr and with_lims:
            # Here we need to add the extra key to the energy key
            matched_prods = self.get_products('combined_ratemap', extra_key=energy_key + extra_key,
                                              telescope=telescope)
        elif psf_corr and not with_lims:
            # Here we don't know the energy key, so we have to look for partial matches in the get_products return
            broad_matches = self.get_products('combined_ratemap', extra_key=None, just_obj=False, telescope=telescope)
            matched_prods = [p[-1] for p in broad_matches if extra_key in p[-2]]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined ratemaps matching your input.")

        return matched_prods

    def _get_spec_prod(self, outer_radius: Union[str, Quantity], obs_id: str = None, inst: str = None,
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                    min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                    telescope: str = None) -> Union[Spectrum, List[Spectrum]]:
        """
        A useful method that wraps the get_products function to allow you to easily retrieve XGA Spectrum objects.
        Simply pass the desired ObsID/instrument, and the same settings you used to generate the spectrum, and the
        spectra(um) will be provided to you. If no match is found then a NoProductAvailableError will be raised.

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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all spectra matching the other criteria will be returned.
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

        if obs_id == 'combined':
            matched_prods = self.get_products('combined_spectrum', obs_id=obs_id, inst=inst, extra_key=spec_storage_name,
                                          telescope=telescope)
        else:
            matched_prods = self.get_products('spectrum', obs_id=obs_id, inst=inst, extra_key=spec_storage_name,
                                          telescope=telescope)

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any spectra matching your input.")

        return matched_prods   


    def get_spectra(self, outer_radius: Union[str, Quantity], obs_id: str = None, inst: str = None,
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                    min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                    telescope: str = None) -> Union[Spectrum, List[Spectrum]]:
        """
        A useful method that wraps the get_products function to allow you to easily retrieve XGA Spectrum objects.
        Simply pass the desired ObsID/instrument, and the same settings you used to generate the spectrum, and the
        spectra(um) will be provided to you. If no match is found then a NoProductAvailableError will be raised.

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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all spectra matching the other criteria will be returned.
        :return: An XGA Spectrum object (if there is an exact match), or a list of XGA Spectrum objects (if there
            were multiple matching products).
        :rtype: Union[Spectrum, List[Spectrum]]
        """
        
        matched_prods = self._get_spec_prod(outer_radius, obs_id, inst, inner_radius, group_spec,
                                               min_counts, min_sn, over_sample, telescope)

        return matched_prods

    def get_combined_spectra(self, outer_radius: Union[str, Quantity], inst: str = None,
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), group_spec: bool = True,
                    min_counts: int = 5, min_sn: float = None, over_sample: float = None,
                    telescope: str = None) -> Union[Spectrum, List[Spectrum]]:
        """
        A useful method that wraps the get_products function to allow you to easily retrieve XGA Spectrum objects.
        Simply pass the desired ObsID/instrument, and the same settings you used to generate the spectrum, and the
        spectra(um) will be provided to you. If no match is found then a NoProductAvailableError will be raised.

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
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all spectra matching the other criteria will be returned.
        :return: An XGA Spectrum object (if there is an exact match), or a list of XGA Spectrum objects (if there
            were multiple matching products).
        :rtype: Union[Spectrum, List[Spectrum]]
        """

        if telescope == 'xmm':
            raise NotImplementedError("Combined spectra are not implemented for XMM observations.")
        
        matched_prods = self._get_spec_prod(outer_radius, 'combined', inst, inner_radius, group_spec,
                                               min_counts, min_sn, over_sample, telescope)

        return matched_prods

    def get_annular_spectra(self, radii: Quantity = None, group_spec: bool = True, min_counts: int = 5,
                            min_sn: float = None, over_sample: float = None, set_id: int = None,
                            telescope: str = None) -> AnnularSpectra:
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
        :param str telescope: Optionally, a specific telescope to search for annular spectra can be supplied. The
            default is None, which means all annular spectra matching the other criteria will be returned.
        :return: An XGA AnnularSpectra object if there is an exact match.
        :rtype: AnnularSpectra
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
            matched_prods = self.get_products('combined_spectrum', extra_key=spec_storage_name, telescope=telescope)
        # But if the user hasn't passed an ID AND the radii are None then we look for partial matches
        elif set_id is None and radii is None:
            matched_prods = [p for p in self.get_products('combined_spectrum', telescope=telescope)
                             if spec_storage_name[0] in p.storage_key and spec_storage_name[1] in p.storage_key]
        # However if they have passed a setID then this over-rides everything else
        else:
            # With the set ID we fetch ALL annular spectra, then use their set_id property to match against
            #  whatever the user passed in
            matched_prods = [p for p in self.get_products('combined_spectrum', telescope=telescope)
                             if p.set_ident == set_id]

        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("No matching AnnularSpectra can be found.")

        return matched_prods

    def get_profiles(self, profile_type: str, obs_id: str = None, inst: str = None, central_coord: Quantity = None,
                     radii: Quantity = None, lo_en: Quantity = None, hi_en: Quantity = None,
                     telescope: str = None) -> Union[BaseProfile1D, List[BaseProfile1D]]:
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
        :param Quantity lo_en: The lower energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed hi_en must be too.
        :param Quantity hi_en: The higher energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed lo_en must be too.
        :param str telescope: Optionally, a specific telescope to search for can be supplied. The default is None,
            which means all profiles matching the other criteria will be returned.
        :return: An XGA profile object (if there is an exact match), or a list of XGA profile objects (if there
            were multiple matching products).
        :rtype: Union[BaseProfile1D, List[BaseProfile1D]]
        """
        if "profile" in profile_type:
            warn("The profile_type you passed contains the word 'profile', which is appended onto a profile type "
                 "by XGA, you need to try this again without profile on the end, unless you gave a generic "
                 "profile a type with 'profile' in.", stacklevel=2)

        search_key = profile_type + "_profile"
        if search_key not in ALLOWED_PRODUCTS:
            warn("{} seems to be a custom profile, not an XGA default type. If this is not true then you have "
                 "passed an invalid profile type.".format(search_key), stacklevel=2)

        matched_prods = self._get_prof_prod(search_key, obs_id, inst, central_coord, radii, lo_en, hi_en, telescope)
        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any {p} profiles matching your input.".format(p=profile_type))

        return matched_prods

    def get_combined_profiles(self, profile_type: str, central_coord: Quantity = None, radii: Quantity = None,
                              lo_en: Quantity = None, hi_en: Quantity = None, telescope: str = None) \
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
        :param Quantity lo_en: The lower energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed hi_en must be too.
        :param Quantity hi_en: The higher energy bound of the profile you wish to retrieve (if applicable), default
            is None, and if this argument is passed lo_en must be too.
        :param str telescope: Optionally, a specific telescope to search for combined profiles can be supplied. The
            default is None, which means all combined profiles matching the other criteria will be returned.
        :return: An XGA profile object (if there is an exact match), or a list of XGA profile objects (if there
            were multiple matching products).
        :rtype: Union[BaseProfile1D, List[BaseProfile1D]]
        """
        if "profile" in profile_type:
            warn("The profile_type you passed contains the word 'profile', which is appended onto a profile type "
                 "by XGA, you need to try this again without profile on the end, unless you gave a generic profile "
                 "a type with 'profile' in.", stacklevel=2)

        search_key = "combined_" + profile_type + "_profile"

        if search_key not in ALLOWED_PRODUCTS:
            warn("That profile type seems to be a custom profile, not an XGA default type. If this is not true "
                 "then you have passed an invalid profile type.", stacklevel=2)

        matched_prods = self._get_prof_prod(search_key, None, None, central_coord, radii, lo_en, hi_en, telescope)
        if len(matched_prods) == 1:
            matched_prods = matched_prods[0]
        elif len(matched_prods) == 0:
            raise NoProductAvailableError("Cannot find any combined {p} profiles matching your "
                                          "input.".format(p=profile_type))

        return matched_prods

    def get_lightcurves(self, outer_radius: Union[str, Quantity] = None, obs_id: str = None, inst: str = None,
                        inner_radius: Union[str, Quantity] = None, lo_en: Quantity = None,
                        hi_en: Quantity = None, time_bin_size: Quantity = None, pattern: Union[dict, str] = 'default',
                        telescope: str = None) -> Union[LightCurve, List[LightCurve]]:
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
        :param str telescope: Optionally, a specific telescope to search for lightcurves can be supplied. The
            default is None, which means all lightcurves matching the other criteria will be returned.
        :return: An XGA LightCurve object (if there is an exact match), or a list of XGA LightCurve objects (if there
            were multiple matching products).
        :rtype: Union[LightCurve, List[LightCurve]]
        """
        from ..generate.common import check_pattern

        if telescope is None and pattern == 'default' and len(self.telescopes) != 1:
            warn("Can't use the 'default' pattern argument value when 'telescope' is None and there is more than"
                 " one telescope associated with the source - this is because different telescopes use different"
                 " pattern styles. The 'pattern' argument value has been set to None.", stacklevel=2)
            pattern = None
        elif telescope is None and pattern == 'default' and len(self.telescopes) == 1:
            telescope = self.telescopes[0]

        # This is where we set up the search string for the patterns specified by the user.
        if pattern is None:
            patt_search = "_pattern"
        elif isinstance(pattern, str):
            pattern = {'xmm': {'pn': '<=4', 'mos': '<=12'}, 'erosita': {'tm': '15', 'combined': '15'}}
            patt_search = {inst: "_pattern" + check_pattern(patt, telescope)[1]
                           for inst, patt in pattern[telescope].items()}
        elif isinstance(pattern, dict):
            if ('mos1' in list(pattern.keys()) or 'mos2' in list(pattern.keys())
                    or any(['tm{}'.format(tm_i) in list(pattern.keys()) for tm_i in range(0, 7)])):
                raise ValueError("Specific MOS/TM instruments do not need to be specified for 'pattern'; i.e. there "
                                 "should be one entry for 'mos' or 'tm'.")
            pattern = {inst: patt.replace(' ', '') for inst, patt in pattern.items()}
            patt_search = {inst: "_pattern" + check_pattern(patt, telescope)[1] for inst, patt in pattern.items()}
        else:
            raise TypeError("The 'pattern' argument must be either 'default', or a dictionary where the keys are "
                            "instrument names and values are string patterns.")

        # Just makes the search easier down the line
        if 'mos' in patt_search:
            patt_search.update({'mos1': patt_search['mos'], 'mos2': patt_search['mos']})

        if 'tm' in patt_search:
            patt_search.update({'tm{}'.format(tm_i): patt_search['tm'] for tm_i in range(0, 7)})

        some_lcs = self._get_lc_prod(outer_radius, obs_id, inst, inner_radius, lo_en, hi_en, time_bin_size, telescope)
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
                                 pattern: Union[dict, str] = "default", telescope: str = None) \
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
        :param str telescope: Optionally, a specific telescope to search for lightcurves can be supplied. The
            default is None, which means all lightcurves matching the other criteria will be returned.
        :return: An XGA AggregateLightCurve object (if there is an exact match), or a list of XGA AggregateLightCurve
            objects (if there were multiple matching products).
        :rtype: Union[AggregateLightCurve, List[AggregateLightCurve]]
        """
        if telescope == 'xmm':
            from ..generate.common import check_pattern
        else:
            raise NotImplementedError("Support for other telescopes has not yet been added to get_lightcurves")

        # TODO SO THIS IS THE LAST GET METHOD THAT NEEDS CONVERTING TO SUPPORT DIFFERENT TELESCOPES - HOWEVER THAT IS
        #  COMPLICATED BY THE FACT THAT WE DON'T CURRENTLY AUTO-CREATE AGGREGATE LIGHTCURVES, AND THEY MIGHT BE THE
        #  ONE THING THAT IS ALREADY ALLOWED TO BE MULTI-TELESCOPE

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

    def get_att_file(self, obs_id: str, telescope: str) -> str:
        """
        Fetches the path to the attitude file for an observation associated with this source.

        :param str obs_id: The ObsID to fetch the attitude file for.
        :param str telescope: The telescope to fetch the attitude file for.
        :return: The path to the attitude file.
        :rtype: str
        """
        if telescope not in self.telescopes:
            raise NotAssociatedError("The {t} telescope is not associated with {n}.".format(t=telescope, n=self.name))

        elif obs_id not in self.obs_ids[telescope]:
            raise NotAssociatedError("{t}-{o} is not associated with {s}".format(t=telescope, o=obs_id, s=self.name))
        else:
            return self._att_files[telescope][obs_id]

    def source_back_regions(self, reg_type: str, telescope: str, obs_id: str = None,
                            central_coord: Quantity = None) -> Tuple[SkyRegion, SkyRegion]:
        """
        A method to retrieve source region and background region objects for a given source type with a
        given central coordinate.

        :param str reg_type: The type of region which we wish to get from the source.
        :param str telescope: The telescope that the region is associated with - this needs to be supplied to
            retrieve an image to convert source regions between pixel and sky coordinates.
        :param str obs_id: The ObsID that the region is associated with (if appropriate).
        :param Quantity central_coord: The central coordinate of the region.
        :return: The method returns both the source region and the associated background region.
        :rtype: Tuple[SkyRegion, SkyRegion]
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
        elif telescope not in self.telescopes:
            raise NotAssociatedError("The telescope {t} is not associated with {s}.".format(t=telescope, s=self.name))
        elif obs_id is not None and obs_id not in self.obs_ids[telescope]:
            raise NotAssociatedError("The ObsID {t}-{o} is not associated with {s}.".format(t=telescope, o=obs_id,
                                                                                            s=self.name))
        elif reg_type not in allowed_rtype:
            raise ValueError("The only allowed region types are {}".format(", ".join(allowed_rtype)))
        elif reg_type == "region" and obs_id is None:
            raise ValueError("ObsID and telescope cannot be None when getting region file regions.")
        elif reg_type == "region" and obs_id is not None:
            # TODO Do I even still use this attribute?
            src_reg = self._regions[telescope][obs_id]
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
        # TODO ALSO DOING THIS WITH MULTI TELESCOPES MAKES ME NERVOUS
        im = self.get_products("image", telescope=telescope)[0]
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

    def within_region(self, region: SkyRegion, telescope: str) -> List[SkyRegion]:
        """
        This method finds contaminating sources (detected by the specified telescope) that lie within the user
        supplied region.

        :param SkyRegion region: The region in which we wish to search for interloper sources (for instance
            a source region or background region).
        :param str telescope: The telescope whose regions we should search.
        :return: A list of regions that lie within the user supplied region.
        :rtype: List[SkyRegion]
        """
        im = self.get_products("image", telescope=telescope)[0]

        crossover = np.array([region.intersection(r).to_pixel(im.radec_wcs).to_mask().data.sum() != 0
                              for r in self._interloper_regions[telescope]])
        reg_within = np.array(self._interloper_regions[telescope])[crossover]

        return reg_within

    def get_interloper_regions(self, telescope: str, flattened: bool = False) -> Union[List, Dict]:
        """
        This get method provides a way to access the regions that have been designated as interlopers (i.e.
        not the source region that a particular Source has been designated to investigate) for all observations.
        They can either be retrieved in a dictionary with ObsIDs as the keys, or a flattened single list with no
        ObsID context.

        :param str telescope: The telescope for which we wish to retrieve regions.
        :param bool flattened: If true then the regions are returned as a single list of region objects. Otherwise
            they are returned as a dictionary with ObsIDs as keys. Default is False.
        :return: Either a list of region objects, or a dictionary with ObsIDs as keys.
        :rtype: Union[List,Dict]
        """
        if type(self) == BaseSource:
            raise TypeError("BaseSource objects don't have enough information to know which sources "
                            "are interlopers.")
        elif telescope not in self.telescopes:
            raise NotAssociatedError("The {t} telescope is not associated with {n}.".format(t=telescope, n=self.name))

        # If flattened then a list is returned rather than the original dictionary with
        if not flattened:
            ret_reg = self._other_regions[telescope]
        else:
            # Iterate through the ObsIDs in the dictionary and add the resulting lists together
            ret_reg = []
            for o in self._other_regions[telescope]:
                ret_reg += self._other_regions[telescope][o]

        return ret_reg

    def get_source_mask(self, reg_type: str, telescope: str, obs_id: str = None,
                        central_coord: Quantity = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to retrieve source and background masks for the given region type.

        :param str reg_type: The type of region for which to retrieve the mask.
        :param str telescope: The telescope for which to retrieve the mask.
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
        src_reg, bck_reg = self.source_back_regions(reg_type, telescope, obs_id, central_coord)

        # I assume that if no ObsID is supplied, then the user wishes to have a mask for the combined data
        if obs_id is None:
            comb_images = self.get_products("combined_image", telescope=telescope)
            if len(comb_images) != 0:
                mask_image = comb_images[0]
            else:
                raise NoProductAvailableError("There are no combined products available to generate a mask for.")
        else:
            # Just grab the first instrument that comes out the get method, the masks should be the same.
            mask_image = self.get_products("image", obs_id, telescope=telescope)[0]

        mask = src_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)
        back_mask = bck_reg.to_pixel(mask_image.radec_wcs).to_mask().to_image(mask_image.shape)

        # If the masks are None, then they are set to an array of zeros
        if mask is None:
            mask = np.zeros(mask_image.shape)
        if back_mask is None:
            back_mask = np.zeros(mask_image.shape)

        return mask, back_mask

    def get_interloper_mask(self, telescope: str, obs_id: str = None, region_distance: Quantity = None) -> ndarray:
        """
        Returns a mask for a given ObsID (or combined data if no ObsID given) that will remove any sources
        that have not been identified as the source of interest.

        :param str telescope: The telescope for which to retrieve the mask.
        :param str obs_id: The ObsID that the mask is associated with (if appropriate).
        :param Quantity region_distance: The distance from the central coordinate within which contaminating regions
            will be included in this mask. Introduced as a counter to the very large numbers of regions associated
            with eROSITA observations. Default is None, in which case all contaminating regions will be included in
            the mask.
        :return: A numpy array of 0s and 1s which acts as a mask to remove interloper sources.
        :rtype: ndarray
        """
        # TODO I HAVE MADE IT SO THAT INTERLOPER MASKS ARE GENERATED ON DEMAND, RATHER THAN JUST AS A SOURCE
        #  IS DECLARED. I HOPE THAT THIS WILL MAKE THE DECLARATION OF SOURCES WITH EROSITA DATA CONSIDERABLY FASTER.
        #  HOWEVER I STILL HAVE NOT MADE USE OF THE NEW REGION DISTANCE FEATURE ANYWHERE YET, SO THE NEXT JOB IS TO
        #  SET A MAXIMUM DISTANCE FOR MASK GENERATION (FOR EROSITA ONLY AT THE MOMENT MAYBE?) THAT WILL SPEED UP THE
        #  PROCESS OF GENERATING THE MASK WHEN IT IS ACTUALLY NEEDED

        # If ObsID is None then we take it that the user wants the combined mask for the specified telescope (combined
        #  as in all contaminating regions from all ObsIDs for that telescope removed). As such we change the value
        #  of the variable here, making the rest of the function a little neater
        if obs_id is None:
            obs_id = 'combined'

        # We check that the user hasn't requested a telescope or ObsID that just isn't associated with this source
        if telescope not in self.telescopes:
            raise NotAssociatedError("Telescope {t} is not associated with {s}; only {a} are "
                                     "available.".format(t=telescope, s=self.name, a=", ".join(self.telescopes)))
        elif obs_id is not None and obs_id != "combined" and obs_id not in self.obs_ids[telescope]:
            raise NotAssociatedError("ObsID {o} is not associated with {s}; only {a} are "
                                     "available".format(o=obs_id, s=self.name,
                                                        a=", ".join(self.obs_ids[telescope])))

        # Check whether an acceptable 'region_distance' argument has been passed.
        if region_distance is not None and (not region_distance.unit.is_equivalent('deg') and
                                            not region_distance.unit.is_equivalent('kpc')):
            raise UnitConversionError("The 'region_distance' argument must be supplied in units that are "
                                      "convertible to degrees or kpc.")
        elif region_distance is not None:
            region_distance = self.convert_radius(region_distance, 'deg').value
            # The reg_dist_key will be used to store the mask generated for this distance, so that if it is asked for
            #  again we won't have to regenerate it
            reg_dist_key = region_distance
        elif region_distance is None:
            # The reg_dist_key value of 'all' will supersede all others, if it exists then that is what will be used
            reg_dist_key = 'all'

        # Check that we aren't a BaseSource, as that won't have any contaminating regions defined
        if self is BaseSource:
            raise TypeError("BaseSource objects don't have enough information to know which sources "
                            "are interlopers.")

        # Here we get to the business of retrieving or generating/storing masks
        # The first case we deal with is where the requested mask doesn't exist, and nor does an 'all' mask, which
        #  is essentially a master interloper mask where no maximum inclusion distance (set by region_distance) was
        #  specified. We will always use the master mask if it exists, but we'll generate the one specified by the
        #  region distance in this case
        if (reg_dist_key not in self._interloper_masks[telescope][obs_id] and
                'all' not in self._interloper_masks[telescope][obs_id]):

            # Grab an image based on whether the ObsID is combined or specific
            if obs_id == 'combined':
                im = self.get_combined_images(telescope=telescope)
            else:
                im = self.get_images(obs_id, telescope=telescope)

            if isinstance(im, list):
                im = im[0]

            # Generate the mask as instructed, with the specified region_distance (which could well be None, which
            #  would result in a 'master mask'
            mask = self._generate_interloper_mask(im, region_distance)
            # Then we store the generated mask for later, in case the same one is requested
            self._interloper_masks[telescope][obs_id][reg_dist_key] = mask

        # If a master mask is present, we'll just use that one
        elif 'all' in self._interloper_masks[telescope][obs_id]:
            mask = self._interloper_masks[telescope][obs_id]['all']
        # And failing that, the specific mask that has been requested (specified by the region_distance argument)
        #  must already exist, so we grab that and serve it to the user.
        else:
            mask = self._interloper_masks[telescope][obs_id][reg_dist_key]

        return mask

    def get_mask(self, reg_type: str, telescope: str, obs_id: str = None,
                 central_coord: Quantity = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Method to retrieve source and background masks for the given region type, WITH INTERLOPERS REMOVED.

        :param str reg_type: The type of region for which to retrieve the interloper corrected mask.
        :param str telescope: The telescope for which to retrieve the mask.
        :param str obs_id: The ObsID that the mask is associated with (if appropriate).
        :param Quantity central_coord: The central coordinate of the region.
        :return: The source and background masks for the requested ObsID (or the combined image if no ObsID).
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # Grabs the source masks without interlopers removed
        src_mask, bck_mask = self.get_source_mask(reg_type, telescope, obs_id, central_coord)
        # Grabs the interloper mask
        interloper_mask = self.get_interloper_mask(telescope, obs_id)

        # Multiplies the uncorrected source and background masks with the interloper masks to correct
        #  for interloper sources
        total_src_mask = src_mask * interloper_mask
        total_bck_mask = bck_mask * interloper_mask

        return total_src_mask, total_bck_mask

    def get_custom_mask(self, outer_rad: Quantity, telescope: str, inner_rad: Quantity = Quantity(0, 'arcsec'),
                        obs_id: str = None, central_coord: Quantity = None,
                        remove_interlopers: bool = True) -> np.ndarray:
        """
        A simple, but powerful method, to generate mask a mask within a custom radius for a given ObsID.

        :param Quantity outer_rad: The outer radius of the mask.
        :param str telescope: The telescope for which to retrieve the mask.
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
            # Doesn't matter which combined images, just need the size and coord conversion powers
            im = self.get_combined_images(telescope=telescope)
        else:
            # Again so long as the image matches the ObsID passed in by the user I don't care what instrument
            #  its from
            im = self.get_images(obs_id=obs_id, telescope=telescope)

        # If it's not an instance of Image that means a list of Images has been returned, and as I only want
        #  the WCS information and the shape of the image I don't care which one we use
        if not isinstance(im, Image):
            im = im[0]

        # Convert the inner and outer radii to degrees, so they can be easily converted to pixels
        outer_rad = self.convert_radius(outer_rad, 'deg')
        inner_rad = self.convert_radius(inner_rad, 'deg')
        pix_to_deg = pix_deg_scale(central_coord, im.radec_wcs)

        # Making sure the inner and outer radii are whole integer numbers, as they are now in pixel units
        outer_rad = np.array([int(np.ceil(outer_rad / pix_to_deg).value)])
        inner_rad = np.array([int(np.floor(inner_rad / pix_to_deg).value)])
        # Convert the chosen central coordinates to pixels
        pix_centre = im.coord_conv(central_coord, 'pix')

        # Generate our custom mask
        custom_mask = annular_mask(pix_centre, inner_rad, outer_rad, im.shape)

        # And applying an interloper mask if the user wants that.
        if remove_interlopers:
            interloper_mask = self.get_interloper_mask(telescope, obs_id)
            custom_mask = custom_mask*interloper_mask
        return custom_mask

    def get_snr(self, outer_radius: Union[Quantity, str], telescope: str, central_coord: Quantity = None,
                lo_en: Quantity = None, hi_en: Quantity = None, obs_id: str = None, inst: str = None,
                psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                psf_iter: int = 15, allow_negative: bool = False, exp_corr: bool = True) -> float:
        """
        This takes a region type and central coordinate and calculates the signal-to-noise ratio.
        The background region is constructed using the back_inn_rad_factor and back_out_rad_factor
        values, the defaults of which are 1.05*radius and 1.5*radius respectively.

        :param Quantity/str outer_radius: The radius that SNR should be calculated within, this can either be a
            named radius such as r500, or an astropy Quantity.
        :param str telescope: The telescope which, when combined with the ObsID, we wish to calculate the
            signal-to-noise for.
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
            zero, which results in a lower signal-to-noise (and can result in a negative signal-to-noise).
        :param bool exp_corr: Should signal to noises be measured with exposure time correction, default is True. I
            recommend that this be true for combined observations, as exposure time could change quite dramatically
            across the combined product.
        :return: The signal-to-noise ratio.
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
            rt = self.get_combined_ratemaps(lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter,
                                            telescope=telescope)

        elif all([obs_id is not None, inst is not None]):
            # Both ObsID and instrument have been set by the user
            rt = self.get_ratemaps(obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter,
                                   telescope=telescope)
        else:
            raise ValueError("If you wish to use a specific ratemap for {s}'s signal to noise calculation, please "
                             " pass both obs_id and inst.".format(s=self.name))

        if isinstance(outer_radius, str):
            # Grabs the interloper removed source and background region masks. If the ObsID is None the get_mask
            #  method understands that means it should return the mask for the combined data
            src_mask, bck_mask = self.get_mask(outer_radius, telescope, obs_id, central_coord)
        else:
            # Here we have the case where the user has passed a custom outer radius, so we need to generate a
            #  custom mask for it
            src_mask = self.get_custom_mask(outer_radius, telescope, obs_id=obs_id, central_coord=central_coord)
            bck_mask = self.get_custom_mask(outer_radius * self._back_out_factor, telescope,
                                            outer_radius * self._back_inn_factor, obs_id=obs_id,
                                            central_coord=central_coord)

        # We use the ratemap's built in signal to noise calculation method
        sn = rt.signal_to_noise(src_mask, bck_mask, exp_corr, allow_negative)

        return sn

    def get_counts(self, outer_radius: Union[Quantity, str], telescope: str, central_coord: Quantity = None,
                   lo_en: Quantity = None, hi_en: Quantity = None, obs_id: str = None, inst: str = None,
                   psf_corr: bool = False, psf_model: str = "ELLBETA", psf_bins: int = 4, psf_algo: str = "rl",
                   psf_iter: int = 15) -> Quantity:
        """
        This takes a region type and central coordinate and calculates the background subtracted X-ray counts.
        The background region is constructed using the back_inn_rad_factor and back_out_rad_factor
        values, the defaults of which are 1.05*radius and 1.5*radius respectively.

        :param Quantity/str outer_radius: The radius that counts should be calculated within, this can either be a
            named radius such as r500, or an astropy Quantity.
        :param str telescope: The telescope which, when combined with the ObsID, we wish to calculate the
            counts for.
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
            rt = self.get_combined_ratemaps(lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter,
                                            telescope=telescope)

        elif all([obs_id is not None, inst is not None]):
            # Both ObsID and instrument have been set by the user
            rt = self.get_ratemaps(obs_id, inst, lo_en, hi_en, psf_corr, psf_model, psf_bins, psf_algo, psf_iter,
                                   telescope=telescope)
        else:
            raise ValueError("If you wish to use a specific ratemap for {s}'s signal to noise calculation, please "
                             " pass both obs_id and inst.".format(s=self.name))

        if isinstance(outer_radius, str):
            # Grabs the interloper removed source and background region masks. If the ObsID is None the get_mask
            #  method understands that means it should return the mask for the combined data
            src_mask, bck_mask = self.get_mask(outer_radius, telescope, obs_id, central_coord)
        else:
            # Here we have the case where the user has passed a custom outer radius, so we need to generate a
            #  custom mask for it
            src_mask = self.get_custom_mask(outer_radius, telescope, obs_id=obs_id, central_coord=central_coord)
            bck_mask = self.get_custom_mask(outer_radius * self._back_out_factor, telescope,
                                            outer_radius * self._back_inn_factor, obs_id=obs_id,
                                            central_coord=central_coord)

        # We use the ratemap's built in background subtracted counts calculation method
        cnts = rt.background_subtracted_counts(src_mask, bck_mask)

        return cnts

    def regions_within_radii(self, inner_radius: Quantity, outer_radius: Quantity, telescope: str,
                             deg_central_coord: Quantity,
                             regions_to_search: Union[np.ndarray, list] = None) -> np.ndarray:
        """
        This function finds and returns any interloper regions (by default) that have any part of their boundary
        within the specified radii, centered on the specified central coordinate. Users may also pass their own
        array of regions to check.

        :param Quantity inner_radius: The inner radius of the area to search for interlopers in.
        :param Quantity outer_radius: The outer radius of the area to search for interlopers in.
        :param str telescope: The telescope for which we wish to retrieve regions within the specified radii.
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

        if telescope not in self.telescopes:
            raise NotAssociatedError("The {t} telescope is not associated with {n}.".format(t=telescope, n=self.name))

        # If no custom regions array was passed, we use the internal array of interloper regions
        if regions_to_search is None:
            regions_to_search = self._interloper_regions[telescope].copy()

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
        # TODO Again this doesn't really belong in the BaseSource class - it is SAS specific so should be moved, but
        #  not right now!
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
        rel_im = self.get_products("image", obs_id, inst, telescope='xmm')[0]
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
            interloper_regions = self.regions_within_radii(inner_radius, outer_radius, 'xmm', deg_central_coord)
        elif interloper_regions is None and not inner_radius.isscalar:
            interloper_regions = self.regions_within_radii(min(inner_radius), max(outer_radius), 'xmm',
                                                           deg_central_coord)

        # So now we convert our interloper regions into their SAS equivalents
        sas_interloper = [self._interloper_sas_string(i, rel_im, output_unit) for i in interloper_regions]

        if inner_radius.isscalar and inner_radius.value != 0:
            # And we need to define a SAS string for the actual region of interest
            sas_source_area = "(({t}) IN annulus({cx},{cy},{ri},{ro}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].value,
                                                     cy=xmm_central_coord[1].value, ri=inner_radius.value/sky_to_deg,
                                                     ro=outer_radius.value/sky_to_deg)
        # If the inner radius is zero then we write a circle region, because it seems that's a LOT faster in SAS
        elif inner_radius.isscalar and inner_radius.value == 0:
            sas_source_area = "(({t}) IN circle({cx},{cy},{r}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].value,
                                                     cy=xmm_central_coord[1].value,
                                                     r=outer_radius.value/sky_to_deg)
        elif not inner_radius.isscalar and inner_radius[0].value != 0:
            sas_source_area = "(({t}) IN elliptannulus({cx},{cy},{wi},{hi},{wo},{ho},{rot},{rot}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].value,
                                                     cy=xmm_central_coord[1].value,
                                                     wi=inner_radius[0].value/sky_to_deg,
                                                     hi=inner_radius[1].value/sky_to_deg,
                                                     wo=outer_radius[0].value/sky_to_deg,
                                                     ho=outer_radius[1].value/sky_to_deg, rot=rot_angle.to('deg').value)
        elif not inner_radius.isscalar and inner_radius[0].value == 0:
            sas_source_area = "(({t}) IN ellipse({cx},{cy},{w},{h},{rot}))"
            sas_source_area = sas_source_area.format(t=c_str, cx=xmm_central_coord[0].value,
                                                     cy=xmm_central_coord[1].value,
                                                     w=outer_radius[0].value / sky_to_deg,
                                                     h=outer_radius[1].value / sky_to_deg,
                                                     rot=rot_angle.to('deg').value)

        # Combining the source region with the regions we need to cut out
        if len(sas_interloper) == 0:
            final_src = sas_source_area
        else:
            final_src = sas_source_area + " &&! " + " &&! ".join(sas_interloper)

        return final_src

    def add_fit_data(self, model: str, tab_line, lums: dict, spec_storage_key: str, telescope: str):
        """
        A method that stores fit results and global information about a set of spectra in a source object.
        Any variable parameters in the fit are stored in an internal dictionary structure, as are any luminosities
        calculated. Other parameters of interest are store in other internal attributes. This probably shouldn't
        ever be used by the user, just other parts of XGA, hence why I've asked for a spec_storage_key to be passed
        in rather than all the spectrum configuration options individually.

        :param str model: The XSPEC definition of the model used to perform the fit. e.g. constant*tbabs*apec
        :param tab_line: The table line with the fit data.
        :param dict lums: The various luminosities measured during the fit.
        :param str spec_storage_key: The storage key of any spectrum that was used in this particular fit. The
            ObsID and instrument used don't matter, as the storage key will be the same and is based off of the
            settings when the spectra were generated.
        :param str telescope: The telescope for which the spectra that have been fit were generated.
        """
        # Just headers that will always be present in tab_line that are not fit parameters
        not_par = ['MODEL', 'TOTAL_EXPOSURE', 'TOTAL_COUNT_RATE', 'TOTAL_COUNT_RATE_ERR',
                   'NUM_UNLINKED_THAWED_VARS', 'FIT_STATISTIC', 'TEST_STATISTIC', 'DOF']

        if telescope not in self.telescopes:
            raise NotAssociatedError("The {t} telescope is not associated with {n}.".format(t=telescope, n=self.name))

        # Various global values of interest
        self._total_exp[telescope][spec_storage_key] = float(tab_line["TOTAL_EXPOSURE"])
        if spec_storage_key not in self._total_count_rate[telescope]:
            self._total_count_rate[telescope][spec_storage_key] = {}
            self._test_stat[telescope][spec_storage_key] = {}
            self._dof[telescope][spec_storage_key] = {}
        self._total_count_rate[telescope][spec_storage_key][model] = [float(tab_line["TOTAL_COUNT_RATE"]),
                                                                      float(tab_line["TOTAL_COUNT_RATE_ERR"])]
        self._test_stat[telescope][spec_storage_key][model] = float(tab_line["TEST_STATISTIC"])
        self._dof[telescope][spec_storage_key][model] = float(tab_line["DOF"])

        # The parameters available will obviously be dynamic, so have to find out what they are and then
        #  for each result find the +- errors
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
        if spec_storage_key not in self._fit_results[telescope]:
            self._fit_results[telescope][spec_storage_key] = {}
        self._fit_results[telescope][spec_storage_key][model] = mod_res

        # And now storing the luminosity results
        if spec_storage_key not in self._luminosities[telescope]:
            self._luminosities[telescope][spec_storage_key] = {}
        self._luminosities[telescope][spec_storage_key][model] = lums

    def get_results(self, outer_radius: Union[str, Quantity], telescope: str, model: str,
                    inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), par: str = None,
                    group_spec: bool = True, min_counts: int = 5, min_sn: float = None, over_sample: float = None):
        """
        Important method that will retrieve fit results from the source object. Either for a specific
        parameter of a given region-model combination, or for all of them. If a specific parameter is requested,
        all matching values from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+). If no parameter is specified, the return will be a dictionary
        of such numpy arrays, with the keys corresponding to parameter names.

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in
            region files), then any inner radius will be ignored.
        :param str telescope: The telescope for which to retrieve spectral fit results.
        :param str model: The name of the fitted model that you're requesting the results
            from (e.g. constant*tbabs*apec).
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default, this is zero arcseconds, resulting in a
            circular spectrum.
        :param str par: The name of the parameter you want a result for.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal-to-noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal-to-noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :return: The requested result value, and uncertainties.
        """
        # TODO the refactoring got a bit dicey - take a look around and make sure I didn't destroy anything
        # First I want to retrieve the spectra that were fitted to produce the result they're looking for,
        #  because then I can just grab the storage key from one of them
        specs = self.get_spectra(outer_radius, None, None, inner_radius, group_spec, min_counts, min_sn, over_sample,
                                 telescope=telescope)
        # I just take the first spectrum in the list because the storage key will be the same for all of them
        if isinstance(specs, list):
            storage_key = specs[0].storage_key
        else:
            storage_key = specs.storage_key

        # Bunch of checks to make sure the requested results actually exist
        if len(self._fit_results[telescope]) == 0:
            raise ModelNotAssociatedError("There are no {t} XSPEC fits associated with {s}".format(t=telescope,
                                                                                                   s=self.name))
        elif storage_key not in self._fit_results[telescope]:
            raise ModelNotAssociatedError("Those {t} spectra have no associated XSPEC fit to {s}".format(t=telescope,
                                                                                                         s=self.name))
        elif model not in self._fit_results[telescope][storage_key]:
            av_mods = ", ".join(self._fit_results[telescope][storage_key].keys())
            raise ModelNotAssociatedError("{m} has not been fitted to those {t} spectra of {s}; available "
                                          "models are {a}".format(m=model, s=self.name, a=av_mods, t=telescope))
        elif par is not None and par not in self._fit_results[telescope][storage_key][model]:
            av_pars = ", ".join(self._fit_results[telescope][storage_key][model].keys())
            raise ParameterNotAssociatedError("{p} was not a free parameter in the {m} fit to those {t} spectra of "
                                              "{s}, the options are {a}".format(p=par, m=model, s=self.name, a=av_pars,
                                                                                t=telescope))

        # Read out into variable for readabilities sake
        fit_data = self._fit_results[telescope][storage_key][model]
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

    def get_luminosities(self, outer_radius: Union[str, Quantity], telescope: str, model: str,
                         inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = None,
                         hi_en: Quantity = None, group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
                         over_sample: float = None):
        """
        Get method for luminosities calculated from model fits to spectra associated with this source.
        Either for given energy limits (that must have been specified when the fit was first performed), or
        for all luminosities associated with that model. Luminosities are returned as a 3 column numpy array;
        the 0th column is the value, the 1st column is the err-, and the 2nd is err+.

        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). If 'region' is chosen (to use the regions in
            region files), then any inner radius will be ignored.
        :param str telescope: The telescope for which to retrieve spectral fit luminosities.
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
        :param float min_sn: The minimum signal-to-noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal-to-noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :return: The requested luminosity value, and uncertainties.
        """
        # First I want to retrieve the spectra that were fitted to produce the result they're looking for,
        #  because then I can just grab the storage key from one of them
        specs = self.get_spectra(outer_radius, None, None, inner_radius, group_spec, min_counts, min_sn, over_sample,
                                 telescope=telescope)
        # I just take the first spectrum in the list because the storage key will be the same for all of them
        if isinstance(specs, list):
            storage_key = specs[0].storage_key
        else:
            storage_key = specs.storage_key

        # Checking the input energy limits are valid, and assembles the key to look for lums in those energy
        #  bounds. If the limits are none then so is the energy key
        if lo_en is not None and hi_en is not None and lo_en > hi_en:
            raise ValueError("The low energy limit cannot be greater than the high energy limit")
        elif lo_en is not None and hi_en is not None:
            en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        else:
            en_key = None

        # Checks that the requested region, model and energy band actually exist
        if len(self._luminosities[telescope]) == 0:
            raise ModelNotAssociatedError("There are no {t} XSPEC fits associated with {s}".format(s=self.name,
                                                                                                   t=telescope))
        elif storage_key not in self._luminosities[telescope]:
            raise ModelNotAssociatedError("These {t} spectra have no associated XSPEC fit to {s}.".format(t=telescope,
                                                                                                          s=self.name))
        elif model not in self._luminosities[telescope][storage_key]:
            av_mods = ", ".join(self._luminosities[storage_key].keys())
            raise ModelNotAssociatedError("{m} has not been fitted to these {t} spectra of {s}; "
                                          "available models are {a}".format(m=model, s=self.name, a=av_mods,
                                                                            t=telescope))
        elif en_key is not None and en_key not in self._luminosities[telescope][storage_key][model]:
            av_bands = ", ".join([en.split("_")[-1] + "keV"
                                  for en in self._luminosities[telescope][storage_key][model].keys()])
            raise ParameterNotAssociatedError("{l}-{u}keV was not a luminosity energy band for the fit to those {t} "
                                              "spectra with {m}; available energy bands are "
                                              "{b}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value,
                                                           m=model, b=av_bands, t=telescope))

        # If no limits specified,the user gets all the luminosities, otherwise they get the one they asked for
        if en_key is None:
            parsed_lums = {}
            for lum_key in self._luminosities[telescope][storage_key][model]:
                lum_value = self._luminosities[telescope][storage_key][model][lum_key]
                parsed_lum = Quantity([lum.value for lum in lum_value], lum_value[0].unit)
                parsed_lums[lum_key] = parsed_lum
            return parsed_lums
        else:
            lum_value = self._luminosities[telescope][storage_key][model][en_key]
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
            raise UnitConversionError("You cannot convert to {} without redshift "
                                      "information.".format(out_unit.to_string()))

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

    def disassociate_obs(self, to_remove: Union[dict, str, list]):
        """
        Method that uses the supplied dictionary to safely remove data from the source. This data will no longer
        be used in any analyses, and would typically be removed because it is of poor quality, or doesn't contribute
        enough to justify its presence.

        These are examples of the different formats that 'to_remove' can take:

        {'telescope1': {'obsid1': ['inst1', 'inst2', 'inst3'], 'obsid2': ['inst2', 'inst3']},
         'telescope2': {'obsidN': ['inst1', 'inst2', 'inst3'], 'obsidM': ['inst2', 'inst3']}}

        OR

        {'telescope1': ['obsid1', 'obsid2']}

        OR

        {'telescope1': 'obsid1'}

        OR

        ['telescope1', 'telescope2']

        OR

        'telescope1'

        :param dict/str/list to_remove: Either a dictionary of observations to remove, (in the style of
            the source.instruments dictionary with the top level keys being telescopes, the mid-level keys being
            ObsIDs, and the lower levels being instrument names), or a similar dictionary with telescope names as
            keys and values of a list/single ObsID, or a single/list of telescope name(s) to remove all data
            related to that telescope.
        """

        # Users can pass just a telescope string, but we then need to convert it to the form
        #  that the rest of the function requires
        if isinstance(to_remove, str):
            # I check that this is a valid telescope - and here I just throw an error if not because it isn't like
            #  there are anymore to check that might be okay
            if to_remove in self.telescopes:
                to_remove = {to_remove: deepcopy(self.instruments[to_remove])}
            else:
                raise NotAssociatedError("{t} is not a telescope associated with {n}.".format(t=to_remove, n=self.name))
        # Here is where they have just passed a list of telescopes, and we need to fill in the blanks with
        #  the ObsIDs instruments currently loaded for those ObsIDs
        elif isinstance(to_remove, list):

            to_remove = {}
            for tel in to_remove:
                if tel in self.telescopes:
                    to_remove[tel] = deepcopy(self.instruments[tel])
                else:
                    warn('{t} is not a telescope associated with {n} and is being skipped.'.format(t=tel, n=self.name),
                         stacklevel=2)
        # In this instance the value is a dictionary, and I have created a disgusting nested set of if statements
        #  and for loops to ensure that the user hasn't passed anything daft
        elif isinstance(to_remove, dict):
            # I will be constructing the to remove dictionary as I go along
            final_to_remove = {}
            # Iterating through the top level keys (telescopes) and values (could be string ObsIDs, lists of string,
            #  ObsIDs, or dictionaries with ObsIDs as keys and single string instruments or lists of strings as values
            for tel, val in to_remove.items():
                # I've tried to be quite forgiving and just issue warnings rather than errors if something incorrect
                #  has been passed - in this case the current top level key isn't actually a valid telescope for this
                #  source
                if tel not in self.telescopes:
                    warn('{t} is not a telescope associated with {n} and is being skipped.'.format(t=tel, n=self.name),
                         stacklevel=2)
                    continue
                # If we get here we set up the to_remove dictionary that will be constructed
                final_to_remove[tel] = {}

                # If the current value is a string then we infer that it's just an ObsID, so we check it is valid
                #  and add the instruments that are associated with it to our final_to_remove dict
                if isinstance(val, str):
                    if val not in self.obs_ids[tel]:
                        warn("{o} is not an ObsID associated with {t} for {n}, and is being "
                             "skipped.".format(o=val, t=tel, n=self.name), stacklevel=2)
                        continue
                    final_to_remove[tel][val] = deepcopy(self.instruments[tel][val])

                # This should only be a list of ObsIDs at this level, so we assume it is and check them against the
                #  loaded ObsIDs, giving a warning if they are not valid
                elif isinstance(val, list):
                    for v_oi in val:
                        if v_oi not in self.obs_ids[tel]:
                            warn("{o} is not an ObsID associated with {t} for {n}, and is being "
                                 "skipped.".format(o=v_oi, t=tel, n=self.name), stacklevel=2)
                            continue
                        # At this point we add all the instruments associated with the current ObsID to the removal
                        #  dictionary
                        final_to_remove[tel][v_oi] = deepcopy(self.instruments[tel][v_oi])

                # If we note that the value is a dictionary, the only thing it should be is a dictionary with ObsIDs
                #  as keys and single instruments, or a list of instruments, as a value
                elif isinstance(val, dict):
                    # So we iterate through the dictionary
                    for v_oi, insts in val.items():
                        # Check that the key is an ObsID, skipping if not
                        if v_oi not in self.obs_ids[tel]:
                            warn("{o} is not an ObsID associated with {t} for {n}, and is being "
                                 "skipped.".format(o=v_oi, t=tel, n=self.name), stacklevel=2)

                        # Then if the value is a string it should be a single instrument, we check it is valid for
                        #  the current ObsID
                        if isinstance(insts, str):
                            if insts.lower() not in self.instruments[tel][v_oi]:
                                warn("{i} is not an instrument associated with {t}-{o} for {n}, and is being "
                                     "skipped.".format(i=insts.lower(), t=tel, o=v_oi, n=self.name))
                                continue
                            # If it is then it is added to the removal dictionary
                            final_to_remove[tel][v_oi] = [insts.lower()]

                        # This should only be a list of instruments, so we check them and add them to the removal
                        #  dictionary if they are valid
                        elif isinstance(insts, list):
                            final_inst_list = []
                            for inst in insts:
                                if inst.lower() not in self.instruments[tel][v_oi] and inst != 'combined':
                                    warn("{i} is not an instrument associated with {t}-{o} for {n}, and is being "
                                         "skipped.".format(i=inst.lower(), t=tel, o=v_oi, n=self.name))
                                    continue
                                final_inst_list.append(inst.lower())
                            # Check to make sure our list of instruments actually has something in it
                            if len(final_inst_list) != 0:
                                final_to_remove[tel][v_oi] = final_inst_list

            # We finally actually set the to_remove variable
            to_remove = deepcopy(final_to_remove)

        # Sets the attribute that tells us whether any data has been removed
        if not self._disassociated:
            self._disassociated = True

        # We want to store knowledge of what data has been removed, if there hasn't been anything taken away yet
        #  then we can just set it equal to the to_remove dictionary
        if len(self._disassociated_obs) == 0:
            self._disassociated_obs = to_remove
        # Otherwise we have to add the data to the existing dictionary structure
        else:
            for tel in to_remove:
                # In this case we can just set the entry for that telescope to the current contents of to_remove for
                #  the current telescope
                if tel not in self._disassociated_obs:
                    self._disassociated_obs[tel] = to_remove[tel]
                # Otherwise we have to be a little more careful as you can't just add dictionaries together
                else:
                    # Cycling through the ObsIDs for the current telescope
                    for o in to_remove[tel]:
                        # If this ObsID isn't in the disassociated dictionary yet, we can just set that entry to
                        #  the to_remove entry
                        if o not in self._disassociated_obs[tel]:
                            self._disassociated_obs[tel] = to_remove[tel][o]
                        # However if there is an entry for this ObsID we need to add the lists together
                        else:
                            self._disassociated_obs[tel][o] += to_remove[tel][o]

        for tel in to_remove:
            # If we're disassociating certain observations, odds on the combined products are no longer valid
            if "combined" in self._products[tel]:
                del self._products[tel]["combined"]
                self._products[tel]['combined'] = {}
                if "combined" in self._interloper_masks[tel]:
                    del self._interloper_masks[tel]["combined"]
                    self._interloper_masks[tel]["combined"] = {}
                self._fit_results[tel] = {}
                self._test_stat[tel] = {}
                self._dof[tel] = {}
                self._total_count_rate[tel] = {}
                self._total_exp[tel] = {}
                self._luminosities[tel] = {}

            # This will be set to True if a whole ObsID of this telescope is removed, not just some instruments
            #  associated with an ObsID - later on in this method that will trigger the reset of interloper regions
            #  (which are based on all ObsID's regions for a particular telescope), and the reset of the interloper
            #  masks based on those regions
            whole_obsid_dis = False
            for o in to_remove[tel]:
                for i in to_remove[tel][o]:
                    # Because of irritating missions like eROSITA where the event lists are distributed pre-combined
                    #  I have this flag which identifies those missions, we can't necessarily delete
                    #  instrument-specific products, though they MIGHT be there
                    if not COMBINED_INSTS[tel] and i in self._products[tel][o]:
                        del self._products[tel][o][i]
                        del self._obs[tel][o][self._obs[tel][o].index(i)]
                    # This is a little out of place, as the top level is iterating through instruments and so am I,
                    #  but in theory the only 'i' for a 'combined inst' mission should be 'combined' - I am just
                    #  removing the knowledge of all individual instruments here.
                    elif COMBINED_INSTS[tel]:
                        self._obs[tel][o] = []

                if len(self._obs[tel][o]) == 0:
                    # We now set the variable that describes whether a whole ObsID has been removed to True
                    whole_obsid_dis = True

                    del self._products[tel][o]
                    del self._detected[tel][o]
                    del self._initial_regions[tel][o]
                    del self._initial_region_matches[tel][o]
                    del self._regions[tel][o]
                    del self._other_regions[tel][o]
                    del self._alt_match_regions[tel][o]
                    # These are made on demand, so need to check if its actually present first
                    if o in self._interloper_masks[tel]:
                        del self._interloper_masks[tel][o]
                    if self._peaks is not None and tel in self._peaks:
                        del self._peaks[tel][o]

                    if o in self._obs[tel]:
                        del self._obs[tel][o]

                    if o in self._obs_sep[tel]:
                        del self._obs_sep[tel][o]

                if len(self._obs[tel]) == 0:
                    del self._products[tel]
                    del self._detected[tel]
                    del self._initial_regions[tel]
                    del self._initial_region_matches[tel]
                    del self._regions[tel]
                    del self._other_regions[tel]
                    del self._alt_match_regions[tel]
                    # These are made on demand, so need to check if its actually present first
                    if tel in self._interloper_masks:
                        del self._interloper_masks[tel]
                    if self._peaks is not None and tel in self._peaks:
                        del self._peaks[tel]

                    del self._obs[tel]
                    if tel in self._obs_sep:
                        del self._obs_sep[tel]

                    warn("All {t} observations have been disassociated from {n}.".format(t=tel, n=self.name),
                         stacklevel=2)

            if whole_obsid_dis and tel in self._other_regions:
                # We replace the interloper regions entry for this telescope (i.e. the combined list of contaminant
                #  regions) here, as it is possible we may have disassociated an ObsID and left its regions behind here.
                # If an ObsID has been entirely removed, it will no longer be in '_other_regions' so this should work
                self._interloper_regions[tel] = [r for o in self._other_regions[tel]
                                                 for r in self._other_regions[tel][o]]
                # We also have to wipe the interloper masks that already exist, as they would have been created with
                #  other ObsID's regions
                self._interloper_masks[tel] = {o: {} for o in self.obs_ids[tel] + ['combined']}
            elif whole_obsid_dis and tel not in self._other_regions:
                if tel in self._interloper_regions:
                    del self._interloper_regions[tel]
                if tel in self._interloper_masks:
                    del self._interloper_masks[tel]

        if len(self._obs) == 0:
            raise NoValidObservationsError("No observations remain associated with {} after cleaning".format(self.name))

        # We attempt to load in matching XGA products if that was the behaviour set by load_products on init
        if self._load_products:
            self._existing_xga_products(self._load_fits, True)

    def obs_check(self, reg_type: str, threshold_fraction: float = 0.5) -> Dict:
        """
        This method uses exposure maps and region masks to determine which telescope/ObsID/instrument combinations
        are not contributing to the analysis. It calculates the area intersection of the mask and exposure
        maps, and if (for a given Telescope-ObsID-Instrument) the ratio of that area to the full area of the region
        calculated is less than the threshold fraction, that telescope-ObsID-instrument will be included in the
        returned rejection dictionary.

        :param str reg_type: The region type for which to calculate the area intersection.
        :param float threshold_fraction: Intersection area/ full region area ratios below this value will mean an
            ObsID-Instrument is rejected.
        :return: A dictionary of telescope keys on the top level, then ObsID keys a level down, with a list of
            instruments as the values, that should be rejected according to the criteria supplied to this method.
        :rtype: Dict
        """
        area = {t: {o: {} for o in self.obs_ids[t]} for t in self.obs_ids}
        full_area = {t: {} for t in self.obs_ids}

        reject_dict = {}

        # TODO This will need a redo once the functions to generate exposure maps are implemented for other telescopes
        # TODO Also just double check I've implemented this right
        for tel in self.telescopes:
            if tel not in ['xmm', 'erosita']:
                warn("The features required for observation checking are not implemented for telescopes "
                     "other than XMM and eROSITA - though it is a priority.", stacklevel=2)
                continue
            else:
                if tel == 'xmm':
                    # Again don't particularly want to do this local import, but its just easier
                    from xga.generate.sas import eexpmap

                    # Going to ensure that individual exposure maps exist for each of the ObsID/instrument combinations
                    #  first, then checking where the source lies on the exposure map
                    eexpmap(self, self._peak_lo_en, self._peak_hi_en)
                elif tel == 'erosita':
                    from xga.generate.esass import expmap
                    expmap(self, self._peak_lo_en, self._peak_hi_en)

                for o in self.obs_ids[tel]:
                    # Exposure maps of the peak finding energy range for this ObsID
                    exp_maps = self.get_expmaps(o, lo_en=self._peak_lo_en, hi_en=self._peak_hi_en, telescope=tel)

                    # Just making sure that the exp_maps variable can be iterated over
                    if not isinstance(exp_maps, list):
                        exp_maps = [exp_maps]

                    m = self.get_source_mask(reg_type, tel, o, central_coord=self._default_coord)[0]
                    full_area[tel][o] = m.sum()

                    for ex in exp_maps:
                        # Grabs exposure map data, then alters it so anything that isn't zero is a one
                        ex_data = ex.data.copy()
                        ex_data[ex_data > 0] = 1
                        # We do this because it then becomes very easy to calculate the intersection area of the mask
                        #  with the XMM chips. Just mask the modified expmap, then sum.
                        area[tel][o][ex.instrument] = (ex_data * m).sum()

                        # Desperately trying to make sure I don't get memory allocation errors, particularly with
                        #  eROSITA with their memory-hogging images/exposure maps
                        del ex_data
                        ex.unload(unload_data=True, unload_header=False)

                if max(list(full_area[tel].values())) == 0:
                    # Everything has to be rejected in this case
                    reject_dict = deepcopy(self.instruments[tel])
                else:
                    for o in area[tel]:
                        for i in area[tel][o]:
                            if full_area[tel][o] != 0:
                                frac = (area[tel][o][i] / full_area[tel][o])
                            else:
                                frac = 0
                            if frac <= threshold_fraction and tel not in reject_dict:
                                reject_dict[tel] = {o: [i]}
                            elif frac <= threshold_fraction and o not in reject_dict[tel]:
                                reject_dict[tel][o] = [i]
                            elif frac <= threshold_fraction and o in reject_dict[tel]:
                                reject_dict[tel][o].append(i)

        return reject_dict

    def snr_ranking(self, outer_radius: Union[Quantity, str], lo_en: Quantity = None, hi_en: Quantity = None,
                    allow_negative: bool = False, telescope: List[str] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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
            zero, which results in a lower signal-to-noise (and can result in a negative signal-to-noise).
        :param List[str] telescope: The telescopes to return snr rankings for. By default these will be all telescopes
            associated to the source.
        :return: Two dictionaries with top level telescope keys, the first dictionary contains N by 2 array, with the ObsID, Instrument combinations in order
            of ascending signal-to-noise, then a dictionary containing an array containing the order SNR ratios.
        :rtype: Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
        """
        if telescope is None:
            telescope = self.telescopes
        
        obs_inst_dict = {}
        snrs_dict = {}

        for tel in telescope:
            # Set up some lists for the ObsID-Instrument combos and their SNRs respectively
            obs_inst = []
            snrs = []
            # We loop through the ObsIDs associated with this source and the instruments associated with those ObsIDs
            for obs_id in self.instruments[tel]:
                for inst in self.instruments[tel][obs_id]:
                    # Use our handy get_snr method to calculate the SNRs we want, then add that and the
                    #  ObsID-inst combo into their respective lists
                    snrs.append(
                        self.get_snr(outer_radius, tel, self.default_coord, lo_en, hi_en, obs_id, inst, allow_negative))
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

            obs_inst_dict[tel] = obs_inst
            snrs_dict[tel] = snrs

        # And return our ordered dictionaries
        return obs_inst_dict, snrs_dict

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
                cnts.append(self.get_counts(outer_radius, 'xmm', self.default_coord, lo_en, hi_en, obs_id, inst))
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
        This method calculates the separation between the user supplied ra_dec coordinates, and the peak
        coordinates in the requested off_unit. If there is no peak attribute and error will be thrown, and if no
        peak has been calculated then the result will be 0.

        :param Unit/str off_unit: The unit that the offset should be in.
        :return: The offset between ra_dec and peak, in the requested unit.
        :rtype: Quantity
        """
        # Check that the source has a peak attribute to fetch, otherwise throw error
        if not hasattr(self, 'peak'):
            raise AttributeError("This source does not have a peak attribute, and so an offset cannot be calculated.")

        # Calculate the euclidean distance between ra_dec and peak
        sep = np.sqrt(abs(self.ra_dec[0] - self.peak[0]) ** 2 + abs(self.ra_dec[1] - self.peak[1]) ** 2)
        # Convert the separation to the requested unit - this will throw an error if the unit is stupid
        conv_sep = self.convert_radius(sep, off_unit)

        # Return the converted separation
        return conv_sep

    def info(self):
        """
        Very simple function that just prints a summary of important information related to the source object.
        """
        print("\n-----------------------------------------------------")
        print("Source Name - {}".format(self._name))
        print("User Coordinates - ({0}, {1}) degrees".format(*self._ra_dec))
        if self._use_peak is not None and self._use_peak:
            print("X-ray Peak - ({0}, {1}) degrees".format(*self.peak.value))
        print("nH - {}".format(self.nH))
        if self._redshift is not None:
            print("Redshift - {}".format(round(self._redshift, 3)))

        if self._regions is not None and "custom" in self._radii:
            if self._redshift is not None:
                region_radius = ang_to_rad(self._custom_region_radius, self._redshift, cosmo=self._cosmo)
            else:
                region_radius = self._custom_region_radius.to("deg")
            print("Custom Region Radius - {}".format(region_radius.round(2)))
        elif self._regions is not None and "point" in self._radii:
            if self._redshift is not None:
                region_radius = ang_to_rad(self._radii['point'], self._redshift, cosmo=self._cosmo)
            else:
                region_radius = self._radii['point'].to("deg")
            print("Point Region Radius - {}".format(region_radius.round(2)))

        if self._r200 is not None:
            print("R200 - {}".format(self._r200.round(2)))
        if self._r500 is not None:
            print("R500 - {}".format(self._r500.round(2)))
        if self._r2500 is not None:
            print("R2500 - {}".format(self._r2500.round(2)))

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

        for tel in self.telescopes:
            pr_tel = PRETTY_TELESCOPE_NAMES[tel]
            print('')
            print('-- ' + pr_tel + ' --')
            print("ObsIDs - {n}".format(n=len(self.obs_ids[tel])))
            for inst in ALLOWED_INST[tel]:
                print("{i} Observations - {n}".format(i=inst.upper(), n=self.num_inst_obs[tel][inst]))
            # TODO resolve how to supply the separation information now that it isn't as simple as on vs off axis
            print("With initial detection - {}".format(sum([1 for o in self._initial_region_matches[tel]
                                                            if self._initial_region_matches[tel][o].sum() == 1])))
            # If a combined exposure map exists, we'll use it to give the user an idea of the total exposure. If there
            #  is only a single observation-instrument combo associated then we will use that, or if the telescope
            #  is a 'combined instruments' telescope (i.e. the data are shipped combined) and there is only one ObsID
            #  then we will use that exposure map (if it exists)
            if len([o+inst for o in self.instruments[tel] for inst in self.instruments[tel][o]]) == 1 or \
                    (COMBINED_INSTS[tel] and len(self._obs[tel]) == 1):
                # In this case there is only one Obs-ID instrument combo for this telescope, and it isn't one of those
                #  tricksy telescopes that ship data from multiple instrument combined (looking at you eROSITA
                #  calibration data
                try:
                    # It is possible this will return a list, but they should just be different energies, so we'll use
                    #  the first one if there are multiple
                    ex = self.get_expmaps(telescope=tel)
                    if isinstance(ex, list):
                        ex = ex[0]
                    print("Total exposure - {}".format(ex.get_exp(self.ra_dec).to('ks').round(2)))
                except NoProductAvailableError:
                    pass

                try:
                    if not COMBINED_INSTS[tel]:
                        im = self.get_images(self.obs_ids[tel][0], self.instruments[tel][self.obs_ids[tel][0]][0],
                                             telescope=tel, lo_en=self.peak_lo_en, hi_en=self.peak_hi_en)
                    else:
                        im = self.get_images(self.obs_ids[tel][0], telescope=tel, lo_en=self.peak_lo_en,
                                             hi_en=self.peak_hi_en)
                    if 'point' in self._radii:
                        print("Point {l}-{u}keV SNR - {s}".format(s=self.get_snr("point", tel, self._default_coord,
                                                                                 obs_id=im.obs_id,
                                                                                 inst=im.instrument).round(2),
                                                                  l=self.peak_lo_en.value.round(2),
                                                                  u=self.peak_hi_en.value.round()))
                    if 'custom' in self._radii:
                        print("Custom {l}-{u}keV SNR - {s}".format(s=self.get_snr("custom", tel, self._default_coord,
                                                                                  obs_id=im.obs_id,
                                                                                  inst=im.instrument).round(2),
                                                                   l=self.peak_lo_en.value.round(2),
                                                                   u=self.peak_hi_en.value.round()))
                    if 'r2500' in self._radii:
                        print("R2500 {l}-{u}keV SNR - {s}".format(s=self.get_snr("r2500", tel, self._default_coord,
                                                                                 obs_id=im.obs_id,
                                                                                 inst=im.instrument).round(2),
                                                                  l=self.peak_lo_en.value.round(2),
                                                                  u=self.peak_hi_en.value.round()))
                    if 'r500' in self._radii:
                        print("R500 {l}-{u}keV SNR - {s}".format(s=self.get_snr("r500", tel, self._default_coord,
                                                                                obs_id=im.obs_id,
                                                                                inst=im.instrument).round(2),
                                                                 l=self.peak_lo_en.value.round(2),
                                                                 u=self.peak_hi_en.value.round()))

                    if 'r200' in self._radii:
                        print("R500 {l}-{u}keV SNR - {s}".format(s=self.get_snr("r200", tel, self._default_coord,
                                                                                obs_id=im.obs_id,
                                                                                inst=im.instrument).round(2),
                                                                 l=self.peak_lo_en.value.round(2),
                                                                 u=self.peak_hi_en.value.round()))
                except NoProductAvailableError:
                    pass

            else:
                try:
                    ex = self.get_combined_expmaps(telescope=tel)
                    if isinstance(ex, list):
                        ex = ex[0]
                    print("Total exposure - {}".format(ex.get_exp(self.ra_dec).to('ks').round(2)))
                except NoProductAvailableError:
                    pass

                try:
                    if 'point' in self._radii:
                        print("Point {l}-{u}keV SNR - {s}".format(s=self.get_snr("point", tel,
                                                                                 self._default_coord).round(2),
                                                                  l=self.peak_lo_en.value.round(2),
                                                                  u=self.peak_hi_en.value.round(2)))
                    if 'custom' in self._radii:
                        print("Custom {l}-{u}keV SNR - {s}".format(s=self.get_snr("custom", tel,
                                                                                  self._default_coord).round(2),
                                                                   l=self.peak_lo_en.value.round(2),
                                                                   u=self.peak_hi_en.value.round(2)))

                    if 'r2500' in self._radii:
                        print("R2500 {l}-{u}keV SNR - {s}".format(s=self.get_snr("r2500", tel,
                                                                                 self._default_coord).round(2),
                                                                  l=self.peak_lo_en.value.round(2),
                                                                  u=self.peak_hi_en.value.round(2)))

                    if 'r500' in self._radii:
                        print("R500 {l}-{u}keV SNR - {s}".format(s=self.get_snr("r500", tel,
                                                                                self._default_coord).round(2),
                                                                 l=self.peak_lo_en.value.round(2),
                                                                 u=self.peak_hi_en.value.round(2)))

                    if 'r200' in self._radii:
                        print("R200 {l}-{u}keV SNR - {s}".format(s=self.get_snr("r200", tel,
                                                                                self._default_coord).round(2),
                                                                 l=self.peak_lo_en.value.round(2),
                                                                 u=self.peak_hi_en.value.round(2)))

                except NoProductAvailableError:
                    pass

            print("Spectra associated - {}".format(len(self.get_products("spectrum", telescope=tel))))

            if len(self._fit_results[tel]) != 0:
                print("Fitted Models - {}".format(" | ".join(self.fitted_models)))

            if 'get_temperature' in dir(self):
                try:
                    tx = self.get_temperature('r500', tel, 'constant*tbabs*apec').value.round(2)
                    # Just average the uncertainty for this
                    print("R500 Tx - {0}±{1}[keV]".format(tx[0], tx[1:].mean().round(2)))
                except (ModelNotAssociatedError, NoProductAvailableError):
                    pass

                try:
                    lx = self.get_luminosities('r500', tel, 'constant*tbabs*apec', lo_en=Quantity(0.5, 'keV'),
                                               hi_en=Quantity(2.0, 'keV')).to('10^44 erg/s').value.round(2)
                    print("R500 0.5-2.0keV Lx - {0}±{1}[e+44 erg/s]".format(lx[0], lx[1:].mean().round(2)))

                except (ModelNotAssociatedError, NoProductAvailableError):
                    pass
        print("-----------------------------------------------------\n")

    def __len__(self) -> int:
        """
        Method to return the length of the products dictionary (which means the number of individual ObsIDs associated
        with this source, across all telescopes), when len() is called on an instance of this class.

        :return: The number of observations, across all telescopes, associated with this source.
        :rtype: int
        """
        return sum([len(self.obs_ids[t]) for t in self.telescopes])


class NullSource(BaseSource):
    """
    A useful, but very limited, source class, which is designed to enable the bulk generation of
    non-source-specific products like images and exposure maps. This source doesn't represent a specific
    astrophysical source, but rather whole sets of ObsIDs (which are not necessarily related). It is
    possible to specify the particular observations to be included, but the default behaviour is to select
    all observations available.

    If large sets of ObsIDs are being included, this can take some time to declare.

    :param List[str]/dict/str obs: The particular observations that are to be included in this NullSource
        declaration. The default is None, in which case all ObsIDs are considered. If a single telescope is
        being considered then a list of ObsIDs may be passed, but if a set of telescopes are being considered
        then a dictionary of lists of ObsIDs, with telescope-name dictionary keys, should be passed.
    :param str/List[str] telescope: The particular telescope(s) that are to be included in this NullSource
        declaration. The default is None, in which case all telescopes are considered. Otherwise, this should
        either be a single telescope name, or a list of telescope names.
    :param bool null_load_products: Controls whether the image and exposure maps that may be specified in the
        configuration file are loaded. This can cause slow-down with very large NullSources, so by default is
        set to False.
    :param bool load_products: This controls whether existing XGA GENERATED products should be loaded on
        declaration of the source. Default is True.
    """
    def __init__(self, obs: Union[List[str], dict, str] = None, telescope: Union[str, List[str]] = None,
                 null_load_products: bool = False, load_products: bool = True):
        """
        A useful, but very limited, source class, which is designed to enable the bulk generation of
        non-source-specific products like images and exposure maps. This source doesn't represent a specific
        astrophysical source, but rather whole sets of ObsIDs (which are not necessarily related). It is
        possible to specify the particular observations to be included, but the default behaviour is to select
        all observations available.

        If large sets of ObsIDs are being included, this can take some time to declare.

        :param List[str]/dict/str obs: The particular observations that are to be included in this NullSource
            declaration. The default is None, in which case all ObsIDs are considered. If a single telescope is
            being considered then a list of ObsIDs may be passed, but if a set of telescopes are being considered
            then a dictionary of lists of ObsIDs, with telescope-name dictionary keys, should be passed.
        :param str/List[str] telescope: The particular telescope(s) that are to be included in this NullSource
            declaration. The default is None, in which case all telescopes are considered. Otherwise, this should
            either be a single telescope name, or a list of telescope names.
        :param bool null_load_products: Controls whether the image and exposure maps that may be specified in the
            configuration file are loaded. This can cause slow-down with very large NullSources, so by default is
            set to False.
        :param bool load_products: This controls whether existing XGA GENERATED products should be loaded on
            declaration of the source. Default is True.
        """
        # Just makes sure that if a single ObsID has been passed, then it is stored in a list - as all other
        #  stages will expect it to be None, a list, or a dictionary
        if isinstance(obs, str):
            obs = [obs]

        super().__init__(0, 0, None, "AllObservations", load_products=load_products, telescope=telescope,
                         sel_null_obs=obs, null_load_products=null_load_products)

        self._ra_dec = np.array([np.NaN, np.NaN])
        self._nH = Quantity(np.NaN, self._nH.unit)
