#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/04/2023, 15:12. Copyright (c) The Contributors

import json
import os
import shutil
from configparser import ConfigParser
from subprocess import Popen, PIPE
from typing import List, Tuple
from warnings import warn

import pandas as pd
import pkg_resources
from astropy.constants import m_p, m_e
from astropy.cosmology import LambdaCDM
from astropy.units import Quantity, def_unit, add_enabled_units
from astropy.wcs import WCS
from fitsio import read_header
from fitsio.header import FITSHDR
from numpy import nan, floor
from tqdm import tqdm

from .exceptions import XGAConfigError

# If XDG_CONFIG_HOME is set, then use that, otherwise use this default config path
CONFIG_PATH = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'xga'))
# The path to the census file, which documents all available ObsIDs and their pointings
CENSUS_FILE = os.path.join(CONFIG_PATH, 'census.csv')
# The path to the blacklist file, which is where users can specify ObsIDs they don't want to be used in analyses
BLACKLIST_FILE = os.path.join(CONFIG_PATH, 'blacklist.csv')
# XGA config file path
CONFIG_FILE = os.path.join(CONFIG_PATH, 'xga.cfg')
# Section of the config file for setting up the XGA module
XGA_CONFIG = {"xga_save_path": "/this/is/required/xga_output/"}
# Will have to make it clear in the documentation what is allowed here, and which can be left out
XMM_FILES = {"root_xmm_dir": "/this/is/required/xmm_obs/data/",
             "clean_pn_evts": "/this/is/required/{obs_id}/pn_exp1_clean_evts.fits",
             "clean_mos1_evts": "/this/is/required/{obs_id}/mos1_exp1_clean_evts.fits",
             "clean_mos2_evts": "/this/is/required/{obs_id}/mos2_exp1_clean_evts.fits",
             "attitude_file": "/this/is/required/{obs_id}/attitude.fits",
             "lo_en": ['0.50', '2.00'],
             "hi_en": ['2.00', '10.00'],
             "pn_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
             "mos1_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_img.fits",
             "mos2_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_img.fits",
             "pn_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
             "mos1_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_expmap.fits",
             "mos2_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_expmap.fits",
             "region_file": "/this/is/optional/xmm_obs/regions/{obs_id}/regions.reg"}
# List of XMM products supported by XGA that are allowed to be energy bound
ENERGY_BOUND_PRODUCTS = ["image", "expmap", "ratemap", "combined_image", "combined_expmap", "combined_ratemap"]
# These are the built in profile types
PROFILE_PRODUCTS = ["brightness_profile", "gas_density_profile", "gas_mass_profile", "1d_apec_norm_profile",
                    "1d_proj_temperature_profile", "gas_temperature_profile", "baryon_fraction_profile",
                    "1d_proj_metallicity_profile", "1d_emission_measure_profile", "hydrostatic_mass_profile"]
COMBINED_PROFILE_PRODUCTS = ["combined_"+pt for pt in PROFILE_PRODUCTS]
# List of all XMM products supported by XGA
ALLOWED_PRODUCTS = ["spectrum", "grp_spec", "regions", "events", "psf", "psfgrid", "ratemap", "combined_spectrum",
                    ] + ENERGY_BOUND_PRODUCTS + PROFILE_PRODUCTS + COMBINED_PROFILE_PRODUCTS
XMM_INST = ["pn", "mos1", "mos2"]
# This list contains banned filter types - these occur in observations that I don't want XGA to try and use
BANNED_FILTS = ['CalClosed', 'Closed']

# Here we read in files that list the errors and warnings in SAS
errors = pd.read_csv(pkg_resources.resource_filename(__name__, "files/sas_errors.csv"), header="infer")
warnings = pd.read_csv(pkg_resources.resource_filename(__name__, "files/sas_warnings.csv"), header="infer")
# Just the names of the errors in two handy constants
SASERROR_LIST = errors["ErrName"].values
SASWARNING_LIST = warnings["WarnName"].values

# XSPEC file extraction (and base fit) scripts
XGA_EXTRACT = pkg_resources.resource_filename(__name__, "xspec_scripts/xga_extract.tcl")
BASE_XSPEC_SCRIPT = pkg_resources.resource_filename(__name__, "xspec_scripts/general_xspec_fit.xcm")
COUNTRATE_CONV_SCRIPT = pkg_resources.resource_filename(__name__, "xspec_scripts/cr_conv_calc.xcm")
# Useful jsons of all XSPEC models, their required parameters, and those parameter's units
with open(pkg_resources.resource_filename(__name__, "files/xspec_model_pars.json5"), 'r') as filey:
    MODEL_PARS = json.load(filey)
with open(pkg_resources.resource_filename(__name__, "files/xspec_model_units.json5"), 'r') as filey:
    MODEL_UNITS = json.load(filey)
ABUND_TABLES = ["feld", "angr", "aneb", "grsa", "wilm", "lodd", "aspl"]
# TODO Populate this further, also actually calculate and verify these myself, the value here is taken
#  from pyproffit code
# For a fully ionised plasma, this is the electron-to-proton ratio
NHC = {"angr": 1.199}
XSPEC_FIT_METHOD = ["leven", "migrad", "simplex"]

# I know this is practically pointless, I could just use m_p, but I like doing things properly.
HY_MASS = m_p + m_e

# Mean molecular weight, mu
# TODO Make sure this doesn't change with abundance table, I think it might?
MEAN_MOL_WEIGHT = 0.61

# A centralised constant to define what radius labels are allowed
RAD_LABELS = ["region", "r2500", "r500", "r200", "custom", "point"]

# Adding a default concordance cosmology set up here - this replaces the original default choice of Planck15
DEFAULT_COSMO = LambdaCDM(70, 0.3, 0.7)


def xmm_obs_id_test(test_string: str) -> bool:
    """
    Crude function to try and determine if a string follows the pattern of an XMM ObsID.

    :param str test_string: The string we wish to test.
    :return: Whether the string is probably an XMM ObsID or not.
    :rtype: bool
    """
    probably_xmm = False
    # XMM ObsIDs are ten characters long, and making sure there is no . that might indicate a file extension.
    if len(test_string) == 10 and '.' not in test_string:
        try:
            # To our constant pain, XMM ObsIDs can convert to integers, so if this works then its likely
            # an XMM ObsID.
            int(test_string)
            probably_xmm = True
        except ValueError:
            pass
    return probably_xmm


def observation_census(config: ConfigParser) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A function to initialise or update the file that stores which observations are available in the user
    specified XMM data directory, and what their pointing coordinates are.
    CURRENTLY THIS WILL NOT UPDATE TO DEAL WITH OBSID FOLDERS THAT HAVE BEEN DELETED.

    :param config: The XGA configuration object.
    :return: ObsIDs and pointing coordinates of available XMM observations.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # The census lives in the XGA config folder, and CENSUS_FILE stores the path to it.
    # If it exists, it is read in, otherwise empty lists are initialised to be appended to.
    if os.path.exists(CENSUS_FILE):
        with open(CENSUS_FILE, 'r') as census:
            obs_lookup = census.readlines()  # Reads the lines of the files
            # This is just ObsIDs, needed to see which ObsIDs have already been processed.
            obs_lookup_obs = [entry.split(',')[0] for entry in obs_lookup[1:]]
    else:
        obs_lookup = ["ObsID,RA_PNT,DEC_PNT,USE_PN,USE_MOS1,USE_MOS2\n"]
        obs_lookup_obs = []

    # Creates black list file if one doesn't exist, then reads it in
    if not os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, 'w') as bl:
            bl.write("ObsID,EXCLUDE_PN,EXCLUDE_MOS1,EXCLUDE_MOS2")
    blacklist = pd.read_csv(BLACKLIST_FILE, header="infer", dtype=str)

    # This part here is to support blacklists used by older versions of XGA, where only a full ObsID was excluded.
    #  Now we support individual instruments of ObsIDs being excluded from use, so there are extra columns expected
    if len(blacklist.columns) == 1:
        # Adds the three new columns, all with a default value of True. So any ObsID already in the blacklist
        #  will have the same behaviour as before, all instruments for the ObsID are excluded
        blacklist[["EXCLUDE_PN", "EXCLUDE_MOS1", "EXCLUDE_MOS2"]] = 'T'
        # If we have even gotten to this stage then the actual blacklist file needs re-writing, so I do
        blacklist.to_csv(BLACKLIST_FILE, index=False)

    # Need to find out which observations are available, crude way of making sure they are ObsID directories
    # This also checks that I haven't run them before
    obs_census = [entry for entry in os.listdir(config["XMM_FILES"]["root_xmm_dir"]) if xmm_obs_id_test(entry)
                  and entry not in obs_lookup_obs]
    if len(obs_census) != 0:
        with tqdm(desc="Assembling list of ObsIDs", total=len(obs_census)) as census_progress:
            for obs in obs_census:
                info = {'ra': None, 'dec': None, "the_rest": []}
                for key in ["clean_pn_evts", "clean_mos1_evts", "clean_mos2_evts"]:
                    evt_path = config["XMM_FILES"][key].format(obs_id=obs)
                    if os.path.exists(evt_path):
                        evts_header = read_header(evt_path)
                        try:
                            # Reads out the filter header, if it is CalClosed/Closed then we can't use it
                            filt = evts_header["FILTER"]
                            submode = evts_header["SUBMODE"]
                            info['ra'] = evts_header["RA_PNT"]
                            info['dec'] = evts_header["DEC_PNT"]
                        except KeyError:
                            # It won't actually, but this will trigger the if statement that tells XGA not to use
                            #  this particular obs/inst combo
                            filt = "CalClosed"

                        # TODO Decide if I want to disallow small window mode observations
                        if filt not in BANNED_FILTS:
                            info["the_rest"].append("T")
                        else:
                            info["the_rest"].append("F")
                    else:
                        info["the_rest"].append("F")

                use_insts = ",".join(info["the_rest"])
                # Write the information to the line that will go in the census csv
                if info["ra"] is not None and info["dec"] is not None:
                    # Format to write to the census.csv that lives in the config directory.
                    obs_lookup.append("{o},{r},{d},{a}\n".format(o=obs, r=info["ra"], d=info["dec"], a=use_insts))
                else:
                    obs_lookup.append("{o},,,{a}\n".format(o=obs, r=info["ra"], d=info["dec"], a=use_insts))

                census_progress.update(1)
        with open(CENSUS_FILE, 'w') as census:
            census.writelines(obs_lookup)

    # I do the stripping and splitting to make it a 3 column array, needed to be lines to write to file
    obs_lookup = pd.DataFrame(data=[entry.strip('\n').split(',') for entry in obs_lookup[1:]],
                              columns=obs_lookup[0].strip("\n").split(','), dtype=str)
    obs_lookup["RA_PNT"] = obs_lookup["RA_PNT"].replace('', nan).astype(float)
    obs_lookup["DEC_PNT"] = obs_lookup["DEC_PNT"].replace('', nan).astype(float)
    obs_lookup["USE_PN"] = obs_lookup["USE_PN"].replace('T', True).replace('F', False)
    obs_lookup["USE_MOS1"] = obs_lookup["USE_MOS1"].replace('T', True).replace('F', False)
    obs_lookup["USE_MOS2"] = obs_lookup["USE_MOS2"].replace('T', True).replace('F', False)
    return obs_lookup, blacklist


def to_list(str_rep_list: str) -> list:
    """
    Convenience function to change a string representation of a Python list into an actual list object.

    :param str str_rep_list: String that represents a Python list. e.g. "['0.5', '2.0']"
    :return: The parsed representative string.
    :rtype: list
    """
    in_parts = str_rep_list.strip("[").strip("]").split(',')
    real_list = [part.strip(' ').strip("'").strip('"') for part in in_parts if part != '' and part != ' ']
    return real_list


def energy_to_channel(energy: Quantity) -> int:
    """
    Converts an astropy energy quantity into an XMM channel.

    :param energy:
    """
    energy = energy.to("eV").value
    chan = int(energy)
    return chan


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


def find_all_wcs(hdr: FITSHDR) -> List[WCS]:
    """
    A play on the function of the same name in astropy.io.fits, except this one will take a fitsio header object
    as an argument, and construct astropy wcs objects. Very simply looks for different WCS entries in the
    header, and uses their critical values to construct astropy WCS objects.

    :return: A list of astropy WCS objects extracted from the input header.
    :rtype: List[WCS]
    """
    wcs_search = [k.split("CTYPE")[-1][-1] for k in hdr.keys() if "CTYPE" in k]
    wcs_nums = [w for w in wcs_search if w.isdigit()]
    wcs_not_nums = [w for w in wcs_search if not w.isdigit()]
    if len(wcs_nums) != 2 and len(wcs_nums) != 0:
        raise KeyError("There are an odd number of CTYPEs with no extra key ")
    elif len(wcs_nums) == 2:
        wcs_keys = [""] + list(set(wcs_not_nums))
    elif len(wcs_nums) == 0:
        wcs_keys = list(set(wcs_not_nums))

    wcses = []
    for key in wcs_keys:
        w = WCS(naxis=2)
        w.wcs.crpix = [hdr["CRPIX1{}".format(key)], hdr["CRPIX2{}".format(key)]]
        w.wcs.cdelt = [hdr["CDELT1{}".format(key)], hdr["CDELT2{}".format(key)]]
        w.wcs.crval = [hdr["CRVAL1{}".format(key)], hdr["CRVAL2{}".format(key)]]
        w.wcs.ctype = [hdr["CTYPE1{}".format(key)], hdr["CTYPE2{}".format(key)]]
        wcses.append(w)

    return wcses


if not os.path.exists(CONFIG_PATH):
    os.makedirs(CONFIG_PATH)

# If first XGA run, creates default config file
if not os.path.exists(CONFIG_FILE):
    xga_default = ConfigParser()
    xga_default.add_section("XGA_SETUP")
    xga_default["XGA_SETUP"] = XGA_CONFIG
    xga_default.add_section("XMM_FILES")
    xga_default["XMM_FILES"] = XMM_FILES
    with open(CONFIG_FILE, 'w') as new_cfg:
        xga_default.write(new_cfg)

    # First time run triggers this message
    raise XGAConfigError("As this is the first time you've used XGA, "
                         "please configure {} to match your setup".format(CONFIG_FILE))

# But if the config file is found, some preprocessing and checks are applied
else:
    xga_conf = ConfigParser()
    # It would be nice to do configparser interpolation, but it wouldn't handle the lists of energy values
    xga_conf.read(CONFIG_FILE)
    keys_to_check = ["root_xmm_dir", "clean_pn_evts", "clean_mos1_evts", "clean_mos2_evts", "attitude_file"]
    # Here I check that the installer has actually changed the three events file paths
    all_changed = all([xga_conf["XMM_FILES"][key] != XMM_FILES[key] for key in keys_to_check])
    if not all_changed:
        raise XGAConfigError("Some events file paths (or the root_xmm_dir) in the config have not "
                             "been changed from default, please configure {} to match your setup".format(CONFIG_FILE))
    elif not os.path.exists(xga_conf["XMM_FILES"]["root_xmm_dir"]):
        raise FileNotFoundError("root_xmm_dir={d} does not appear to exist, if it an SFTP mount check the "
                                "connection.".format(d=xga_conf["XMM_FILES"]["root_xmm_dir"]))

    # Now I do the same for the XGA_SETUP section
    keys_to_check = ["xga_save_path"]
    # Here I check that the installer has actually changed the three events file paths
    all_changed = all([xga_conf["XGA_SETUP"][key] != XGA_CONFIG[key] for key in keys_to_check])
    if not all_changed:
        raise XGAConfigError("You have not changed the xga_save_path value in the config file")
    elif not os.path.exists(xga_conf["XGA_SETUP"]["xga_save_path"]):
        # This is the folder where any files generated by XGA get written
        # Its taken as is from the config file, so it can be absolute, or relative to the project directory
        # Can also be overwritten at runtime by the user, so that's nice innit
        os.makedirs(xga_conf["XGA_SETUP"]["xga_save_path"])

    no_check = ["root_xmm_dir", "lo_en", "hi_en"]
    for key, value in xga_conf["XMM_FILES"].items():
        # Here we attempt to deal with files where people have defined their file paths
        # relative to the root_xmm_dir
        if key not in no_check and xga_conf["XMM_FILES"]["root_xmm_dir"] not in xga_conf["XMM_FILES"][key] \
                and xga_conf["XMM_FILES"][key][0] != '/':
            xga_conf["XMM_FILES"][key] = os.path.join(os.path.abspath(xga_conf["XMM_FILES"]["root_xmm_dir"]),
                                                      xga_conf["XMM_FILES"][key])

    # As it turns out, the ConfigParser class is a pain to work with, so we're converting to a dict here
    # Addressing works just the same
    xga_conf = {str(sect): dict(xga_conf[str(sect)]) for sect in xga_conf}
    try:
        xga_conf["XMM_FILES"]["lo_en"] = to_list(xga_conf["XMM_FILES"]["lo_en"])
        xga_conf["XMM_FILES"]["hi_en"] = to_list(xga_conf["XMM_FILES"]["hi_en"])
    except KeyError:
        raise KeyError("Entries have been removed from config file, "
                       "please leave all in place, even if they are empty")

    # Do a little pre-checking for the energy entries
    if len(xga_conf["XMM_FILES"]["lo_en"]) != len(xga_conf["XMM_FILES"]["hi_en"]):
        raise ValueError("lo_en and hi_en entries in the config "
                         "file do not parse to lists of the same length.")

    # Make sure that this is the absolute path
    xga_conf["XMM_FILES"]["root_xmm_dir"] = os.path.abspath(xga_conf["XMM_FILES"]["root_xmm_dir"]) + "/"
    # Read dataframe of ObsIDs and pointing coordinates into constant
    CENSUS, BLACKLIST = observation_census(xga_conf)
    OUTPUT = os.path.abspath(xga_conf["XGA_SETUP"]["xga_save_path"]) + "/"

    # Make a storage directory where specific source name directories will then be created, there profile objects
    #  created for those sources will be saved
    if not os.path.exists(OUTPUT + "profiles"):
        os.makedirs(OUTPUT + "profiles")

    # Also making a storage directory specifically for products which are combinations of different ObsIDs
    #  and instruments
    if not os.path.exists(OUTPUT + "combined"):
        os.makedirs(OUTPUT + "combined")

    # And create an inventory file for that directory
    if not os.path.exists(OUTPUT + "combined/inventory.csv"):
        with open(OUTPUT + "combined/inventory.csv", 'w') as inven:
            inven.writelines(["file_name,obs_ids,insts,info_key,src_name,type"])

    if "num_cores" in xga_conf["XGA_SETUP"]:
        # If the user has set a number of cores in the config file then we'll use that.
        NUM_CORES = int(xga_conf["XGA_SETUP"]["num_cores"])
    else:
        # Going to allow multi-core processing to use 90% of available cores by default, but
        # this can be over-ridden in individual SAS calls.
        NUM_CORES = max(int(floor(os.cpu_count() * 0.9)), 1)  # Makes sure that at least one core is used

    xmm_sky = def_unit("xmm_sky")
    xmm_det = def_unit("xmm_det")
    # These are largely defined so that I can use them for when I'm normalising profile plots, that way
    #  the view method can just write the units nicely the way it normally does
    r200 = def_unit('r200', format={'latex': r"\mathrm{R_{200}}"})
    r500 = def_unit('r500', format={'latex': r"\mathrm{R_{500}}"})
    r2500 = def_unit('r2500', format={'latex': r"\mathrm{R_{2500}}"})

    # This is a dumb and annoying work-around for a readthedocs problem where units were being added multiple times
    try:
        Quantity(1, 'r200')
    except ValueError:
        add_enabled_units([r200, r500, r2500, xmm_det, xmm_det])

    # Here we check to see whether SAS is installed (along with all the necessary paths)
    SAS_VERSION = None
    if "SAS_DIR" not in os.environ:
        warn("SAS_DIR environment variable is not set, unable to verify SAS is present on system, as such "
             "all functions in xga.sas will not work.")
        SAS_VERSION = None
        SAS_AVAIL = False
    else:
        # This way, the user can just import the SAS_VERSION from this utils code
        sas_out, sas_err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
        SAS_VERSION = sas_out.decode("UTF-8").strip("]\n").split('-')[-1]
        SAS_AVAIL = True

    # This checks for the CCF path, which is required to use cifbuild, which is required to do basically
    #  anything with SAS
    if SAS_AVAIL and "SAS_CCFPATH" not in os.environ:
        warn("SAS_CCFPATH environment variable is not set, this is required to generate calibration files. As such "
             "functions in xga.sas will not work.")
        SAS_AVAIL = False

    # Equivelant for the XSPEC dependency
    XSPEC_VERSION = None
    # Got to make sure we can access command line XSPEC.
    if shutil.which("xspec") is None:
        warn("Unable to locate an XSPEC installation.")
    else:
        try:
            # The XSPEC into text includes the version, so I read that out and parse it
            null_path = pkg_resources.resource_filename(__name__, "xspec_scripts/null_script.xcm")
            xspec_out, xspec_err = Popen("xspec - {}".format(null_path), stdout=PIPE, stderr=PIPE,
                                         shell=True).communicate()
            xspec_vline = [line for line in xspec_out.decode("UTF-8").split('\n') if 'XSPEC version' in line][0]
            XSPEC_VERSION = xspec_vline.split(': ')[-1]
        # I know broad exceptions are a sin, but if anything goes wrong here then XGA needs to assume that XSPEC
        #  is messed up in some way
        except:
            XSPEC_VERSION = None

    # This defines the meaning of different colours of region - this will eventually be user configurable in the
    #  configuration file, but putting it here means that the user can still change the definitions programmatically
    # Definitions of the colours of XCS regions can be found in the thesis of Dr Micheal Davidson
    #  University of Edinburgh - 2005.
    # Red - Point source
    # Green - Extended source
    # Magenta - PSF-sized extended source
    # Blue - Extended source with significant point source contribution
    # Cyan - Extended source with significant Run1 contribution
    # Yellow - Extended source with less than 10 counts
    SRC_REGION_COLOURS = {'pnt': ["red"], 'ext': ["green", "magenta", "blue", "cyan", "yellow"]}


