#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 5/13/26, 9:59 PM. Copyright (c) The Contributors.

import importlib
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from configparser import ConfigParser
from functools import wraps
from subprocess import Popen, PIPE
from typing import Tuple, List, Union
from warnings import warn, simplefilter

import numpy as np
import pandas as pd
from astropy.constants import m_p, m_e
from astropy.cosmology import LambdaCDM
from astropy.io import fits
from astropy.units import Quantity, def_unit, add_enabled_units, add_enabled_equivalencies
from astropy.wcs import WCS
from fitsio import FITSHDR
from packaging.version import Version
from tqdm import tqdm

from .exceptions import XGAConfigError, InvalidTelescopeError, NoTelescopeDataError

# This warning filter enables the DeprecationWarning which is in the _deprecated decorator
# We set it to once so that the same warning is not shown multiple times in the same session,
#  important for parallel processing where the module may be imported multiple times.
simplefilter('once', DeprecationWarning)

_INITIALISED = False
# The set of variables that should be lazily loaded
_LAZY_VARS = {
    'CONFIG_PATH', 'CONFIG_FILE', 'xga_conf', 'CENSUS', 'BLACKLIST',
    'COMBINED_INSTS', 'SAS_AVAIL', 'SAS_VERSION', 'ESASS_AVAIL',
    'ESASS_VERSION', 'CIAO_AVAIL', 'CIAO_VERSION', 'CALDB_AVAIL',
    'CALDB_VERSION', 'XSPEC_VERSION', 'OUTPUT', 'NUM_CORES',
    'USABLE', 'VALID_CONFIG', 'CENSUS_FILES', 'BLACKLIST_FILES',
    'ABUND_TABLES'
}
# This set contains any variables from _LAZY_VARS that should NOT be exposed at the top level
#  of the xga module (i.e. via xga.VAR_NAME). This allows for internal lazy variables in utils.py.
# Any variable name in this set must also be present in _LAZY_VARS
_KEEP_PRIVATE_LAZY_VARS = set()


def reinitialise_xga(config_dir: str = None):
    """
    A function to re-initialise XGA, allowing the user to change the configuration directory
    mid-session, or to pick up changes to the configuration file.

    :param str config_dir: The new configuration directory to use. If None, the current
        XGA_CONFIG_DIR environment variable (or default) will be used.
    """
    global _INITIALISED
    # If a new config directory is passed, we update the environment variable
    if config_dir is not None:
        os.environ['XGA_CONFIG_DIR'] = os.path.abspath(config_dir)

    # We remove the lazy variables from the module namespace so that __getattr__
    #  is triggered again on next access.
    for var in _LAZY_VARS:
        if var in globals():
            del globals()[var]

    # Reset the initialised flag
    _INITIALISED = False


def __getattr__(name):
    """
    A module level __getattr__ which allows us to lazily load the XGA configuration and census, as well as
    check for the availability of backend software. This helps to avoid race conditions when importing
    XGA from other modules (like DAXA).

    :param str name: The name of the attribute to be returned.
    :return: The value of the attribute.
    """
    global _INITIALISED
    if name in _LAZY_VARS:
        if not _INITIALISED:
            _initialise_xga()

        # SOFTWARE VERSION DISCOVERY LOGIC
        # We only perform these checks if they haven't been done yet
        if name in ['SAS_VERSION', 'SAS_AVAIL']:
            global SAS_VERSION, SAS_AVAIL
            if 'SAS_VERSION' not in globals() or SAS_VERSION is None:
                SAS_VERSION, SAS_AVAIL = _get_sas_info()
            return globals()[name]

        if name in ['ESASS_VERSION', 'ESASS_AVAIL']:
            global ESASS_VERSION, ESASS_AVAIL
            if 'ESASS_VERSION' not in globals() or ESASS_VERSION is None:
                ESASS_VERSION, ESASS_AVAIL = _get_esass_info()
            return globals()[name]

        if name in ['CIAO_VERSION', 'CIAO_AVAIL', 'CALDB_VERSION', 'CALDB_AVAIL']:
            global CIAO_VERSION, CIAO_AVAIL, CALDB_VERSION, CALDB_AVAIL
            if 'CIAO_VERSION' not in globals() or CIAO_VERSION is None:
                CIAO_VERSION, CIAO_AVAIL, CALDB_VERSION, CALDB_AVAIL = _get_ciao_info()
            return globals()[name]

        if name in ['XSPEC_VERSION', 'XSPEC_FIT_METHOD', 'ABUND_TABLES']:
            global XSPEC_VERSION, XSPEC_FIT_METHOD, ABUND_TABLES
            if 'XSPEC_VERSION' not in globals() or XSPEC_VERSION is None:
                XSPEC_VERSION, XSPEC_FIT_METHOD, ABUND_TABLES = _get_xspec_info()
            return globals()[name]

        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _get_sas_info() -> Tuple[Union[Version, None], bool]:
    """
    Internal helper to discover SAS version and availability. Only triggered if XMM is usable.
    """
    # We only care to check for software if the mission is actually configured
    if not USABLE.get('xmm', False):
        return None, False

    sas_version = None
    sas_avail = False
    # Next up, we check to see what version of SAS (if any) is installed - for the XMM-Newton mission
    # Here we check to see whether SAS is installed (along with all the necessary paths)
    if "SAS_DIR" not in os.environ:
        warn("SAS_DIR environment variable is not set, unable to verify SAS is present on system, as such "
             "all functions in xga.sas will not work.", stacklevel=3)
    else:
        # This way, the user can just import the SAS_VERSION from this utils code
        sas_out, sas_err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
        ver_str = sas_out.decode("UTF-8").strip("]\n")

        # The ver_str can contain info about sas (gui) or other things, we want the part in brackets or the end
        if '[' in ver_str:
            ver_str = ver_str.split('[')[-1].split(']')[0]

        # This is an unfortunate hard coding of parsing SAS version from the return of sas --version
        #  It seems the version string structure changed in SAS 21/22, so we try to catch that here
        # This is how SAS 20 needs to be treated
        if ver_str[:6] == 'xmmsas':
            sas_version = Version(ver_str.split('-')[-1])
        # And hopefully this is how everything else needs to be treated
        else:
            sas_version = Version(ver_str.split('-')[0])
        sas_avail = True

    # This checks for the CCF path, which is required to use cifbuild, which is required to do basically
    #  anything with SAS
    if sas_avail and "SAS_CCFPATH" not in os.environ:
        warn("SAS_CCFPATH environment variable is not set, this is required to generate calibration files. As such "
             "functions in xga.sas will not work.", stacklevel=3)
        sas_avail = False

    return sas_version, sas_avail


def _get_esass_info() -> Tuple[Union[str, None], bool]:
    """
    Internal helper to discover eSASS version and availability. Only triggered if eROSITA is usable.
    """
    if not (USABLE.get('erosita', False) or USABLE.get('erass', False)):
        return None, False

    esass_version = None
    esass_avail = False

    # Run the 'which' command for evtool, one of the eSASS tasks
    which_evtool = shutil.which("evtool")

    # This checks for an installation of eSASS
    if which_evtool is None:
        warn("No eSASS installation detected on system, as such all functions in xga.generate.esass will not work.",
             stacklevel=3)
    else:
        esass_avail = True
        # Version strings for eSASS are handled a bit differently as they're not always standard versions
        # but we'll try to wrap them in Version objects regardless for consistency
        if 'ESASS4EDR' in which_evtool.upper():
            # Version('0.1.0')  # Dummy numeric version for EDR
            esass_version = "ESASS4EDR"
        elif 'ESASS4DR1' in which_evtool.upper():
            # Version('1.0.0')  # Dummy numeric version for DR1
            esass_version = "ESASS4DR1"
        else:
            warn("Unknown eSASS installation detected on system, as such some functions in "
                 "xga.generate.esass may not work.", stacklevel=3)

    return esass_version, esass_avail


def _get_ciao_info() -> Tuple[Union[Version, None], bool, Union[Version, None], bool]:
    """
    Internal helper to discover CIAO version and availability. Only triggered if Chandra is usable.
    """
    if not USABLE.get('chandra', False):
        return None, False, None, False

    ciao_version = None
    ciao_avail = False
    caldb_version = None
    caldb_avail = False

    ciao_out, ciao_err = Popen("ciaover -v", stdout=PIPE, stderr=PIPE, shell=True).communicate()
    # Just turn those pesky byte outputs into strings
    ciao_out = ciao_out.decode("UTF-8")
    ciao_err = ciao_err.decode("UTF-8")

    if "ciaover: command not found" in ciao_err:
        warn("No CIAO installation detected on system, "
             "as such all functions in xga.generate.ciao will not work.", stacklevel=3)
    else:
        # The ciaover output is over a series of lines, with different info on each
        split_out = [en.strip(' ') for en in ciao_out.split('\n')]
        # Strip the CIAO version out of the ciaover output
        ciao_version = Version(split_out[1].split(':')[-1].split('CIAO')[-1].strip(' ').split(' ')[0])
        ciao_avail = True

        # Finally, we check to see what version of CALDB (if any) is installed
        if 'not installed' in split_out[5].lower():
            warn("A Chandra CALDB installation cannot be identified on your system, and as such "
                 "Chandra data cannot be processed.", stacklevel=3)
        else:
            # Strip out the CALDB version
            caldb_version = Version(split_out[5].split(':')[-1].strip())
            caldb_avail = True

    return ciao_version, ciao_avail, caldb_version, caldb_avail


def _get_xspec_info() -> Tuple[Union[Version, None], List[str], List[str]]:
    """
    Internal helper to discover XSPEC version and availability.
    """
    xspec_version = None
    # Fit methods and abundance tables are static, but we return them here to keep things grouped
    fit_methods = ["leven", "migrad", "simplex"]
    abund_tables = ["feld", "angr", "aneb", "grsa", "wilm", "lodd", "aspl"]

    if shutil.which("xspec") is None:
        warn("Unable to locate an XSPEC installation.", stacklevel=3)
    else:
        try:
            # null_script.xcm does absolutely nothing, it's just a way to get the version out
            null_path = importlib.resources.files(__name__) / "xspec_scripts/null_script.xcm"
            xspec_out, xspec_err = Popen("xspec - {}".format(null_path), stdout=PIPE, stderr=PIPE,
                                         shell=True).communicate()
            # Got to parse the stdout to get the XSPEC version
            xspec_vline = [line for line in xspec_out.decode("UTF-8").split('\n') if 'XSPEC version' in line][0]
            xspec_version = Version(xspec_vline.split(': ')[-1])
        except (Exception,):
            xspec_version = None

    return xspec_version, fit_methods, abund_tables


def _initialise_xga():
    """
    The internal function that actually performs the XGA configuration and census loading, as well as
    checking for the availability of backend software.
    """
    global _INITIALISED, CONFIG_PATH, CONFIG_FILE, xga_conf, VALID_CONFIG, USABLE, CENSUS, BLACKLIST, \
        COMBINED_INSTS, SAS_VERSION, SAS_AVAIL, ESASS_VERSION, ESASS_AVAIL, CIAO_VERSION, CIAO_AVAIL, \
        CALDB_VERSION, CALDB_AVAIL, XSPEC_VERSION, OUTPUT, NUM_CORES, CENSUS_FILES, BLACKLIST_FILES, \
        SASERROR_LIST, SASWARNING_LIST, XSPEC_FIT_METHOD, ABUND_TABLES

    if 'XGA_CONFIG_DIR' in os.environ:
        CONFIG_PATH = os.path.abspath(os.environ['XGA_CONFIG_DIR'])
    else:
        CONFIG_PATH = os.path.join(os.environ.get('XDG_CONFIG_HOME',
                                                 os.path.join(os.path.expanduser('~'), '.config')), 'xga')

    if not os.path.exists(CONFIG_PATH):
        os.makedirs(CONFIG_PATH)

    CONFIG_FILE = os.path.join(CONFIG_PATH, 'xga.cfg')

    # These are used for the observation census files
    CENSUS_FILES = {tel: os.path.join(CONFIG_PATH, tel, '{}_census.csv'.format(tel)) for tel in ALLOWED_INST}
    BLACKLIST_FILES = {tel: os.path.join(CONFIG_PATH, tel, '{}_blacklist.csv'.format(tel)) for tel in ALLOWED_INST}

    USABLE = {tele: False for tele in TELESCOPES}
    VALID_CONFIG = {tel: False for tel in TELESCOPES}
    # ------------- Creating/checking the entries in the configuration file -------------
    # This chunk of utils will be dedicated to making sure that the configuration file has been created (by default with
    #  sections for every telescope that XGA supports), or that if it already exists it contains valid entries.
    # THIS STAGE USED TO FAIL ENTIRELY IF THERE WEREN'T VALID ENTRIES - but now I realise that sometimes you just want
    #  to use the XGA products with your own data files without using the source/sample classes, so it will no longer fail
    #  at this stage.

    # In this case we find that the configuration file does not exist, and we set it up using the default sections and
    #  configurations that were set up toward the top of this file
    if not os.path.exists(CONFIG_FILE):
        # Define a configuration object
        xga_default = ConfigParser()
        # This adds the overall XGA setup section - controls global things like where XGA generated files are stored, and
        #  how many cores XGA is allowed to use (though that can also be set at import time).
        xga_default.add_section("XGA_SETUP")
        xga_default["XGA_SETUP"] = XGA_CONFIG

        # Now we cycle through all the telescopes supported by XGA, adding their default sections to the configuration.
        #  I now regret calling the sections {telescope}_FILES, but honestly it doesn't bother me enough to endure the
        #  hassle of changing it now - it would be a breaking change for any existing installations
        for tel in TELESCOPES:
            cur_sec_name = "{}_FILES".format(tel.upper())
            xga_default.add_section(cur_sec_name)
            xga_default[cur_sec_name] = tele_conf_sects[tel]

        # The default configuration file is now written to the path that we expect to find the config file at
        with open(CONFIG_FILE, 'w') as new_cfg:
            xga_default.write(new_cfg)

        # First time run triggers this message - it used to be an error, and so XGA wouldn't advance beyond this point
        #  with a new configuration file, but we want people to be able to use the product classes without configuring
        warn("This is the first time you've used XGA; to use most functionality you will need to configure {} to match "
             "your setup, though you can use product classes regardless.".format(CONFIG_FILE), stacklevel=2)

    xga_conf = ConfigParser()
    # It would be nice to do configparser interpolation, but it wouldn't handle the lists of energy values
    xga_conf.read(CONFIG_FILE)

    # If the current section name exists in xga_conf then all is well, there hasn't been an update to XGA which added
    #  a new telescope installed - however if a section is missing then we need to add it. This is slightly inelegant
    #  because I also iterate through the telescopes slightly further down, but it is just easier for this to go here
    #  as I need to write the updated configuration file to disk, and I'm about to turn it into a dictionary so it is
    #  easier to do that here
    # Set up a flag, so we can know if any sections have been added
    altered = False
    for tel in TELESCOPES:
        cur_sec_name = "{}_FILES".format(tel.upper())
        if cur_sec_name not in xga_conf:
            # If there isn't already a files section for one of the telescopes now supported by XGA, then we add it to
            #  the existing configuration file
            xga_conf.add_section(cur_sec_name)
            xga_conf[cur_sec_name] = tele_conf_sects[tel]
            altered = True
    # If we altered the existing configuration file, then we need to save the altered configuration to disk
    if altered:
        with open(CONFIG_FILE, 'w') as update_cfg:
            xga_conf.write(update_cfg)

    # As it turns out, the ConfigParser class is a pain to work with, so we're converting to a dict here
    # Addressing works just the same
    xga_conf = {str(sect): dict(xga_conf[str(sect)]) for sect in xga_conf}

    # ------------- Final setup of important constants from the configuration file -------------
    # We make sure to create the absolute output path from what was specified in the configuration file
    OUTPUT = os.path.abspath(xga_conf["XGA_SETUP"]["xga_save_path"]) + "/"

    # The default behaviour for the generation of new configuration files is to set the num_cores entry to 'auto',
    #  though that isn't a given with older configuration files - as such we check and don't assume it will be there.
    #  If the num_cores keyword is present and ISN'T auto then we use the user specified core count
    if "num_cores" in xga_conf["XGA_SETUP"] and xga_conf["XGA_SETUP"]["num_cores"] != "auto":
        # If the user has set a number of cores in the config file then we'll use that.
        NUM_CORES = int(xga_conf["XGA_SETUP"]["num_cores"])
    # In this case though the user has not specified a number of cores to use thus we will use 90% of the available
    #  cores on the system
    else:
        # Going to allow multi-core processing to use 90% of available cores by default, but
        # this can be over-ridden in individual SAS calls.
        NUM_CORES = max(int(np.floor(os.cpu_count() * 0.9)), 1)  # Makes sure that at least one core is used
    # ------------------------------------------------------------------------------------------

    # Firstly, we check if the entries in the general XGA configuration section are valid - the output path is the first to
    #  check, though we are actually only checking for a very unlikely edge case. That somebody is using this new version
    #  of XGA with an old, unmodified, configuration file.
    if xga_conf['XGA_SETUP']['xga_save_path'] == "/this/is/required/xga_output/":
        raise XGAConfigError("The 'xga_save_path' entry is currently '/this/is/required/xga_output/', which at one point"
                             " was the default entry in new configuration files; please change it before proceeding.")

    # Optionally there can be a num_cores entry in the overall settings section, and if there is we check that it is
    #  a valid value - though we don't throw an error if it isn't, we just default to XGA automatically determining
    #  the number of cores to use.
    if 'num_cores' in xga_conf['XGA_SETUP'] and xga_conf['XGA_SETUP']['num_cores'] != 'auto' and \
            not isinstance(xga_conf['XGA_SETUP']['num_cores'], int):
        warn("The 'num_cores' entry ({}) in the configuration file is not valid, it should either be an integer or "
             "auto; here we default to 'auto'.".format(xga_conf['XGA_SETUP']['num_cores']), stacklevel=2)
        xga_conf['XGA_SETUP']['num_cores'] = 'auto'

    # Now we check the telescope-specific sections
    for tel in TELESCOPES:
        cur_sec_name = "{}_FILES".format(tel.upper())
        cur_sec = xga_conf[cur_sec_name]

        # The upper and lower energy bounds defined in the config file for existing image/exposure maps files are a
        #  string representation of a list, and we want to turn them back into an actual list
        poss_ens = ['lo_en', 'hi_en']
        if sum([en in cur_sec for en in poss_ens]) == 1:
            raise XGAConfigError("Both lo_en and hi_en entries must be present, not one or the other.")
        # This just converts the string representation of the energy list into an actual list of energies
        elif sum([en in cur_sec for en in poss_ens]) == 2:
            for en_conf in poss_ens:
                in_parts = cur_sec[en_conf].strip("[").strip("]").split(',')
                real_list = [part.strip(' ').strip("'").strip('"') for part in in_parts if part != '' and part != ' ']
                cur_sec[en_conf] = real_list

        # Now we check that the directory we're pointed to for the root data directory of the current telescope
        #  actually exists
        # This variable keeps track of if the root_dir for this telescope actually exists
        root_dir_exists = False
        if os.path.exists(cur_sec['root_{t}_dir'.format(t=tel)]):
            root_dir_exists = True

        # This is a pretty blunt-force approach, but honestly I think it should work fine consider we just want to
        #  check whether any of the required sections have been left as the default values (meaning that telescope
        #  hasn't been configured and can't be used).
        # We use this 'all_req_changed' flag to store if any of the required entries have been left at their default
        #  values - even one of these being left at default means we can't use this telescope.
        all_req_changed = True
        # As we're already iterating through the entries in this section we will also check to see if the file paths
        #  have been defined relative to the root directory, so we'll define this list of entries not to check
        #  for that
        no_check = poss_ens + ['root_{t}_dir'.format(t=tel)]
        for entry in cur_sec:
            if "/this/is/required/" in cur_sec[entry]:
                any_req_defaults = False
            elif (entry not in no_check and cur_sec['root_{t}_dir'.format(t=tel)] not in cur_sec[entry] and
                  cur_sec[entry][0] != '/'):
                # Replace the current definition with an absolute one s
                cur_sec[entry] = os.path.join(os.path.abspath(cur_sec['root_{t}_dir'.format(t=tel)]), cur_sec[entry])

        # We make sure that the root directory is an absolute path, just for our sanity later on
        cur_sec['root_{t}_dir'.format(t=tel)] = os.path.abspath(cur_sec['root_{t}_dir'.format(t=tel)]) + "/"

        # This tells the rest of XGA that the current telescope is usable! If these conditions aren't fulfilled then
        #  the USABLE entry for the current telescope will stay at the default value of False
        if all_req_changed and root_dir_exists:

            USABLE[tel] = True
    # -----------------------------------------------------------------------------------


    # ------------- Generating the observation censuses for all USABLE telescopes -------------
    # Read dataframe of ObsIDs and pointing coordinates into dictionaries
    CENSUS = {}
    BLACKLIST = {}
    # Also create a dictionary that tells parts of XGA whether it should expect the event lists of the different
    #  instruments to be separate, or combined (I'm looking at you eROSITA CalPV).
    COMBINED_INSTS = {}

    # Checking if someone had been using the XMM only version of XGA previously - with this update to implement the
    #  infrastructure to support different telescopes the census/blacklist files will exist for EACH telescope
    #  individually
    old_census_path = os.path.join(CONFIG_PATH, 'census.csv')
    if os.path.exists(old_census_path):
        # Changing the .config/xga directory to the updated structure for the multi-telescope version of xga
        os.makedirs(CONFIG_PATH + '/xmm/')
        new_xmm_census_path = os.path.join(CONFIG_PATH, 'xmm', 'xmm_census.csv')
        # Moving the xmm census to the correct place and renaming it with the updated naming scheme
        shutil.move(old_census_path, new_xmm_census_path)
        # Doing the same for the blacklist
        old_bl_path = os.path.join(CONFIG_PATH, 'blacklist.csv')
        new_bl_path = os.path.join(CONFIG_PATH, 'xmm', 'xmm_blacklist.csv')
        shutil.move(old_bl_path, new_bl_path)

    for tel in USABLE:
        # We only care to have/make a census if the telescope is actually set up and usable
        if USABLE[tel]:
            CENSUS[tel], BLACKLIST[tel] = build_observation_census(tel, num_cores=NUM_CORES)

        # Populate the dictionary that says whether the event lists for a given telescope are combined or not - it would
        #  have been so much easier if they were all always separate, but the eROSITA CalPV ones weren't released like
        #  that and I bet eRASS won't be either
        if 'clean_{}_evts'.format(tel.lower()) in xga_conf['{}_FILES'.format(tel.upper())]:
            COMBINED_INSTS[tel] = True
        else:
            COMBINED_INSTS[tel] = False
    # -----------------------------------------------------------------------------------------

    _INITIALISED = True


def _deprecated(message):
    """
    An internal function designed to be used as a decorator for any methods or functions which have been
    deprecated - means that the warning will be shown when they're imported or used.

    :param str message: The warning message which should be shown.
    """
    def deprecated_function(dep_func):
        """
        This ensures that the decorated, deprecated, function is actually still run.
        :param dep_func: The method which is deprecated.
        :return: Returns the wrapped function.
        """
        # The wraps decorator updates the wrapper function to look like wrapped function by copying attributes
        #  such as __name__, __doc__ (the docstring)
        @wraps(dep_func)
        def wrapper(*args, **kwargs):
            # Shows the warning message - this makes sure it is shown when the function is used.
            warn(message, DeprecationWarning, stacklevel=2)
            return dep_func(*args, **kwargs)

        return wrapper
    return deprecated_function



# ------------- Defining functions useful in the rest of the setup process -------------
def rebuild_census(telescope: Union[str, List[str]] = None, full_rebuild: bool = False,
                   clean_dead: bool = False, num_cores: int = None):
    """
    A function to manually trigger a census rebuild for specific or all telescopes. This can be used to pick up
    new observations that have been added to the data directories, or to clean up the census if observations
    have been removed.

    :param str/List[str] telescope: The telescope(s) for which the census should be rebuilt. If None, then all
        usable telescopes will be rebuilt.
    :param bool full_rebuild: If True, the existing census file will be deleted and rebuilt entirely from scratch.
    :param bool clean_dead: If True, the census will be checked for entries that no longer have a corresponding
        ObsID directory in the data path, and those entries will be removed.
    :param int num_cores: The number of cores to use for parallel header extraction.
        Defaults to NUM_CORES.
    """
    # This triggers the initial setup if it hasn't happened yet
    from xga import USABLE, CENSUS, BLACKLIST, NUM_CORES as global_nc

    if num_cores is None:
        num_cores = global_nc

    # Standard check of telescope choices
    telescope = check_telescope_choices(telescope)

    for tel in telescope:
        # We only care to have/make a census if the telescope is actually set up and usable
        if USABLE[tel]:
            census_file = os.path.join(CONFIG_PATH, tel, "{}_census.csv".format(tel))

            if full_rebuild:
                # Means we delete the existing census file and start again
                if os.path.exists(census_file):
                    os.remove(census_file)

            # This handles both the full rebuild (since we just deleted the file)
            # and the standard 'find new data' update
            CENSUS[tel], BLACKLIST[tel] = build_observation_census(tel, num_cores=num_cores)

            if clean_dead:
                # Cleanup of dead entries
                census = CENSUS[tel]
                rel_root_dir = xga_conf[tel.upper() + '_FILES']['root_{t}_dir'.format(t=tel)]

                # We get a list of all directories in the root directory that look like ObsIDs
                # We use a set for O(1) lookup efficiency
                current_obs = {poss_oi for poss_oi in os.listdir(rel_root_dir)
                               if os.path.isdir(os.path.join(rel_root_dir, poss_oi)) and obs_id_test(tel, poss_oi)}

                # Filter the census to only keep rows where the ObsID is still present
                original_len = len(census)
                # We access the ObsID column
                census = census[census['ObsID'].isin(current_obs)]

                if len(census) != original_len:
                    # Update the internal CENSUS dictionary
                    CENSUS[tel] = census
                    # Save the cleaned census back to disk
                    census.to_csv(census_file, index=False)
                    warn(f"Cleaned {original_len - len(census)} dead entries from {tel} census.", stacklevel=2)


def obs_id_test(test_tele: str, test_string: str) -> bool:
    """
    This function uses regular expressions for the structure of different telescope's ObsIDs to check that a
    given string conforms with the structure expected for an ObsID.

    :param str test_tele: The telescope for the ObsID we wish to check, different missions have different
        ObsID structures.
    :param str test_string: The string we wish to test.
    :return: Whether the string an ObsID from the specified telescope or not.
    :rtype: bool
    """
    # Just in case an integer ObsID is passed, we'll try to catch it here
    if not isinstance(test_string, str):
        test_string = str(test_string)

    # This uses a dictionary of ObsID regex patterns defined in the telescope data section at the top of this
    #  file to check that the given test string is a match to the structure defined by the regex for this telescope
    return bool(re.match(OBS_ID_REGEX[test_tele], test_string))


def _extract_header_info(obs: str, tel: str, evt_path_keys: List[str], evt_path_insts: List[str],
                         inst_from_evt: bool, tele_conf: dict):
    """
    An internal helper function to extract pointing and instrument usability information from
    a single observation's event lists. This is designed to be run in parallel.

    :param str obs: The ObsID to extract info for.
    :param str tel: The telescope name.
    :param List[str] evt_path_keys: The configuration keys for event lists.
    :param List[str] evt_path_insts: The instrument names associated with those keys.
    :param bool inst_from_evt: Whether instrument info should be pulled from the event list header.
    :param dict tele_conf: The configuration dictionary for the specific telescope.
    :return: A dictionary of information for the census.
    :rtype: dict
    """
    # Set up the data that will be added to the census for the current observation
    info = {'ObsID': obs}

    # Iterating through the identified event list keys in the config for the current telescope
    for evt_key_ind, evt_key in enumerate(evt_path_keys):
        evt_path = tele_conf[evt_key].format(obs_id=obs)

        if os.path.exists(evt_path):
            # Just read in the header of the events file - want to avoid reading a big old table of
            #  events into memory, as we might be doing this a bunch of times
            try:
                # Using getheader is optimized for just grabbing the header
                evts_header = fits.getheader(evt_path, extname='EVENTS')
            except (Exception,):
                # If anything goes wrong, we assume the file is corrupted or unusable
                if inst_from_evt:
                    for i in ALLOWED_INST[tel]:
                        info['USE_{}'.format(i.upper())] = 'F'
                else:
                    info['USE_{}'.format(evt_path_insts[evt_key_ind].upper())] = 'F'
                continue

            # pointing coordinates
            if tel in ['erosita', 'erass'] and evts_header.get('OBS_MODE') == 'SURVEY':
                if evts_header.get('RA_CEN', 0.0) == 0.0 and evts_header.get('RA_OBJ', 0.0) != 0.0:
                    info['RA_PNT'] = evts_header.get('RA_OBJ')
                else:
                    info['RA_PNT'] = evts_header.get("RA_CEN")

                if evts_header.get('DEC_CEN', 0.0) == 0.0 and evts_header.get('DEC_OBJ', 0.0) != 0.0:
                    info['DEC_PNT'] = evts_header.get('DEC_OBJ')
                else:
                    info['DEC_PNT'] = evts_header.get("DEC_CEN")
            else:
                info['RA_PNT'] = evts_header.get("RA_PNT")
                info['DEC_PNT'] = evts_header.get("DEC_PNT")

            # Filter check
            if 'FILTER' in evts_header:
                good_filt = evts_header['FILTER'] not in BANNED_FILTS[tel]
            else:
                good_filt = True

            if inst_from_evt:
                hdr_insts = [evts_header[h_key] for h_key in list(evts_header.keys())
                             if 'INSTRUM' in h_key and 'INSTRUME' not in h_key]
                for i in ALLOWED_INST[tel]:
                    use_key = 'USE_{}'.format(i.upper())
                    info[use_key] = 'T' if (i.upper() in hdr_insts and good_filt) else 'F'
            else:
                use_key = 'USE_{}'.format(evt_path_insts[evt_key_ind].upper())
                info[use_key] = 'T' if good_filt else 'F'

        else:
            # If the file path doesn't exist then we have to set the usable column(s) to False!
            if inst_from_evt:
                for i in ALLOWED_INST[tel]:
                    info['USE_{}'.format(i.upper())] = 'F'
            else:
                info['USE_{}'.format(evt_path_insts[evt_key_ind].upper())] = 'F'

    return info


def build_observation_census(tel: str, num_cores: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A function that builds/updates the census and blacklist for each telescope.

    :param str tel: The name of the telescope we are setting up a census/blacklist for.
    :param int num_cores: The number of cores to use for parallel header extraction.
    :return: The census and blacklist dataframes for the input telescope.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame]
    """
    # The census_dir is the directory containing the blacklist and census for each telescope
    census_dir = os.path.join(CONFIG_PATH, tel) + '/'
    # If it doesn't exist yet we create the directory
    if not os.path.exists(census_dir):
        os.makedirs(census_dir)

    # This variable stores the path to the blacklist file
    blacklist_file = os.path.join(CONFIG_PATH, tel, "{}_blacklist.csv".format(tel))
    # This extracts the relevant list of the instruments for this telescope
    rel_insts = ALLOWED_INST[tel]
    # Creates blacklist file if one doesn't exist, then reads it in
    if not os.path.exists(blacklist_file):
        inst_list = ["EXCLUDE_{}".format(inst.upper()) for inst in rel_insts]
        blacklist = pd.DataFrame(columns=["ObsID"] + inst_list)
        blacklist.to_csv(blacklist_file, index=False)
    # If a blacklist already exists then of course we don't need to make one, just read it in
    else:
        blacklist = pd.read_csv(blacklist_file, header="infer", dtype=str)

    # This part here is to support blacklists used by older versions of XGA, where only a full ObsID was excluded.
    if len(blacklist.columns) == 1:
        blacklist_columns = ["EXCLUDE_{}".format(inst.upper()) for inst in rel_insts]
        blacklist[blacklist_columns] = 'T'
        blacklist.to_csv(blacklist_file, index=False)

    # This variable stores the path to the census file for this telescope
    census_file = os.path.join(CONFIG_PATH, tel, "{}_census.csv".format(tel))
    # If CENSUS FILE exists, it is read in, otherwise an empty dataframe is initialised
    if os.path.exists(census_file):
        obs_lookup = pd.read_csv(census_file, header="infer", dtype=str)
    else:
        # Making the columns in the census
        inst_list = ["USE_{}".format(inst.upper()) for inst in rel_insts]
        obs_lookup = pd.DataFrame(columns=["ObsID", "RA_PNT", "DEC_PNT"] + inst_list)

    # Need to find out which observations are available. Using scandir is more efficient
    #  than listdir + isdir.
    tele_conf = xga_conf[tel.upper() + '_FILES']
    rel_root_dir = tele_conf['root_{t}_dir'.format(t=tel)]
    existing_obs = set(obs_lookup['ObsID'].values)

    new_obs_census = []
    with os.scandir(rel_root_dir) as entries:
        for entry in entries:
            if entry.is_dir() and obs_id_test(tel, entry.name) and entry.name not in existing_obs:
                new_obs_census.append(entry.name)

    if len(new_obs_census) != 0:
        evt_path_keys = [e_key for e_key in tele_conf if 'evts' in e_key and 'clean' in e_key]
        evt_path_insts = [e_key.split('_')[1] for e_key in evt_path_keys]
        inst_from_evt = False if len(evt_path_insts) == len(ALLOWED_INST[tel]) else True

        # Run extraction in parallel
        new_census_rows = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Create a list of futures
            futures = [executor.submit(_extract_header_info, obs, tel, evt_path_keys, evt_path_insts,
                                       inst_from_evt, tele_conf) for obs in new_obs_census]

            with tqdm(desc="Assembling list of {} ObsIDs".format(tel), total=len(new_obs_census)) as census_progress:
                for future in futures:
                    new_census_rows.append(future.result())
                    census_progress.update(1)

        # We add the new observations into our existing census dataframe
        new_data = pd.DataFrame(new_census_rows, columns=obs_lookup.columns)
        obs_lookup = pd.concat([obs_lookup, new_data], ignore_index=True)
        # This then saves the dataframe to its rightful place
        obs_lookup.to_csv(census_file, index=False)

    # We do actually convert some values from what they are stored in the file as, to Python bools
    obs_lookup["RA_PNT"] = obs_lookup["RA_PNT"].replace('', np.nan).astype(float)
    obs_lookup["DEC_PNT"] = obs_lookup["DEC_PNT"].replace('', np.nan).astype(float)

    # Adding in columns for the instruments
    rel_inst_cols = ["USE_{}".format(inst.upper()) for inst in rel_insts]
    obs_lookup[rel_inst_cols] = obs_lookup[rel_inst_cols].apply(lambda x: x == 'T')

    # Finally we return the census and the blacklist
    return obs_lookup, blacklist


# This function also now exists in imagetools.miss, which is where it will live forevermore - I wouldn't normally
#  duplicate a whole function just to show a deprecation warning, but this function isn't going to change so it is
#  safe enough
@_deprecated(message="The XGA 'find_all_wcs' function should be imported from imagetools.misc, in the future "
                     "it will be removed from utils.")
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


# Very handy function that is used in several places - just not really sure where else to put it but here
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


def check_telescope_choices(telescope: Union[str, List[str]]) -> List[str]:
    """
    This function centralises some checks that might be made in various places in the module. The checks are of
    the choices which the user has made regarding which telescopes to use for a function, and it makes sure that the
    chosen telescopes are valid names (i.e. they are real and supported by XGA), that the data have been set up with
    XGA, and makes sure that a list of strings is returned (even if there is only one) so that the output is always
    consistent.

    :param str/List[str] telescope: The telescope choice made by the user - the formatting and validity of which will
        be checked by this function.
    :return: A list of telescope names (or a single name) that are valid and that have made it through the checks.
    :rtype: List[str]
    """
    # If the telescope is set to None then the function will search all the telescopes for matching data, determining
    #  which are available by grabbing all the keys from the CENSUS dictionary
    if telescope is None:
        telescope = list(CENSUS.keys())
        # If there are no keys in the CENSUS dictionary then we know that no telescopes have been successfully
        #  setup with XGA and this function isn't going to work
        if len(telescope) == 0:
            raise NoTelescopeDataError("No telescope data is currently available to XGA.")

    # Just making sure the telescope names supplied by the user are lowercase, as that is how they are stored in
    #  XGA constants and products - also have to account for the fact that either a single string or a list
    #  can be passed
    if not isinstance(telescope, list):
        telescope = [telescope.lower()]
    else:
        telescope = [t.lower() for t in telescope]

    # Here we check if ANY of the passed telescopes aren't actually recognised by XGA, as I want to tell them
    #  that they have either made a typo or are labouring under a misconception about which telescopes are
    #  supported
    if any([t not in TELESCOPES for t in telescope]):
        which_bad = [t for t in telescope if t not in TELESCOPES]
        raise InvalidTelescopeError("XGA does not support the following telescopes; "
                                    "{}".format(', '.join(which_bad)))
    # If the user made specific requests of telescope, and they are ALL not available, we throw an error
    elif all([not USABLE[t] for t in telescope]):
        raise NoTelescopeDataError("None of the requested telescopes ({}) have data available to "
                                   "XGA.".format(', '.join(telescope)))
    # However if the user made specific requests of telescope, and SOME are not available then they get a warning
    elif any([not USABLE[t] for t in telescope]):
        # This isn't elegant, but oh well - we have to make sure that we only let those telescopes through
        #  that have actually been set up and are working with XGA
        which_bad = [t for t in telescope if not USABLE[t]]
        telescope = [t for t in telescope if USABLE[t]]
        warn("Some requested telescopes are not currently set up with XGA; {}".format(", ".join(which_bad)),
             stacklevel=2)

    return telescope
# --------------------------------------------------------------------------------------


# ------------- Defining constants to do with backend software -------------
# Various parts of XGA can rely on different pieces of backend software - checking
#  for their presence, and determining their versions etc. happens when a
#  relevant constant is accessed.
# We still need to define some constants that can be static though - defining them/
#  reading the information in on demand would be overkill for this

# Here we read in files that list the errors and warnings in SAS
errors = pd.read_csv(importlib.resources.files(__name__) / "files/sas_errors.csv", header="infer")
warnings = pd.read_csv(importlib.resources.files(__name__) / "files/sas_warnings.csv", header="infer")
# Just the names of the errors in two handy constants
SASERROR_LIST = errors["ErrName"].values
SASWARNING_LIST = warnings["WarnName"].values

# XSPEC file extraction (and base fit) scripts
XGA_EXTRACT = importlib.resources.files(__name__) / "xspec_scripts/xga_extract.tcl"
BASE_XSPEC_SCRIPT = importlib.resources.files(__name__) / "xspec_scripts/general_xspec_fit.xcm"
COUNTRATE_CONV_SCRIPT = importlib.resources.files(__name__) / "xspec_scripts/cr_conv_calc.xcm"
CROSS_ARF_XSPEC_SCRIPT = importlib.resources.files(__name__) / "xspec_scripts/crossarf_xspec_fit.xcm"

# Useful jsons of all XSPEC models, their required parameters, and those parameter's units
with open(importlib.resources.files(__name__) / "files/xspec_model_pars.json5", 'r') as filey:
    MODEL_PARS = json.load(filey)

with open(importlib.resources.files(__name__) / "files/xspec_model_units.json5", 'r') as filey:
    MODEL_UNITS = json.load(filey)

with open(importlib.resources.files(__name__) / "files/mission_event_column_name_map.json", 'r') as filey:
    MISSION_COL_DB = json.load(filey)
# --------------------------------------------------------------------------


# ------------- Defining constants to do with the telescope data -------------
# This chunk of this file sets

# This dictionary both defines the telescopes that XGA is compatible with, and their allowed instruments. These mission
#  and instrument names should all be lowercase, that will be the general storage convention throughout XGA
ALLOWED_INST = {"xmm": ["pn", "mos1", "mos2"],
                "erosita": ["tm1", "tm2", "tm3", "tm4", "tm5", "tm6", "tm7"],
                "erass": ["tm1", "tm2", "tm3", "tm4", "tm5", "tm6", "tm7"],
                "chandra": ["acis"]}

# I provide a list of the top-level keys of the ALLOWED_INST dictionary, as a quick way of accessing the supported
#  telescope names
TELESCOPES = list(ALLOWED_INST.keys())
# This dictionary won't be used much, but it's just so we have access to some properly formatted telescope names
PRETTY_TELESCOPE_NAMES = {'xmm': 'XMM', 'erosita': 'eROSITA', 'erass': 'eRASS', 'chandra': 'Chandra'}
# Here we define regular expressions that will allow use to verify the structure of an ObsID for a particular
#  telescope - this functionality is also in DAXA mission classes, so we may just switch to using them in the
#  future to avoid features duplication
OBS_ID_REGEX = {'xmm': '^[0-9]{10}$', "erosita": '^[0-9]{6}$', "erass": '^[0-9]{6}$', "chandra": '^[0-9]{1,5}'}

# This is another sort of duplication of a DAXA feature, and stores the default search distances to be used for
#  each telescope in the xga.match.separation_match function. These are loosely based on the field of view of
#  each telescope. In cases where different instruments on a telescope have significantly different field of view,
#  there may be multi-level dictionaries
# TODO when I chuck ROSAT in here add an entry like 'rosat': {'PSPCB': Quantity(60, 'arcmin'),
#  'PSPCC': Quantity(60, 'arcmin'), 'HRI': Quantity(19, 'arcmin'), 'RASS': Quantity(3, 'deg')}}
DEFAULT_TELE_SEARCH_DIST = {'xmm': Quantity(30, 'arcmin'), 'erosita': Quantity(60, 'arcmin'),
                            'erass': Quantity(108, 'arcmin'), 'chandra': Quantity(30, 'arcmin')}

# This list contains banned filter types - these occur in observations that I don't want XGA to try and use
BANNED_FILTS = {"xmm": ['CalClosed', 'Closed'],
                "erosita": ['CALIB', 'CLOSED'],
                "erass": ['CALIB', 'CLOSED'],
                "chandra": []}
# ----------------------------------------------------------------------------


# ------------- Defining constants to do with the configuration file -------------
# These will largely be the dictionaries that get turned into the various discrete sections of the configuration
#  file, with one for 'general configuration', and one for each separate telescope supported by XGA. Those dictionaries
#  will contain different entries, depending on the telescope, but the general idea is to point XGA at the available
#  events lists, images, and source regions

# Section of the config file for setting up the XGA module. The output path is where XGA generated files get stored
#  and the num_cores entry allows you to manually set the maximum number of cores that XGA can use - though by default
#  it is set to auto, which will use 90% of the total on your system
XGA_CONFIG = {"xga_save_path": "xga_output",
              "num_cores": 'auto'}

# Will have to make it clear in the documentation which paths can be left unspecified, and indeed that whole sections
#  can be left out if there are no relevant data archives for those telescopes. There will be a section for each
#  mission/telescope's configuration

# NOTE - THE 'root_{telescope name}_dir' IS REQUIRED - MAKE SURE TO ADD ONE FOR EVERY SUPPORTED TELESCOPE
# NOTE THE SECOND - THE PATHS TO CLEANED EVENTS ARE REQUIRED TO BE STRUCTURED
#  LIKE 'clean_{indicator of instrument}_evts' - it helps make the other bits of this setup process more generalised
# These are the pertinent bits of information for XMM - mainly the general 'where does data live' stuff
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

# The information required to use eROSITA data
EROSITA_FILES = {"root_erosita_dir": "/this/is/required/erosita_obs/data/",
                 "clean_erosita_evts": "/this/is/required/{obs_id}/{obs_id}.fits",
                 "lo_en": ['0.50', '2.00'],
                 "hi_en": ['2.00', '10.00'],
                 "erosita_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV_img.fits",
                 "erosita_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV_expmap.fits",
                 "region_file": "/this/is/optional/erosita_obs/regions/{obs_id}/regions.reg"}

# The information required to use eRASS data
ERASS_FILES = {"root_erass_dir": "/this/is/required/erass_obs/data/",
               "clean_erass_evts": "/this/is/required/{obs_id}/{obs_id}.fits",
               "lo_en": ['0.50', '2.00'],
               "hi_en": ['2.00', '10.00'],
               "erass_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV_img.fits",
               "erass_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV_expmap.fits",
               "region_file": "/this/is/optional/erass_obs/regions/{obs_id}/regions.reg"}

# The information required to use Chandra data
CHANDRA_FILES = {"root_chandra_dir": "/this/is/required/chandra_obs/data/",
                 "clean_acis_evts": "/this/is/required/{obs_id}/{obs_id}_ACIS_evts.fits",
                 "acis_badpix_file": "/this/is/required/{obs_id}/{obs_id}_badpix.fits",
                 "acis_mask_file": "/this/is/required/{obs_id}/{obs_id}_mask.fits",
                 "attitude_file": "/this/is/required/{obs_id}/{obs_id}_asol.fits",
                 "lo_en": ['0.5'],
                 "hi_en": ['7.0'],
                 "acis_image": "/this/is/optional/{obs_id}/{obs_id}-ACIS-{lo_en}-{hi_en}keV_img.fits",
                 "acis_expmap": "/this/is/optional/{obs_id}/{obs_id}-ACIS-{lo_en}-{hi_en}keV_expmap.fits",
                 "region_file": "/this/is/optional/chandra_obs/regions/{obs_id}/regions.reg"}

# We set up this dictionary for later, it makes programmatically grabbing the section dictionaries easier.
tele_conf_sects = {'xmm': XMM_FILES, 'erosita': EROSITA_FILES, 'erass': ERASS_FILES, 'chandra': CHANDRA_FILES}
# -------------------------------------------------------------------------------


# ------------- Defining constants to do with general XGA stuff -------------

# List of products supported by XGA that are allowed to be energy bound
ENERGY_BOUND_PRODUCTS = ["image", "expmap", "ratemap", "combined_image", "combined_expmap", "combined_ratemap"]
# These are the built-in profile types
PROFILE_PRODUCTS = ["brightness_profile", "gas_density_profile", "gas_mass_profile", "1d_apec_norm_profile",
                    "1d_proj_temperature_profile", "gas_temperature_profile", "baryon_fraction_profile",
                    "1d_proj_metallicity_profile", "1d_emission_measure_profile", "hydrostatic_mass_profile",
                    "specific_entropy_profile"]
COMBINED_PROFILE_PRODUCTS = ["combined_"+pt for pt in PROFILE_PRODUCTS]
# List of all products supported by XGA
ALLOWED_PRODUCTS = ["spectrum", "grp_spec", "regions", "events", "psf", "psfgrid", "ratemap", "combined_spectrum",
                    ] + ENERGY_BOUND_PRODUCTS + PROFILE_PRODUCTS + COMBINED_PROFILE_PRODUCTS

# A centralised constant to define what radius labels are allowed
RAD_LABELS = ["r2500", "r500", "r200", "custom", "point"]

# Adding a default concordance cosmology set up here - this replaces the original default choice of Planck15
DEFAULT_COSMO = LambdaCDM(70, 0.3, 0.7)

# The maximum difference in radii to consider them a match (used in get methods to avoid radii not matching)
RAD_MATCH_PRECISION = Quantity(1e-8, 'deg')

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
# ---------------------------------------------------------------------------


# ------------- Defining constants to do with physics -------------
# I know this is practically pointless, I could just use m_p
HY_MASS = m_p + m_e

# Mean molecular weight, mu
# TODO Make sure this doesn't change with abundance table, I think it might?
MEAN_MOL_WEIGHT = 0.61

# TODO Populate this further, also actually calculate and verify these myself, the value here is taken
#  from pyproffit code
# For a fully ionised plasma, this is the electron-to-proton ratio
NHC = {"angr": 1.199}
# -----------------------------------------------------------------


# ------------- Defining constants to do with units -------------
# Any new units we set up for the module will live in this section

# These are largely defined so that I can use them for when I'm normalising profile plots, that way
#  the view method can just write the units nicely the way it normally does
r200 = def_unit('r200', format={'latex': r"\mathrm{R_{200}}"})
r500 = def_unit('r500', format={'latex': r"\mathrm{R_{500}}"})
r2500 = def_unit('r2500', format={'latex': r"\mathrm{R_{2500}}"})

# These allow us to set up astropy quantities in units of some of the internal systems of telescopes - obviously
#  they don't convert to anything, but they still let us work within the astropy coordinate framework
xmm_sky = def_unit("xmm_sky")
xmm_det = def_unit("xmm_det")
#
erosita_sky = def_unit("erosita_sky")
erosita_det = def_unit("erosita_det")
#
erass_sky = def_unit("erass_sky")
erass_det = def_unit("erass_det")
#
chandra_sky = def_unit("chandra_sky")
chandra_det = def_unit("chandra_det")

# Define a dictionary to store the units in to make dynamic access easier. Also
#  allows us to test if we forgot to define a unit when adding a new mission.
MISSION_XY_UNITS = {'xmm': {'skyxy': xmm_sky, 'detxy': xmm_det},
                    'erosita': {'skyxy': erosita_sky, 'detxy': erosita_det},
                    'erass': {'skyxy': erass_sky, 'detxy': erass_det},
                    'chandra': {'skyxy': chandra_sky, 'detxy': chandra_det}
                    }

# Generic units for SKY and DET coordinate systems
SKY_XY_UNIT = def_unit("skyxy")
DET_XY_UNIT = def_unit("detxy")

XY_UNIT_EQUIVS = [(uno, SKY_XY_UNIT if unt == 'skyxy' else DET_XY_UNIT,
                   lambda x: x, lambda x: x)
                  for mn, uns in MISSION_XY_UNITS.items() for unt, uno in uns.items()]

# This is a dumb and annoying work-around for a readthedocs problem where units were being added multiple times
try:
    Quantity(1, 'r200')
except ValueError:
    # Adding the unit instances we created to the astropy pool of units - means we can do things like just defining
    #  Quantity(10000, 'xmm_det') rather than importing xmm_det from utils and using it that way
    add_enabled_units([r200, r500, r2500, xmm_sky, xmm_det, erosita_sky, erosita_det,
                       erass_sky, erass_det, chandra_sky, chandra_det,
                       SKY_XY_UNIT, DET_XY_UNIT])

    # We add the equivalencies for some of the custom units we defined
    add_enabled_equivalencies(XY_UNIT_EQUIVS)
# ---------------------------------------------------------------


