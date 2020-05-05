#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 03/05/2020, 12:14. Copyright (c) David J Turner

import os
from configparser import ConfigParser
from subprocess import Popen, PIPE
from astropy.io import fits
from tqdm import tqdm
import sys
from pandas import DataFrame
from numpy import nan

from xga.exceptions import XGAConfigError, HeasoftError, SASNotFoundError

# Got to make sure we're able to import the PyXspec module.
# Currently raises an error, but perhaps later on I'll relax this to a warning.
try:
    import xspec
except ModuleNotFoundError:
    raise HeasoftError("Unable to import PyXspec, you have to make sure to set a PYTHON environment "
                       "variable before installing HEASOFT/XSPEC.")

# This one I'm less likely to relax to a warnings
if "SAS_DIR" not in os.environ:
    raise SASNotFoundError("SAS_DIR environment variable is not set, unable to verify SAS is present on system")
else:
    # This way, the user can just import the SAS_VERSION from this utils code
    out, err = Popen("sas --version", stdout=PIPE, stderr=PIPE, shell=True).communicate()
    SAS_VERSION = out.decode("UTF-8").strip("]\n").split('-')[-1]

# If XDG_CONFIG_HOME is set, then use that, otherwise use this default config path
CONFIG_PATH = os.environ.get('XDG_CONFIG_HOME', os.path.join(os.path.expanduser('~'), '.config', 'xga'))
# The path to the census file, which documents all available ObsIDs and their pointings
CENSUS_FILE = os.path.join(CONFIG_PATH, 'census.csv')
# XGA config file path
CONFIG_FILE = os.path.join(CONFIG_PATH, 'xga.cfg')
# Section of the config file for setting up the XGA module
XGA_CONFIG = {"xga_save_path": "/this/is/required/xga_output/"}
# Will have to make it clear in the documentation what is allowed here, and which can be left out
# TODO Figure out how on earth to deal with separate exp1 and exp2 etc events lists/images.
#  For now just ignore them I guess?
XMM_FILES = {"root_xmm_dir": "/this/is/required/xmm_obs/data/",
             "clean_pn_evts": "/this/is/required/{obs_id}/pn_exp1_clean_evts.fits",
             "clean_mos1_evts": "/this/is/required/{obs_id}/mos1_exp1_clean_evts.fits",
             "clean_mos2_evts": "/this/is/required/{obs_id}/mos2_exp1_clean_evts.fits",
             "ccf_index_file": "/this/is/optional/{obs_id}/odf/ccf.cif",
             "lo_en": ['0.50', '2.00'],
             "hi_en": ['2.00', '10.00'],
             "pn_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
             "mos1_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_img.fits",
             "mos2_image": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_img.fits",
             "pn_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits",
             "mos1_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_expmap.fits",
             "mos2_expmap": "/this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_expmap.fits"}


def xmm_obs_id_test(test_string: str) -> bool:
    """
    Crude function to try and determine if a string follows the pattern of an XMM ObsID
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


def observation_census(config: ConfigParser) -> DataFrame:
    """
    A function to initialise or update the file that stores which observations are available in the user
    specified XMM data directory, and what their pointing coordinates are.
    CURRENTLY THIS WILL NOT UPDATE TO DEAL WITH OBSID FOLDERS THAT HAVE BEEN DELETED.
    :param config: The XGA configuration object.
    :return: ObsIDs and pointing coordinates of available XMM observations.
    :rtype: DataFrame
    """
    # The census lives in the XGA config folder, and CENSUS_FILE stores the path to it.
    # If it exists, it is read in, otherwise empty lists are initialised to be appended to.
    if os.path.exists(CENSUS_FILE):
        with open(CENSUS_FILE, 'r') as census:
            obs_lookup = census.readlines()  # Reads the lines of the files
            # This is just ObsIDs, needed to see which ObsIDs have already been processed.
            obs_lookup_obs = [entry.split(',')[0] for entry in obs_lookup[1:]]
    else:
        obs_lookup = ["ObsID,RA_PNT,DEC_PNT\n"]
        obs_lookup_obs = []

    # Need to find out which observations are available, crude way of making sure they are ObsID directories
    # This also checks that I haven't run them before
    obs_census = [entry for entry in os.listdir(config["XMM_FILES"]["root_xmm_dir"]) if xmm_obs_id_test(entry)
                  and entry not in obs_lookup_obs]
    if len(obs_census) != 0:
        census_progress = tqdm(desc="Assembling list of ObsID pointings", total=len(obs_census))
        for obs in obs_census:
            ra_pnt = ''
            dec_pnt = ''
            # Prepared to check all three events files, but if one succeeds the rest are skipped for efficiency
            for key in ["clean_pn_evts", "clean_mos1_evts", "clean_mos2_evts"]:
                evt_path = config["XMM_FILES"][key].format(obs_id=obs)
                if os.path.exists(evt_path) and ra_pnt == '' and dec_pnt == '':
                    with fits.open(evt_path, mode='readonly') as evts:
                        try:
                            ra_pnt = evts[0].header["RA_PNT"]
                            dec_pnt = evts[0].header["DEC_PNT"]
                        except KeyError:
                            pass
                    break
                    # If this part has run successfully there's no need to open the other images
            # Format to write to the census.csv that lives in the config directory.
            obs_lookup.append("{o},{r},{d}\n".format(o=obs, r=ra_pnt, d=dec_pnt))
            census_progress.update(1)
        census_progress.close()
        with open(CENSUS_FILE, 'w') as census:
            census.writelines(obs_lookup)

    # I do the stripping and splitting to make it a 3 column array, needed to be lines to write to file
    obs_lookup = DataFrame(data=[entry.strip('\n').split(',') for entry in obs_lookup[1:]],
                           columns=obs_lookup[0].strip("\n").split(','), dtype=str)
    obs_lookup["RA_PNT"] = obs_lookup["RA_PNT"].replace('', nan).astype(float)
    obs_lookup["DEC_PNT"] = obs_lookup["DEC_PNT"].replace('', nan).astype(float)
    return obs_lookup


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
    keys_to_check = ["root_xmm_dir", "clean_pn_evts", "clean_mos1_evts", "clean_mos2_evts"]
    # Here I check that the installer has actually changed the three events file paths
    all_changed = all([xga_conf["XMM_FILES"][key] != XMM_FILES[key] for key in keys_to_check])
    if not all_changed:
        raise XGAConfigError("Some events file paths (or the root_xmm_dir) in the config have not "
                             "been changed from default")
    elif not os.path.exists(xga_conf["XMM_FILES"]["root_xmm_dir"]):
        raise FileNotFoundError("That root_xmm_dir does not appear to exist, "
                                "if it an SFTP mount check the connection.")

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

    # Make sure that this is the absolute path
    xga_conf["XMM_FILES"]["root_xmm_dir"] = os.path.abspath(xga_conf["XMM_FILES"]["root_xmm_dir"]) + "/"
    # Read dataframe of ObsIDs and pointing coordinates into constant
    CENSUS = observation_census(xga_conf)



