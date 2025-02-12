#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

import os
import shutil
import sys

from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from daxa.mission import XMMPointed, eRASS1DE
from daxa.archive import Archive
from daxa.process.simple import full_process_xmm, full_process_erosita

# Tests can be run on two modes which is controlled here
# TEST_MODE = DEV, this assumes that the data to test on is already downloaded and in the right 
# place, it will also not delete the xga_output folder in /tests/test_data
# TEST_MODE = RUN, this assumes that no data has been downloaded and will delete the xga_output
# folder after running the tests, and will delete all the data that has been downloaded

TEST_MODE = 'DEV'

#sys.path.append(os.path.abspath("..") + 'xga/')

# Any useful constants
A907_LOC = Quantity([149.59209, -11.05972], 'deg')
im_path = os.path.join(os.path.abspath("."), "test_data/0201903501/images/0201903501_pn_exp1-0.50-2.00keVimg.fits")
exp_path = os.path.join(os.path.abspath("."), "test_data/0201903501/images/0201903501_pn_exp1-0.50-2.00keVexpmap.fits")
A907_IM_PN_INFO = [im_path, '0201903501', 'pn', '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV')]
A907_EX_PN_INFO = [exp_path, '0201903501', 'pn', '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV')]

def write_config(cwd, module):
    """
    Writes a config file written for the unit tests for daxa or xga.

    :param str cwd: The absolute path of the current working directory - which when running tests
        should be the level above the tests directory.
    :param str module: The module to write the config for - either 'daxa' or 'xga'
    """

    if module == 'xga':
        config_content = f"""[XGA_SETUP]
        xga_save_path = {cwd}/tests/test_data/xga_output
        num_cores = 1

        [XMM_FILES]
        root_xmm_dir = {cwd}/tests/test_data/xga_tests/archives/processed_data/xmm_pointed/
        clean_pn_evts = {{obs_id}}/events/obsid{{obs_id}}-instPN-subexpALL-en-finalevents.fits
        clean_mos1_evts = {{obs_id}}/obsid{{obs_id}}-instM1-subexpALL-en-finalevents.fits
        clean_mos2_evts = {{obs_id}}/obsid{{obs_id}}-instM2-subexpALL-en-finalevents.fits
        attitude_file = {{obs_id}}/P{{obs_id}}OBX000ATTTSR0000.FIT
        lo_en = ['0.50', '2.00']
        hi_en = ['2.00', '10.00']
        pn_image = {{obs_id}}/images/obsid{{obs_id}}-instPN-subexpALL-en{{lo_en}}_{{hi_en}}keV-image.fits
        mos1_image = {{obs_id}}/images/obsid{{obs_id}}-instM1-subexpALL-en{{lo_en}}_{{hi_en}}keV-image.fits
        mos2_image = {{obs_id}}/images/obsid{{obs_id}}-instM2-subexpALL-en{{lo_en}}_{{hi_en}}keV-image.fits
        pn_expmap = {{obs_id}}/images/obsid{{obs_id}}-instPN-subexpALL-en{{lo_en}}_{{hi_en}}keV-expmap.fits
        mos1_expmap = {{obs_id}}/images/obsid{{obs_id}}-instM1-subexpALL-en{{lo_en}}_{{hi_en}}keV-expmap.fits
        mos2_expmap = {{obs_id}}/images/obsid{{obs_id}}-instM2-subexpALL-en{{lo_en}}_{{hi_en}}keV-expmap.fits
        region_file = {cwd}/tests/test_data/src_regs/xmm/{{obs_id}}.reg

        [EROSITA_FILES]
        root_erosita_dir = {cwd}/tests/test_data/xga_tests/archives/processed_data/erosita_all_sky_de_dr1/
        clean_erosita_evts = {{obs_id}}/events/obsid{{obs_id}}-instTM1_TM2_TM3_TM4_TM5_TM6_TM7-subexpALL-en0.2_10.0keV-finalevents.fits
        lo_en = ['0.5']
        hi_en = ['2.0']
        erosita_image = /this/is/optional/
        erosita_expmap = /this/is/optional/
        region_file = {cwd}/tests/test_data/src_regs/erosita/{{obs_id}}.reg
        """
    elif module == 'daxa':
        config_content = f"""[DAXA_SETUP]
        daxa_save_path = {cwd}/tests/test_data/
        num_cores = 1"""
    else:
        raise ValueError('Something has gone wrong')

    # Writing the content to the specified file
    with open(f'tests/test_data/{module}.cfg', 'w') as file:
        file.write(config_content)    

def move_og_cfg(module):
    """
    If the user already has XGA and DAXA installed, this function will move them to 
    /tests/og_configs so that new configs for unit tests can be written there.

    :param str module:
        This is either xga or daxa.
    """
    # For some reason an error is happening when I used the path '~/.config' so I need to use 
    # expanduser to return the absolute path to the home directory
    home_dir = os.path.expanduser("~")
    # config_path is the absolute path to ~/.config/{module}
    config_path = home_dir + "/.config/{module}"

    # Now we need to move the original files that where in ~/.config/{module} elsewhere, because the 
    # testing files will have to go into ~/.config/{module}
    if os.path.exists(config_path):
        # this will move the original file to /tests/og_configs
        print(f'moving original {module} to /tests/og_configs')
        shutil.move(config_path, './tests/og_configs')

    print(f'remaking the {module} dir')
    # Then we want to remake the config/xga directory
    os.makedirs(config_path)
    print(f'moving the test config file to the .config/{module} dir')
    # Then move the test config file there
    shutil.move(f'./tests/test_data/{module}.cfg', config_path)

def obtain_test_data(src_ra, src_dec):
    # For some reason error where happening when I used the path '~/.config' so I need to use expanduser
    # to return the absolute path to the home directory
    home_dir = os.path.expanduser("~")
    # xga_config_path is the absolute path to ~/.config/daxa
    daxa_config_path = home_dir + "/.config/daxa"

    # Now we need to move the original files that where in ~/.config/daxa elsewhere, because the testing
    # files will have to go into ~/.config/daxa
    if os.path.exists(daxa_config_path):
        # this will move the original file to the tests directory
        print('moving original /daxa to tests dir')
        shutil.move(daxaconfig_path, './tests/')

    print('remaking the xga dir')
    # Then we want to remake the config/xga directory
    os.makedirs(daxa_config_path)
    print('moving the test config file to the .config/daxa dir')
    # Then move the test config file there
    shutil.move('./tests/test_data/daxa.cfg', daxa_config_path)

    xm = XMMPointed()
    er = eRASS1DE()
    position = SkyCoord(src_ra, src_dec, unit='deg')
    xm.filter_on_positions(position)

# TODO work out how to run this before all the tests in different files
# This should run before all of the tests are run
def set_up_tests(TEST_MODE):
    cwd = os.getcwd()

    move_og_cfg('xga')
    write_config(cwd, 'xga')

    if TEST_MODE == 'RUN':
        move_og_cfg('daxa')
        write_config(cwd, 'daxa')
        obtain_test_data()

# TODO work out how to run this after all the tests have been run in different files
def restore_og_cfg():
    """
    Restore the user's original xga config setup after tests are run.
    """
    # For some reason error where happening when I used the path '~/.config' so I need to use expanduser
    # to return the absolute path to the home directory
    home_dir = os.path.expanduser("~")
    # xga_config_path is the absolute path to ~/.config/xga
    xga_config_path = home_dir + "/.config/xga"

    if os.path.exists(xga_config_path):
        print('deleting test config dir')
        # Then we delete the test config file
        shutil.rmtree(xga_config_path)
        print('moving the original config file back')
        # And then move the original back
        shutil.move('./tests/xga', home_dir + '/.config')