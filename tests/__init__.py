#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

import os
import shutil
import sys

from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from daxa.mission import XMMPointed, eRASS1DE
from daxa.archive import Archive

#sys.path.append(os.path.abspath("..") + 'xga/')

# Any useful constants
A907_LOC = Quantity([149.59209, -11.05972], 'deg')
im_path = os.path.join(os.path.abspath("."), "test_data/0201903501/images/0201903501_pn_exp1-0.50-2.00keVimg.fits")
exp_path = os.path.join(os.path.abspath("."), "test_data/0201903501/images/0201903501_pn_exp1-0.50-2.00keVexpmap.fits")
A907_IM_PN_INFO = [im_path, '0201903501', 'pn', '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV')]
A907_EX_PN_INFO = [exp_path, '0201903501', 'pn', '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV')]

def set_up_test_config():
    def write_xga_config(cwd):
        """
        Writes an xga config file with paths specifically for the unit tests.

        :param str cwd: The absolute path of the current working directory - which when running tests
            should be the level above the tests directory.
        """
        # Creating the configuration content
        config_content = f"""[XGA_SETUP]
        xga_save_path = {cwd}/tests/test_data/xga_output
        num_cores = 1

        [XMM_FILES]
        root_xmm_dir = {cwd}/tests/test_data/xmm/
        clean_pn_evts = {{obs_id}}/eclean/pn_exp1_clean_evts.fits
        clean_mos1_evts = {{obs_id}}/eclean/mos1_exp1_clean_evts.fits
        clean_mos2_evts = {{obs_id}}/eclean/mos2_exp1_clean_evts.fits
        attitude_file = {{obs_id}}/epchain/P{{obs_id}}OBX000ATTTSR0000.FIT
        lo_en = ['0.50', '2.00']
        hi_en = ['2.00', '10.00']
        pn_image = {{obs_id}}/images/{{obs_id}}_pn_exp1-{{lo_en}}-{{hi_en}}keVimg.fits
        mos1_image = {{obs_id}}/images/{{obs_id}}_mos1_exp1-{{lo_en}}-{{hi_en}}keVimg.fits
        mos2_image = {{obs_id}}/images/{{obs_id}}_mos2_exp1-{{lo_en}}-{{hi_en}}keVimg.fits
        pn_expmap = {{obs_id}}/images/{{obs_id}}_pn_exp1-{{lo_en}}-{{hi_en}}keVexpmap.fits
        mos1_expmap = {{obs_id}}/images/{{obs_id}}_mos1_exp1-{{lo_en}}-{{hi_en}}keVexpmap.fits
        mos2_expmap = {{obs_id}}/images/{{obs_id}}_mos2_exp1-{{lo_en}}-{{hi_en}}keVexpmap.fits
        region_file = {{obs_id}}_sources.reg

        [EROSITA_FILES]
        root_erosita_dir = {cwd}/tests/test_data/erosita/
        clean_erosita_evts = {{obs_id}}/{{obs_id}}-clean_evts.fits
        lo_en = ['0.5']
        hi_en = ['2.0']
        erosita_image = {{obs_id}}/{{obs_id}}_bin_87_{{lo_en}}-{{hi_en}}keVimg.fits
        erosita_expmap = {{obs_id}}/{{obs_id}}_combined_{{lo_en}}-{{hi_en}}keVexpmap.fits
        region_file = {{obs_id}}/{{obs_id}}.reg
        """

        # Writing the content to the specified file
        with open('tests/test_data/xga.cfg', 'w') as file:
            file.write(config_content)

    # Actually writing the test config file
    cwd = os.getcwd()
    write_xga_config(cwd)

    # For some reason error where happening when I used the path '~/.config' so I need to use expanduser
    # to return the absolute path to the home directory
    home_dir = os.path.expanduser("~")
    # xga_config_path is the absolute path to ~/.config/xga
    xga_config_path = home_dir + "/.config/xga"

    # Now we need to move the original files that where in ~/.config/xga elsewhere, because the testing
    # files will have to go into ~/.config/xga
    if os.path.exists(xga_config_path):
        # this will move the original file to the tests directory
        print('moving original /xga to tests dir')
        shutil.move(xga_config_path, './tests/')

    print('remaking the xga dir')
    # Then we want to remake the config/xga directory
    os.makedirs(xga_config_path)
    print('moving the test config file to the .config/xga dir')
    # Then move the test config file there
    shutil.move('./tests/test_data/xga.cfg', xga_config_path)

def obtain_test_data(src_ra, src_dec):
    # The user might have daxa installed already, so we will temporarily make a new cfg file if ythe
    cwd = os.getcwd()
    write_daxa_config(cwd)

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
    er.filter_on_positions(position, search_distance=Quantity(3.6, 'deg'))
    arch = Archive('xga_tests', [er, xm])


# This will be run in the tearDownClass method, which will happen even if tests fail
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