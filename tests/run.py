import os 
import unittest
import shutil 
from subprocess import Popen, PIPE

from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from daxa.mission import XMMPointed, eRASS1DE
from daxa.archive import Archive
from daxa.process.simple import full_process_xmm, full_process_erosita

from . import TEST_MODE, SRC_INFO, SUPP_SRC_INFO

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
        root_xmm_dir = {cwd}/tests/test_data/daxa_out/archives/xga_tests/processed_data/xmm_pointed/
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
        root_erosita_dir = {cwd}/tests/test_data/daxa_out/archives/xga_tests/processed_data/erosita_all_sky_de_dr1/
        clean_erosita_evts = {{obs_id}}/events/obsid{{obs_id}}-instTM1_TM2_TM3_TM4_TM5_TM6_TM7-subexpALL-en0.2_10.0keV-finalevents.fits
        lo_en = ['0.5']
        hi_en = ['2.0']
        erosita_image = /this/is/optional/
        erosita_expmap = /this/is/optional/
        region_file = {cwd}/tests/test_data/src_regs/erosita/{{obs_id}}.reg
        """
    elif module == 'daxa':
        config_content = f"""[DAXA_SETUP]
        daxa_save_path = {cwd}/tests/test_data/daxa_out
        num_cores = 1
        """
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
    config_path = home_dir + f"/.config/{module}"

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

def obtain_test_data():
    """
    Uses DAXA to download the test data for the source. This processes the data too.
    """
    # need to import here so it retrieves the new test config, instead of the user's config
    import daxa
    from daxa import daxa_conf

    print(daxa_conf)

    xm = XMMPointed()
    er = eRASS1DE()

    xm.filter_on_positions([[SRC_INFO['ra'], SRC_INFO['dec']], [SUPP_SRC_INFO['ra'], 
                             SUPP_SRC_INFO['dec']]])
    er.filter_on_positions([[SRC_INFO['ra'], SRC_INFO['dec']], [SUPP_SRC_INFO['ra'], 
                             SUPP_SRC_INFO['dec']]], search_distance=Quantity(3.6, 'deg'))

    arch = Archive('xga_tests', [xm, er])

    full_process_erosita(arch)
    full_process_xmm(arch)


def set_up_tests():
    """
    To be run before any tests are run. This moves original config files, creates test config files. 
    It then also downloads data if TEST_MODE is set to RUN.
    """
    cwd = os.getcwd()

    write_config(cwd, 'xga')
    move_og_cfg('xga')

    if TEST_MODE == 'RUN':
        write_config(cwd, 'daxa')
        move_og_cfg('daxa')
        obtain_test_data()

def restore_og_cfg(module):
    """
    Restore the user's original xga config setup after tests are run.
    """
    # For some reason error where happening when I used the path '~/.config' so I need to use expanduser
    # to return the absolute path to the home directory
    home_dir = os.path.expanduser("~")
    # xga_config_path is the absolute path to ~/.config/xga
    config_path = home_dir + f"/.config/{module}"

    if os.path.exists(config_path):
        print('deleting test config dir')
        # Then we delete the test config file
        shutil.rmtree(config_path)
        print('moving the original config file back')
        # And then move the original back
        shutil.move(f'./tests/{module}', home_dir + '/.config')

def clean_up_test_files():
    """
    This is run after all the tests to remove the extra files that have been created during the 
    tests, only if TEST_MODE == RUN.
    """
    if os.path.exists("./tests/xga_output"):
        shutil.rmtree("./tests/xga_output")
    if os.path.exists("./tests/test_data/daxa_out"):
        shutil.rmtree("./tests/test_data/daxa_out")

#  TODO work out how to run this after all the tests have been run in different files
def clean_up_tests():
    """
    Run after all tests have been run. This restores any original configuration files and if
    TEST_MODE == RUN, it will delete any extra files that have been created, and delete downloaded
    data.
    """
    restore_og_cfg('xga')

    if TEST_MODE == 'RUN':
        restore_og_cfg('daxa')
        clean_up_test_files()


if __name__ == "__main__":
    set_up_tests()  # Run before any tests

    if TEST_MODE == 'COV':
        cmd = 'coverage run -m unittest discover'
        out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
        out = out.decode("UTF-8", errors='ignore')
        err = err.decode("UTF-8", errors='ignore')
        with open("coverage.txt", "w") as text_file:
            text_file.write(out + err)
    
    else:
        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.discover(start_dir="tests")

        runner = unittest.TextTestRunner()
        result = runner.run(suite)

    clean_up_tests()  # Run after all tests