import unittest
import os
import shutil

from astropy.units import Quantity

# I know it is horrible to write code in the middle of importing modules, but this all needs to 
# happen before xga is imported, as we are moving config files

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

# Now when xga is imported it will make a new census with the test_data
import xga
from xga.sources import GalaxyCluster
from xga.generate.esass import evtool_image
from xga.products.phot import Image
from xga.products.spec import Spectrum
from xga.generate.esass import srctool_spectrum

# This will be run in the tearDownClass method, which will happen even if tests fail
def restore_og_cfg():
    """
    Restore the user's original xga config setup after tests are run.
    """
    if os.path.exists(xga_config_path):
        print('deleting test config dir')
        # Then we delete the test config file
        shutil.rmtree(xga_config_path)
        print('moving the original config file back')
        # And then move the original back
        shutil.move('./tests/xga', home_dir + '/.config')


class TestGalaxyCluster(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        This is run once before all tests. Here we define class objects that we want to test.
        """
        cls.test_src = GalaxyCluster(149.59209, -11.05972, 0.16, r500=Quantity(1200, 'kpc'), 
                                      r200=Quantity(1700, 'kpc'), name="A907", use_peak=False,
                                      search_distance={'erosita': Quantity(3.6, 'deg')})
    @classmethod
    def tearDownClass(cls):
        """
        This is run once after all the tests.
        """
        # This function restores the user's original config file and deletes the test one made
        restore_og_cfg()
        # Then we will delete all the products that xga has made so there aren't loads of big files
        # in the package
#        shutil.rmtree('tests/test_data/xga_output')

    def test_obs_ids_assigned(self):
        obs = self.test_src.obs_ids

        expected_xmm_obs = set(['0404910601', '0201901401', '0201903501'])
        expected_ero_obs = set(['147099', '148102', '151102', '150099'])
        xmm_obs = set(obs['xmm'])
        ero_obs = set(obs['erosita'])

        assert expected_ero_obs == ero_obs
        assert expected_xmm_obs == xmm_obs

    def test_existing_prods_loaded_in_img(self):
        """
        Testing _existing_xga_products() in BaseSource for images.
        """
        src = self.test_src
        evtool_image(src, Quantity(0.3, 'keV'), Quantity(3, 'keV'))

        del(src)

        src = GalaxyCluster(149.59209, -11.05972, 0.16, r500=Quantity(1200, 'kpc'), 
                                      r200=Quantity(1700, 'kpc'), name="A907", use_peak=False,
                                      search_distance={'erosita': Quantity(3.6, 'deg')})
        
        img = src.get_images(lo_en=Quantity(0.3, 'keV'), hi_en=Quantity(3, 'keV'))

        assert all([isinstance(im, Image) for im in img])

    def test_existing_prods_loaded_in_img_combined(self):
        """
        Testing _existing_xga_products() in BaseSource for combined images.
        """
        src = self.test_src
        evtool_image(src, Quantity(0.3, 'keV'), Quantity(3, 'keV'), combine_obs=True)

        del(src)

        src = GalaxyCluster(149.59209, -11.05972, 0.16, r500=Quantity(1200, 'kpc'), 
                                      r200=Quantity(1700, 'kpc'), name="A907", use_peak=False,
                                      search_distance={'erosita': Quantity(3.6, 'deg')})
        
        img = src.get_combined_images(lo_en=Quantity(0.3, 'keV'), hi_en=Quantity(3, 'keV'))

        assert isinstance(img, Image)
        assert img.obs_id == 'combined'

    def test_existing_prods_loaded_in_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra.
        """
        src = self.test_src
        srctool_spectrum(src, 'r500', combine_obs=False)
        
        del(src)

        src = GalaxyCluster(149.59209, -11.05972, 0.16, r500=Quantity(1200, 'kpc'), 
                                      r200=Quantity(1700, 'kpc'), name="A907", use_peak=False,
                                      search_distance={'erosita': Quantity(3.6, 'deg')})
        
        spec = src.get_spectra('r500', telescope='erosita')

        assert all(isinstance(sp, Spectrum) for sp in spec)

    def test_existing_prods_loaded_in_comb_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra made from combined obs.
        """
        src = self.test_src
        srctool_spectrum(src, 'r500', combine_obs=True)
        
        del(src)

        src = GalaxyCluster(149.59209, -11.05972, 0.16, r500=Quantity(1200, 'kpc'), 
                                      r200=Quantity(1700, 'kpc'), name="A907", use_peak=False,
                                      search_distance={'erosita': Quantity(3.6, 'deg')})
        
        spec = src.get_combined_spectra('r500', inst='combined', telescope='erosita')

        assert isinstance(spec, Spectrum)

    def test_existing_prods_loaded_in_idv_inst_comb_obs_spectra(self):
        """
        Testing _existing_xga_products() in BaseSource for spectra made from combined obs.
        """
        src = self.test_src
        srctool_spectrum(src, 'r500', combine_obs=True, combine_tm=False)
        
        del(src)

        src = GalaxyCluster(149.59209, -11.05972, 0.16, r500=Quantity(1200, 'kpc'), 
                                      r200=Quantity(1700, 'kpc'), name="A907", use_peak=False,
                                      search_distance={'erosita': Quantity(3.6, 'deg')})
        
        spec = src.get_combined_spectra('r500', telescope='erosita', inst='tm1')

        assert isinstance(spec, Spectrum)
        assert spec.instrument == 'tm1'

# TODO combined_spectrum should get things with instruments and obsids

if __name__ == "__main__":
     unittest.main()