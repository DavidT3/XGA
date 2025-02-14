#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from daxa.mission import XMMPointed, eRASS1DE
from daxa.archive import Archive
from daxa.process.simple import full_process_xmm, full_process_erosita

# Tests can be run on two modes which is controlled here
# TEST_MODE = DEV, this assumes that the data to test on is already downloaded and in the right 
# place, it will also not delete the xga_output folder in /tests/test_data
# TEST_MODE = RUN, this assumes that no data has been downloaded and will delete the xga_output
# folder after running the tests, and will delete all the data that has been downloaded
TEST_MODE = 'DEV'

# This is the main source for running the tests on
SRC_INFO = {'ra': 226.0318, 'dec': -2.8046, 'z': 0.2093, 'name': "1eRASS_J150407.6-024816"}