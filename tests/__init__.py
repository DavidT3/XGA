#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors
from astropy.units import Quantity

from xga.sources import GalaxyCluster

# Tests can be run on two modes which is controlled here
# TEST_MODE = DEV, this assumes that the data to test on is already downloaded and in the right 
# place, it will also not delete the xga_output folder in /tests/test_data
# TEST_MODE = RUN, this assumes that no data has been downloaded and will delete the xga_output
# folder after running the tests, and will delete all the data that has been downloaded
# TEST_MODE = COV, this is identical to the assumption of TEST_MODE = DEV, but it also calculates
# the coverage of the tests and appends the output to a coverage.txt file
TEST_MODE = 'COV'

# This is the main source for running the tests on
SRC_INFO = {'ra': 226.0318, 'dec': -2.8046, 'z': 0.2093, 'name': "1eRASS_J150407.6-024816"}

# This is another cluster that we can use to test ClusterSample objects
SUPP_SRC_INFO = {'ra': 226.03181, 'dec': -2.80458, 'z': 0.209, 'name': '1eRASS J150407.6-024816'}

SRC_ALL_TELS = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})

