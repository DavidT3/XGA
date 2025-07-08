#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors
from astropy.units import Quantity
import numpy as np
import pandas as pd

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
expected_xmm_obs = set(['0401040101', '0840580201', '0840580101'])
expected_ero_obs = set(['227093', '227090', '224093'])

# This is another cluster that we can use to test ClusterSample objects
SUPP_SRC_INFO = {'ra': 55.7164, 'dec': -53.6292, 'z': 0.0587, 'name': 'A3158'}

# Making a df to make a sample from
column_names = ['name', 'ra', 'dec', 'z', 'r500']
cluster_data = np.array([[SRC_INFO['name'], SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], 500],
                         [SUPP_SRC_INFO['name'], SUPP_SRC_INFO['ra'], SUPP_SRC_INFO['dec'], SUPP_SRC_INFO['z'], 500]])

CLUSTER_SMP = pd.DataFrame(data=cluster_data, columns=column_names)
CLUSTER_SMP[['ra', 'dec', 'z', 'r500']] = CLUSTER_SMP[['ra', 'dec', 'z', 'r500']].astype(float)

SRC_ALL_TELS = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
    
SRC_XMM = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     telescope='xmm')
SRC_ERO = GalaxyCluster(SRC_INFO['ra'], SRC_INFO['dec'], SRC_INFO['z'], r500=Quantity(500, 'kpc'),
                                     name=SRC_INFO['name'], use_peak=False,
                                     telescope='erosita',
                                     search_distance={'erosita': Quantity(3.6, 'deg')})
