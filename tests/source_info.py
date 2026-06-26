#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 4/27/26, 12:01 PM. Copyright (c) The Contributors.

# This file contains information about the sources used in XGA's unit tests.
# It is kept separate from the rest of the testing infrastructure to avoid
# unnecessary imports (and potential circular dependencies) when only the
# source coordinates/info are needed.

# This is the main source for running the tests on
SRC_INFO = {'ra': 226.0318, 'dec': -2.8046, 'z': 0.2093, 'name': "1eRASS_J150407.6-024816"}
EXPECTED_XMM_OBS = {'0401040101', '0840580201', '0840580101'}
EXPECTED_ERO_OBS = {'227093', '227090', '224093'}

# This is another cluster that we can use to test ClusterSample objects
SUPP_SRC_INFO = {'ra': 55.7164, 'dec': -53.6292, 'z': 0.0587, 'name': 'A3158'}
