#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 20/02/2023, 14:04. Copyright (c) The Contributors

import os
import sys

from astropy.units import Quantity

sys.path.append(os.path.abspath("..") + 'xga/')


# Any useful constants
A907_LOC = Quantity([149.59209, -11.05972], 'deg')
im_path = os.path.join(os.path.abspath("."), "test_data/0201903501/images/0201903501_pn_exp1-0.50-2.00keVimg.fits")
exp_path = os.path.join(os.path.abspath("."), "test_data/0201903501/images/0201903501_pn_exp1-0.50-2.00keVexpmap.fits")
A907_IM_PN_INFO = [im_path, '0201903501', 'pn', '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV')]
A907_EX_PN_INFO = [exp_path, '0201903501', 'pn', '', '', '', Quantity(0.5, 'keV'), Quantity(2.0, 'keV')]

