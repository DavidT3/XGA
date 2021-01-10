#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 10/01/2021, 16:51. Copyright (c) David J Turner

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

xcs_sdss_r500_52 = ScalingRelation(np.array([1.67, 0.96]), np.array([0.13, 0.08]), power_law, Quantity(60),
                                   Quantity(0.8e+44, 'erg / s'), r"$\lambda$", r"E(z)$^{-1}$L$_{\rm{x},500,0.5-2.0}$",
                                   relation_author='Giles et al.', relation_year='In Prep', relation_doi='',
                                   relation_name='SDSSRM-XCS$_{T_{x},vol}$ 0.5-2.0keV',
                                   x_lims=Quantity([20, 220]))








