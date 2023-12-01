#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 01/12/2023, 09:55. Copyright (c) The Contributors

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

# TODO THIS ONE WAS FIT BY ME, AND IS PRELIMINARY

xcs_sdss_r500_52_TL = ScalingRelation(np.array([0.30226566, 1.08843811]), np.array([0.01372833, 0.02184595]), power_law,
                                      Quantity(0.8e+44, 'erg / s'), Quantity(4, 'keV'),
                                      r"L$_{\rm{x},500,0.5-2.0}$", r"E(z)T$_{\rm{x},500}$",
                                      relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ $R_{500}$ 0.5-2.0keV',
                                      dim_hubb_ind=1, x_en_bounds=Quantity([0.5, 2.0], 'keV'),
                                      x_lims=Quantity([2e+42, 2e+45], 'erg/s'))

xcs_sdss_r2500_52_TL = ScalingRelation(np.array([0.27682937, 1.24930134]), np.array([0.01307604, 0.02552076]),
                                       power_law, Quantity(0.8e+44, 'erg / s'), Quantity(4, 'keV'),
                                       r"L$_{\rm{x},2500,0.5-2.0}$", r"E(z)T$_{\rm{x},2500}$",
                                       relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ $R_{2500}$ 0.5-2.0keV',
                                       dim_hubb_ind=1, x_en_bounds=Quantity([0.5, 2.0], 'keV'),
                                       x_lims=Quantity([1e+42, 1e+45], 'erg/s'))
