#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 10/01/2021, 16:51. Copyright (c) David J Turner

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

# TODO All the XCS-SDSS relations also have an intrinsic scatter parameter, but that isn't
#  implemented yet(see issue #289)


xcs_sdss_r500_52 = ScalingRelation(np.array([2.51, 0.97]), np.array([0.11, 0.06]), power_law, Quantity(4, 'keV'),
                                   Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500}$",
                                   r"E(z)$^{-1}$L$_{\rm{x},500,0.5-2.0}$", relation_author='Giles et al.',
                                   relation_year='In Prep', relation_doi='',
                                   relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ 0.5-2.0keV',
                                   x_lims=Quantity([1, 12], 'keV'))

xcs_sdss_r500_bol = ScalingRelation(np.array([2.94, 3.06]), np.array([0.12, 0.18]), power_law, Quantity(4, 'keV'),
                                    Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500}$",
                                    r"E(z)$^{-1}$L$_{\rm{x},500,bol}$",
                                    relation_author='Giles et al.', relation_year='In Prep', relation_doi='',
                                    relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ Bolometric',
                                    x_lims=Quantity([1, 12], 'keV'))



