#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 15/09/2025, 16:11. Copyright (c) The Contributors

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

# These are from the classic M-T relation published by Arnaud
arnaud_m200 = ScalingRelation(np.array([1.72, 5.34]), np.array([0.10, 0.22]), power_law, Quantity(5, 'keV'),
                              Quantity(1e+14, 'solMass'), r"T$_{\rm{x}}$", "E(z)M$_{200}$",
                              x_lims=Quantity([1, 12], 'keV'), relation_name='Hydrostatic Mass-Temperature',
                              relation_author='Arnaud et al.', relation_year='2005',
                              relation_doi='10.1051/0004-6361:20052856', dim_hubb_ind=1,
                              telescope='xmm', outer_aperture='R200')

arnaud_m500 = ScalingRelation(np.array([1.71, 3.84]), np.array([0.09, 0.14]), power_law, Quantity(5, 'keV'),
                              Quantity(1e+14, 'solMass'), r"T$_{\rm{x}}$", "E(z)M$_{500}$",
                              x_lims=Quantity([1, 12], 'keV'), relation_name='Hydrostatic Mass-Temperature',
                              relation_author='Arnaud et al.', relation_year='2005',
                              relation_doi='10.1051/0004-6361:20052856', dim_hubb_ind=1, telescope='xmm',
                              outer_aperture='R500')

arnaud_m2500 = ScalingRelation(np.array([1.70, 1.69]), np.array([0.07, 0.05]), power_law, Quantity(5, 'keV'),
                               Quantity(1e+14, 'solMass'), r"T$_{\rm{x}}$", "E(z)M$_{2500}$",
                               x_lims=Quantity([1, 12], 'keV'), relation_name='Hydrostatic Mass-Temperature',
                               relation_author='Arnaud et al.', relation_year='2005',
                               relation_doi='10.1051/0004-6361:20052856', dim_hubb_ind=1, telescope='xmm',
                               outer_aperture='R2500')

