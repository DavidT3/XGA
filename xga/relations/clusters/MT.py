#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 17/04/2023, 21:04. Copyright (c) The Contributors

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

# These are from the classic M-T relation published by Arnaud
arnaud_m200 = ScalingRelation(np.array([1.72, 5.34]), np.array([0.10, 0.22]), power_law, Quantity(5, 'keV'),
                              Quantity(1e+14, 'solMass'), r"T$_{\rm{x}}$", "E(z)M$_{200}$",
                              x_lims=Quantity([1, 12], 'keV'), relation_name='Hydrostatic Mass-Temperature',
                              relation_author='Arnaud et al.', relation_year='2005',
                              relation_doi='10.1051/0004-6361:20052856', dim_hubb_ind=1)

arnaud_m500 = ScalingRelation(np.array([1.71, 3.84]), np.array([0.09, 0.14]), power_law, Quantity(5, 'keV'),
                              Quantity(1e+14, 'solMass'), r"T$_{\rm{x}}$", "E(z)M$_{500}$",
                              x_lims=Quantity([1, 12], 'keV'), relation_name='Hydrostatic Mass-Temperature',
                              relation_author='Arnaud et al.', relation_year='2005',
                              relation_doi='10.1051/0004-6361:20052856', dim_hubb_ind=1)

arnaud_m2500 = ScalingRelation(np.array([1.70, 1.69]), np.array([0.07, 0.05]), power_law, Quantity(5, 'keV'),
                               Quantity(1e+14, 'solMass'), r"T$_{\rm{x}}$", "E(z)M$_{2500}$",
                               x_lims=Quantity([1, 12], 'keV'), relation_name='Hydrostatic Mass-Temperature',
                               relation_author='Arnaud et al.', relation_year='2005',
                               relation_doi='10.1051/0004-6361:20052856', dim_hubb_ind=1)

# These are the XXL weak-lensing mass to temperature scaling relation(s, if I can be bothered to put more than
#  one of them). I've averaged the parameter errors
# TODO This doesn't seem to be working, chat to Paul - THE ERROR ON THE SECOND PARAMETER IS DEFINITELY WRONG
xxl_m500 = ScalingRelation(np.array([1.78, 3.63]), np.array([0.345, 0.165]), power_law, Quantity(1, 'keV'),
                           Quantity(1e+13, 'solMass'), r"T$_{\rm{x},300kpc}$", "E(z)$^{-1}$M$_{500}$",
                           x_lims=Quantity([1, 12], 'keV'), relation_name='Weak Lensing Mass-Temperature',
                           relation_author='Lieu et al.', relation_year='2016',
                           relation_doi='10.1051/0004-6361/201526883', dim_hubb_ind=-1)

# TODO SECOND PARAMETER ERROR ALSO WRONG HERE, I raised 10 to the power of the intercept, but not sure what
#  to do with the errors - THIS MAY BE BAD
xxl_cosmos_cccp_m500 = ScalingRelation(np.array([1.67, 3.72]), np.array([0.12, 0.09]), power_law, Quantity(1, 'keV'),
                                       Quantity(1e+13, 'solMass'), r"T$_{\rm{x},300kpc}$", "E(z)$^{-1}$M$_{500}$",
                                       x_lims=Quantity([1, 12], 'keV'), relation_name='Weak Lensing Mass-Temperature',
                                       relation_author='Lieu et al.', relation_year='2016',
                                       relation_doi='10.1051/0004-6361/201526883', dim_hubb_ind=-1)

# TODO SAME PROBLEM HERE
arnaud_gm500 = ScalingRelation(np.array([2.10, 4.48]), np.array([0.05, 0.01]), power_law, Quantity(5, 'keV'),
                               Quantity(1e+13, 'solMass'), r"T$_{\rm{x}}$", "E(z)$^{-1}$M$_{g,500}$",
                               x_lims=Quantity([1, 12], 'keV'), relation_name='Gas Mass-Temperature',
                               relation_author='Arnaud et al.', relation_year='2007',
                               relation_doi='10.1051/0004-6361:20078541', dim_hubb_ind=-1)

