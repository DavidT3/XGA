#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 25/02/2021, 13:52. Copyright (c) David J Turner

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

# These are from the classic M-T relation paper published by Arnaud, as R-T relations are a byproduct of M-T relations
arnaud_r200 = ScalingRelation(np.array([0.57, 1674]), np.array([0.02, 23]), power_law, Quantity(5, 'keV'),
                              Quantity(1, 'kpc'), r"T$_{\rm{x}}$", "E(z)R$_{200}$",
                              relation_author='Arnaud et al.', relation_year='2005',
                              relation_doi='10.1051/0004-6361:20052856',
                              relation_name=r'R$_{200}$-Temperature', x_lims=Quantity([1, 12], 'keV'))

arnaud_r500 = ScalingRelation(np.array([0.57, 1104]), np.array([0.02, 13]), power_law, Quantity(5, 'keV'),
                              Quantity(1, 'kpc'), r"T$_{\rm{x}}$", "E(z)R$_{500}$",
                              relation_author='Arnaud et al.', relation_year='2005',
                              relation_doi='10.1051/0004-6361:20052856',
                              relation_name=r'R$_{500}$-Temperature', x_lims=Quantity([1, 12], 'keV'))

arnaud_r2500 = ScalingRelation(np.array([0.56, 491]), np.array([0.02, 4]), power_law, Quantity(5, 'keV'),
                               Quantity(1, 'kpc'), r"T$_{\rm{x}}$", "E(z)R$_{2500}$",
                               relation_author='Arnaud et al.', relation_year='2005',
                               relation_doi='10.1051/0004-6361:20052856',
                               relation_name=r'R$_{2500}$-Temperature', x_lims=Quantity([1, 12], 'keV'))

# These are equivelant relations specifically measured from the hot clusters (T > 3.5keV) in their sample
arnaud_r200_hot = ScalingRelation(np.array([0.5, 1714]), np.array([0.05, 30]), power_law, Quantity(5, 'keV'),
                                  Quantity(1, 'kpc'), r"T$_{\rm{x}}$", "E(z)R$_{200}$",
                                  relation_author='Arnaud et al.', relation_year='2005',
                                  relation_doi='10.1051/0004-6361:20052856',
                                  relation_name=r'T$_{\rm{x}}>3.5$keV R$_{200}$-Temperature',
                                  x_lims=Quantity([1, 12], 'keV'))

arnaud_r500_hot = ScalingRelation(np.array([0.5, 1129]), np.array([0.05, 17]), power_law, Quantity(5, 'keV'),
                                  Quantity(1, 'kpc'), r"T$_{\rm{x}}$", "E(z)R$_{500}$",
                                  relation_author='Arnaud et al.', relation_year='2005',
                                  relation_doi='10.1051/0004-6361:20052856',
                                  relation_name=r'T$_{\rm{x}}>3.5$keV R$_{500}$-Temperature',
                                  x_lims=Quantity([1, 12], 'keV'))

arnaud_r2500_hot = ScalingRelation(np.array([0.5, 500]), np.array([0.03, 5]), power_law, Quantity(5, 'keV'),
                                   Quantity(1, 'kpc'), r"T$_{\rm{x}}$", "E(z)R$_{2500}$",
                                   relation_author='Arnaud et al.', relation_year='2005',
                                   relation_doi='10.1051/0004-6361:20052856',
                                   relation_name=r'Overdensity T$_{\rm{x}}>3.5$keV R$_{2500}$-Temperature',
                                   x_lims=Quantity([1, 12], 'keV'))
