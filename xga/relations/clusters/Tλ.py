#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 28/02/2024, 11:21. Copyright (c) The Contributors

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

xcs_sdss_r500 = ScalingRelation(np.array([1.01, 1.01]), np.array([0.04, 0.03]), power_law, Quantity(60),
                                Quantity(4, 'keV'), r"$\lambda$", r"T$_{\rm{x},500}$",
                                x_lims=Quantity([20, 220]), relation_name='SDSSRM-XCS$_{T_{x},vol}$ 0.5-2.0keV',
                                relation_author='Giles et al.', relation_year='2022',
                                relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                dim_hubb_ind=0)
