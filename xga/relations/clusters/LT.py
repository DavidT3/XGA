#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 28/05/2024, 15:03. Copyright (c) The Contributors

import numpy as np
from astropy.units import Quantity

from ...models.misc import power_law
from ...products.relation import ScalingRelation

# TODO All the XCS-SDSS relations also have an intrinsic scatter parameter, but that isn't
#  implemented yet(see issue #289)

# ---------------------------- Giles et al. SDSSRM-XCS Scaling Relations within R500 ----------------------------------
xcs_sdss_r500_52 = ScalingRelation(np.array([2.63, 0.97]), np.array([0.12, 0.06]), power_law, Quantity(4, 'keV'),
                                   Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500}$",
                                   r"E(z)$^{-1}$L$_{\rm{x},500,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                   relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ 0.5-2.0keV',
                                   relation_author='Giles et al.', relation_year='2022',
                                   relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                   dim_hubb_ind=-1)

xcs_sdss_r500_52_target = ScalingRelation(np.array([2.63, 1.04]), np.array([0.2, 0.09]), power_law, Quantity(4, 'keV'),
                                          Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500}$",
                                          r"E(z)$^{-1}$L$_{\rm{x},500,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                          relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol,tar}$ 0.5-2.0keV',
                                          relation_author='Giles et al.', relation_year='2022',
                                          relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                          dim_hubb_ind=-1)

xcs_sdss_r500_52_serin = ScalingRelation(np.array([2.0, 0.66]), np.array([0.22, 0.09]), power_law, Quantity(4, 'keV'),
                                         Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500}$",
                                         r"E(z)$^{-1}$L$_{\rm{x},500,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                         relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol,serin}$ 0.5-2.0keV',
                                         relation_author='Giles et al.', relation_year='2022',
                                         relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                         dim_hubb_ind=-1)

xcs_sdss_r500_bol = ScalingRelation(np.array([3.07, 3.05]), np.array([0.12, 0.18]), power_law, Quantity(4, 'keV'),
                                    Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500}$",
                                    r"E(z)$^{-1}$L$_{\rm{x},500,bol}$", x_lims=Quantity([1, 12], 'keV'),
                                    relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ Bolometric',
                                    relation_author='Giles et al.', relation_year='2022',
                                    relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                    dim_hubb_ind=-1)
# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------- Giles et al. SDSSRM-XCS Scaling Relations within R500 CORE-EXCISED ---------------------
xcs_sdss_r500ce_52 = ScalingRelation(np.array([2.46, 0.74]), np.array([0.10, 0.03]), power_law, Quantity(4, 'keV'),
                                     Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500ce}$",
                                     r"E(z)$^{-1}$L$_{\rm{x},500ce,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                     relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ 0.5-2.0keV',
                                     relation_author='Giles et al.', relation_year='2022',
                                     relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                     dim_hubb_ind=-1)

xcs_sdss_r500ce_52_target = ScalingRelation(np.array([2.58, 0.73]), np.array([0.16, 0.05]), power_law,
                                            Quantity(4, 'keV'), Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500ce}$",
                                            r"E(z)$^{-1}$L$_{\rm{x},500ce,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                            relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol,tar}$ 0.5-2.0keV',
                                            relation_author='Giles et al.', relation_year='2022',
                                            relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                            dim_hubb_ind=-1)

xcs_sdss_r500ce_52_serin = ScalingRelation(np.array([1.84, 0.54]), np.array([0.21, 0.07]), power_law,
                                           Quantity(4, 'keV'), Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},500ce}$",
                                           r"E(z)$^{-1}$L$_{\rm{x},500ce,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                           relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol,serin}$ 0.5-2.0keV',
                                           relation_author='Giles et al.', relation_year='2022',
                                           relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                           dim_hubb_ind=-1)

# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------- Giles et al. SDSSRM-XCS Scaling Relations within R2500 ---------------------------------
xcs_sdss_r2500_52 = ScalingRelation(np.array([2.89, 0.57]), np.array([0.13, 0.04]), power_law, Quantity(4, 'keV'),
                                    Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},2500}$",
                                    r"E(z)$^{-1}$L$_{\rm{x},2500,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                    relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol}$ 0.5-2.0keV',
                                    relation_author='Giles et al.', relation_year='2022',
                                    relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                    dim_hubb_ind=-1)

xcs_sdss_r2500_52_target = ScalingRelation(np.array([2.69, 0.68]), np.array([0.19, 0.06]), power_law,
                                           Quantity(4, 'keV'), Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},2500}$",
                                           r"E(z)$^{-1}$L$_{\rm{x},2500,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                           relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol,tar}$ 0.5-2.0keV',
                                           relation_author='Giles et al.', relation_year='2022',
                                           relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                           dim_hubb_ind=-1)

xcs_sdss_r2500_52_serin = ScalingRelation(np.array([2.56, 0.44]), np.array([0.33, 0.07]), power_law, Quantity(4, 'keV'),
                                          Quantity(0.8e+44, 'erg / s'), r"T$_{\rm{x},2500}$",
                                          r"E(z)$^{-1}$L$_{\rm{x},2500,0.5-2.0}$", x_lims=Quantity([1, 12], 'keV'),
                                          relation_name=r'SDSSRM-XCS$_{T_{\rm{x}},vol,serin}$ 0.5-2.0keV',
                                          relation_author='Giles et al.', relation_year='2022',
                                          relation_doi='https://doi.org/10.1093/mnras/stac2414',
                                          dim_hubb_ind=-1)
# ---------------------------------------------------------------------------------------------------------------------



