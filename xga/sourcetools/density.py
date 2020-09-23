#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 23/09/2020, 17:09. Copyright (c) David J Turner

from typing import Union

import numpy as np
from astropy.units import Quantity

from ..samples.extended import ClusterSample
from ..sas.spec import evselect_spectrum
from ..sources import GalaxyCluster, BaseSource
from ..utils import NHC, ABUND_TABLES
from ..xspec.fakeit import cluster_cr_conv
from ..xspec.fit import single_temp_apec


def cluster_density_profile(sources: Union[GalaxyCluster, ClusterSample], reg_type: str, abund_table: str = "angr",
                            lo_en: Quantity = Quantity(0.5, 'keV'), hi_en: Quantity = Quantity(2.0, 'keV')):
    # If its a single source I shove it in a list so I can just iterate over the sources parameter
    #  like I do when its a Sample object
    if isinstance(sources, BaseSource):
        sources = [BaseSource]

    if abund_table not in ABUND_TABLES:
        ab_list = ", ".join(ABUND_TABLES)
        raise ValueError("{0} is not in the accepted list of abundance tables; {1}".format(abund_table, ab_list))

    try:
        hy_to_elec = NHC[abund_table]
    except KeyError:
        raise NotImplementedError("That is an acceptable abundance table, but I haven't added the conversion factor "
                                  "to the dictionary yet")

    evselect_spectrum(sources, reg_type)
    single_temp_apec(sources, reg_type, abund_table=abund_table)

    temps = Quantity([src.get_temperature(reg_type, "tbabs*apec")[0] for src in sources], 'keV')
    cluster_cr_conv(sources, reg_type, temps, abund_table=abund_table)

    norm_convs = []
    # TODO Think of a better name for this variable
    # These are from the distance and redshift, also the normalising 10^-14 (see my paper for
    #  more of an explanation)
    phys_convs = []
    for src in sources:
        # Both the angular_diameter_distance and redshift are guaranteed to be present here because redshift
        #  is REQUIRED to define GalaxyCluster objects
        factor = ((4*np.pi*(src.angular_diameter_distance.to("cm")*(1+src.redshift))**2) / (hy_to_elec*10**-14)).value
        phys_convs.append(factor)
        norm_convs.append(src.combined_norm_conv_factor(reg_type, lo_en, hi_en).value)

    norm_convs = np.array(norm_convs)
    phys_convs = np.array(phys_convs)

    # At this point we need to multiply by countrate/volume for each source, and this should convert it
    #  to density (after square rooting the result).
    conv_factors = norm_convs * phys_convs
    print(conv_factors)






