#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 14/04/2023, 15:14. Copyright (c) The Contributors

import pandas as pd
from astropy.cosmology import Cosmology
from astropy.units import Quantity, Unit, UnitConversionError

from xga import DEFAULT_COSMO, NUM_CORES
from xga.products import ScalingRelation
from xga.relations.clusters.RT import arnaud_r500

LT_REQUIRED_COLS = ['ra', 'dec', 'name', 'redshift']


def luminosity_temperature_pipeline(sample_data: pd.DataFrame, start_aperture: Quantity, min_iter: int = 3,
                                    convergence_frac: float = 0.1, rad_temp_rel: ScalingRelation = arnaud_r500,
                                    cosmo: Cosmology = DEFAULT_COSMO, num_cores: int = NUM_CORES):

    # I want the sample to be passed in as a DataFrame, so I can easily extract the information I need
    if not isinstance(sample_data, pd.DataFrame):
        raise TypeError("The sample_data argument must be a Pandas DataFrame, with the following columns; "
                        "{}".format(', '.join(LT_REQUIRED_COLS)))

    # Also have to make sure that the required information exists in the dataframe, otherwise obviously this tool
    #  is not going to work
    if not set(LT_REQUIRED_COLS).issubset(sample_data.columns):
        raise KeyError("Not all required columns ({}) are present in the sample_data "
                       "DataFrame.".format(', '.join(LT_REQUIRED_COLS)))

    # A key part of this process is a relation between the temperature we measure, and the overdensity radius. As
    #  scaling relations can be between basically any two parameters, and I want this relation object to be an XGA
    #  scaling relation instance, I need to check some things with the rad_temp_rel passed by the user
    if not isinstance(rad_temp_rel, ScalingRelation):
        raise TypeError("The rad_temp_rel argument requires an XGA ScalingRelation instance.")
    elif not rad_temp_rel.x_unit.is_equivalent(Unit('keV')):
        raise UnitConversionError("This pipeline requires a radius-temperature relation, but the x-unit of the "
                                  "rad_temp_rel relation is {bu}. It cannot be converted to "
                                  "keV.".format(bu=rad_temp_rel.x_unit.to_string()))
    elif not rad_temp_rel.y_unit.is_equivalent(Unit('kpc')):
        raise UnitConversionError("This pipeline requires a radius-temperature relation, but the y-unit of the "
                                  "rad_temp_rel relation is {bu}. It cannot be converted to "
                                  "kpc.".format(bu=rad_temp_rel.y_unit.to_string()))

    all_sample_data = sample_data.copy()

    # while sample_data.shape[0] != 0:
    #     samp = ClusterSample(sample_data['ra'].values, sample_data['dec'].values, sample_data['redshift'].values,
    #                          sample_data['name'].values, r500=start_aperture, use_peak=False, clean_obs_threshold=0.7)
    #     single_temp_apec(samp, samp.r500, one_rmf=False, num_cores=num_cores)
    #
    #     # TODO Check the rad_temp_rel, what the axes are, whether I accounted for E(z) in it, etc.
    #     rad_temp_rel.







