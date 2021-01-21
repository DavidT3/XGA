#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 21/01/2021, 11:45. Copyright (c) David J Turner

from typing import List, Union

from astropy.units import Quantity

from ... import NUM_CORES
from ...samples.base import BaseSample
from ...sas import spectrum_set
from ...sources import BaseSource


def single_temp_apec_profile(sources: Union[BaseSource, BaseSample], radii: Union[Quantity, List[Quantity]],
                             start_temp: Quantity = Quantity(3.0, "keV"), start_met: float = 0.3,
                             lum_en: Quantity = Quantity([[0.5, 2.0], [0.01, 100.0]], "keV"), freeze_nh: bool = True,
                             freeze_met: bool = True, link_norm: bool = False, lo_en: Quantity = Quantity(0.3, "keV"),
                             hi_en: Quantity = Quantity(7.9, "keV"), par_fit_stat: float = 1., lum_conf: float = 68.,
                             abund_table: str = "angr", fit_method: str = "leven", group_spec: bool = True,
                             min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                             num_cores: int = NUM_CORES):
    """
    A function that allows for the fitting of sets of annular spectra (generated from objects such as galaxy
    clusters) with an absorbed plasma emission model (tbabs*apec). This function fits the annuli completely
    independently of one another.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param List[Quantity]/Quantity radii: A list of non-scalar quantities containing the boundary radii of the
        annuli for the sources. A single quantity containing at least three radii may be passed if one source
        is being analysed, but for multiple sources there should be a quantity (with at least three radii), PER
        source.
    :param Quantity start_temp: The initial temperature for the fit.
    :param start_met: The initial metallicity for the fit (in ZSun).
    :param Quantity lum_en: Energy bands in which to measure luminosity.
    :param bool freeze_nh: Whether the hydrogen column density should be frozen.
    :param bool freeze_met: Whether the metallicity parameter in the fit should be frozen.
    :param bool link_norm: Whether the normalisations of different spectra should be linked during fitting.
    :param Quantity lo_en: The lower energy limit for the data to be fitted.
    :param Quantity hi_en: The upper energy limit for the data to be fitted.
    :param float par_fit_stat: The delta fit statistic for the XSPEC 'error' command.
    :param float lum_conf: The confidence level for XSPEC luminosity measurements.
    :param str abund_table: The abundance table to use for the fit.
    :param str fit_method: The XSPEC fit method to use.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable. e.g. if
        over_sample=3 then the minimum width of a group is 1/3 of the resolution FWHM at that energy.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
        slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to 90% of available.
    """
    # We make sure the requested sets of annular spectra have actually been generated
    spectrum_set(sources, radii, group_spec, min_counts, min_sn, over_sample, one_rmf, num_cores)



