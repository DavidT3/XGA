#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 13/11/2023, 14:18. Copyright (c) The Contributors

import os.path
import warnings
from typing import Union, Tuple, List

from astropy.units import Quantity

from .misc import cifbuild
from ..samples.base import BaseSample
from ..sources import BaseSource, GalaxyCluster
from ..sources.base import NullSource
from ..utils import RAD_LABELS, NUM_CORES
from ..exceptions import NotAssociatedError

def region_setup(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
                 inner_radius: Union[str, Quantity], disable_progress: bool, obs_id: str, num_cores: int = NUM_CORES) \
        -> Tuple[Union[BaseSource, BaseSample], List[Quantity], List[Quantity]]:
    """
    The preparation and value checking stage for SAS spectrum generation.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')).
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default this is zero arcseconds, resulting in a circular spectrum.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    :param str obs_id: Only used if the 'region' radius name is passed, the ObsID to retrieve the region for.
    :param int num_cores: The number of cores to be used, will be passed to cifbuild.
    :return: The source objects, a list of inner radius quantities, and a list of outer radius quantities.
    :rtype: Tuple[Union[BaseSource, BaseSample], List[Quantity], List[Quantity]]
    """
    # NullSources are not allowed to have spectra, as they can have any observations associated and thus won't
    #  necessarily overlap
    if isinstance(sources, NullSource):
        raise TypeError("You cannot create spectra of a NullSource")

    if isinstance(sources, BaseSource):
        sources = [sources]

    # Checking that the user hasn't passed BaseSources
    if not all([type(src) != BaseSource for src in sources]):
        raise TypeError("You cannot generate spectra from a BaseSource object, really you shouldn't be using "
                        "them at all, they are mostly useful as a superclass.")

    # Issuing a warning to the user that one or one sources have not been detected
    if not all([src.detected for src in sources]):
        warnings.warn("Not all of these sources have been detected, the spectra generated may not be helpful.")

    # Checking that inner radii that have been passed into the spectrum generation aren't nonsense
    if isinstance(inner_radius, str) and inner_radius not in RAD_LABELS:
        raise ValueError("You have passed a radius name rather than a value for 'inner_radius', but it is "
                         "not a valid name, please use one of the following:\n {}".format(", ".join(RAD_LABELS)))

    elif isinstance(inner_radius, str) and inner_radius in ["r2500", "r500", "r200"] and \
            not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("The {} radius is only valid for GalaxyCluster objects".format(inner_radius))

    # One radius can be passed for a whole sample, but this checks to make sure that if there are multiple sources,
    #  and multiple radii have been passed, there are the same number of sources and radii
    elif isinstance(inner_radius, Quantity) and len(sources) != 1 and not inner_radius.isscalar \
            and len(sources) != len(inner_radius):
        raise ValueError("Your sample has {s} sources, but your inner_radius variable only has {i} entries. Please "
                         "pass only one inner_radius or the same number as there are "
                         "sources".format(s=len(sources), i=len(inner_radius)))

    # Checking that outer_radius radii that have been passed into the spectrum generation aren't nonsense
    if isinstance(outer_radius, str) and outer_radius not in RAD_LABELS:
        raise ValueError("You have passed a radius name rather than a value for 'outer_radius', but it is "
                         "not a valid name, please use one of the following:\n {}".format(", ".join(RAD_LABELS)))
    elif isinstance(outer_radius, str) and outer_radius in ["r2500", "r500", "r200"] and \
            not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("The {} radius is only valid for GalaxyCluster objects".format(outer_radius))
    elif isinstance(outer_radius, Quantity) and len(sources) != 1 and not outer_radius.isscalar \
            and len(sources) != len(outer_radius):
        raise ValueError("Your sample has {s} sources, but your outer_radius variable only has {o} entries. Please "
                         "pass only one outer_radius or the same number as there are "
                         "sources".format(s=len(sources), o=len(outer_radius)))

    # A crude way to store the radii but I'm tired and this will work fine
    final_inner = []
    final_outer = []
    # I need to convert the radii to the same units and compare them, and to make sure they
    #  are actually in distance units. The distance unit checking is done by convert_radius
    for s_ind, src in enumerate(sources):
        # Converts the inner and outer radius for this source into the same unit
        if isinstance(outer_radius, str) and outer_radius != 'region':
            cur_out_rad = src.get_radius(outer_radius, 'deg')
        elif isinstance(outer_radius, str) and outer_radius == 'region':
            reg = src.source_back_regions('region', obs_id)[0]
            cur_out_rad = Quantity([reg.width.to('deg').value/2, reg.height.to('deg').value/2], 'deg')
        elif outer_radius.isscalar:
            cur_out_rad = src.convert_radius(outer_radius, 'deg')
        else:
            cur_out_rad = src.convert_radius(outer_radius[s_ind], 'deg')

        # We need to check that the outer radius isn't region, because for region objects we ignore whatever
        #  inner radius has been passed and just set it 0
        if outer_radius == 'region':
            cur_inn_rad = Quantity([0, 0], 'deg')
        elif isinstance(inner_radius, str):
            cur_inn_rad = src.get_radius(inner_radius, 'deg')
        elif inner_radius.isscalar:
            cur_inn_rad = src.convert_radius(inner_radius, 'deg')
        else:
            cur_inn_rad = src.convert_radius(inner_radius[s_ind], 'deg')

        # Then we can check to make sure that the outer radius is larger than the inner radius
        if outer_radius != 'region' and cur_inn_rad >= cur_out_rad:
            raise ValueError("The inner_radius of {s} is greater than or equal to the outer_radius".format(s=src.name))
        else:
            final_inner.append(cur_inn_rad)
            final_outer.append(cur_out_rad)

    # Have to make sure that all observations have an up to date cif file.
    cifbuild(sources, disable_progress=disable_progress, num_cores=num_cores)

    return sources, final_inner, final_outer


def check_pattern(pattern: Union[str, int]) -> Tuple[str, str]:
    """
    A very simple (and not exhaustive) checker for XMM SAS pattern expressions.

    :param str/int pattern: The pattern selection expression to be checked.
    :return: A string pattern selection expression, and a pattern representation that should be safe for naming
        SAS files with.
    :rtype: Tuple[str, str]
    """

    if isinstance(pattern, int):
        pattern = '==' + str(pattern)
    elif not isinstance(pattern, str):
        raise TypeError("Pattern arguments must be either an integer (we then assume only events with that pattern "
                        "should be selected) or a SAS selection command (e.g. 'in [1:4]' or '<= 4').")

    # First off I remove whitespace from the beginning and end of the term
    pattern = pattern.strip()
    # pattern = pattern.replace(' ', '')
    # Sometimes we will pass in patterns that have been converted to the 'XGA format', if you want to call it
    #  that. This namely happens when we're reading light curves that have been saved to disk back in. As such we
    #  replace the XGA-isms with their original string meanings
    pattern = pattern.replace('lteq', '<=').replace('gteq', '>=').replace('eq', '==') \
        .replace('lteq', '<=').replace('lt', '<').replace('gt', '>')

    # Then we check for understandable selection commands; inequalities, equals, and 'in'
    if pattern[:2] not in ['in', '<=', '>=', '=='] and pattern[:1] not in ['<', '>']:
        raise ValueError("First part of a pattern statement must be either 'in', '<=', '>=', '==', '<', or '>'.")

    if pattern[:2] == 'in' and '[' not in pattern and '(' not in pattern:
        raise ValueError("If a pattern statement uses 'in', either a '[' (for inclusive lower limit) or '(' (for "
                         "exclusive lower limit) must be in the statement.")

    if pattern[:2] == 'in' and ']' not in pattern and ')' not in pattern:
        raise ValueError("If a pattern statement uses 'in', either a ']' (for inclusive upper limit) or ')' (for "
                         "exclusive upper limit) must be in the statement.")

    if pattern[:2] == 'in' and ':' not in pattern:
        raise ValueError("If a pattern statement uses 'in', either a ':' must be present in the statement to separate "
                         "lower and upper limits.")

    # SAS doesn't like having file names with special characters, so I am trying to come up with safe replacements
    #  that still convey what the pattern selection was
    # .replace('in', '')
    patt_file_name = pattern.replace(' ', '').replace('<=', 'lteq').replace('>=', 'gteq').replace('==', 'eq')\
        .replace('<=', 'lteq').replace('<', 'lt').replace('>', 'gt')

    return pattern, patt_file_name


def _gen_detmap_cmd(source: BaseSource, obs_id: str, inst: str, bin_size: int = 200) -> Tuple[str, str, str]:
    """
    An internal method for generating SAS commands required to create detector maps for the weighting of ARFs.

    :param BaseSource source: The source for which the parent method is generating ARFs for, and that needs
        a detector map.
    :param str obs_id: The ObsID of the data we are generating ARFs for.
    :param str inst: The instrument of the data we are generating ARFs for. NOTE - ideally this instrument WILL NOT
        be used for the detector map, as it is beneficial to source a detector map from a different instrument to
        the one you are generating ARFs for.
    :param int bin_size: The x and y binning that should be applied to the image. Larger numbers will cause ARF
        generation to be faster, but arguably the results will be less accurate.
    :return: The command to generate the requested detector map (will be blank if the detector map already
        exists), the path where the detmap will be after the command is run (i.e. the ObsID directory if it was
        already generated, or the temporary directory if it has just been generated), and the final output path
        of the detector.
    :rtype: Tuple[str, str, str]
    """
    # This is the command that will be filled out to generate the detmap of our dreams!
    detmap_cmd = "evselect table={e} imageset={d} xcolumn=DETX ycolumn=DETY imagebinning=binSize ximagebinsize={bs} " \
                 "yimagebinsize={bs} {ex}"

    # Some settings depend on the instrument, we use different patterns for different instruments
    if "pn" in inst:
        # Also the upper channel limit is different for EPN and EMOS detectors
        spec_lim = 20479
        # This is an expression without region information to be used for making the detmaps
        #  required for ARF generation, we start off assuming we'll use a MOS observation as the detmap
        d_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0)'"

        # The detmap for the arfgen call should ideally not be from the same instrument as the observation,
        #  so for PN we preferentially select MOS2 (as MOS1 was damaged). However if there isn't a MOS2
        #  events list from the same observation then we select MOS1, and failing that we use PN.
        try:
            detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos2')[0]
        except NotAssociatedError:
            try:
                detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos1')[0]
            except NotAssociatedError:
                detmap_evts = source.get_products("events", obs_id=obs_id, inst='pn')[0]
                # If all is lost and there are no MOS event lists then we must revert to the PN expression
                d_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0)'"

    elif "mos" in inst:
        spec_lim = 11999
        # This is an expression without region information to be used for making the detmaps
        #  required for ARF generation, we start off assuming we'll use the PN observation as the detmap
        d_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0)'"

        # The detmap for the arfgen call should ideally not be from the same instrument as the observation,
        #  so for MOS observations we preferentially select PN. However if there isn't a PN events list
        #  from the same observation then for MOS2 we select MOS1, and for MOS1 we select MOS2 (as they
        #  are rotated wrt one another it is still semi-valid), and failing that MOS2 will get MOS2 and MOS1
        #  will get MOS1.
        if inst[-1] == 1:
            cur = '1'
            opp = '2'
        else:
            cur = '2'
            opp = '1'
        try:
            detmap_evts = source.get_products("events", obs_id=obs_id, inst='pn')[0]
        except NotAssociatedError:
            # If we must use a MOS detmap then we have to use the MOS expression
            d_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0)'"
            try:
                detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos' + opp)[0]
            except NotAssociatedError:
                detmap_evts = source.get_products("events", obs_id=obs_id, inst='mos' + cur)[0]
    else:
        raise ValueError("You somehow have an illegal value for the instrument name...")

    det_map = "{o}_{i}_bin{bs}_detmap.fits".format(o=detmap_evts.obs_id, i=detmap_evts.instrument, bs=bin_size)
    det_map_path = os.path.join(OUTPUT, obs_id, det_map)

    # If the detmap that we've decided we need already exists, then we don't need to generate it again
    if os.path.exists(det_map_path):
        d_cmd_str = ""
        det_map_cmd_path = det_map_path
    else:
        # Does the same thing for the evselect command to make the detmap
        d_cmd_str = detmap_cmd.format(e=detmap_evts.path, d=det_map, ex=d_expr, bs=bin_size) + "; "
        det_map_cmd_path = det_map

    return d_cmd_str, det_map_cmd_path, det_map_path
