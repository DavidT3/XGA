# This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
# Last modified by Jessica Pilling (jp735@sussex.ac.uk) Wed Oct 11 2023, 13:51. Copyright (c) The Contributors

from typing import Union, List

from astropy.units import Quantity

from .._common import region_setup

from .. import OUTPUT, NUM_CORES
from ...sources import BaseSource
from ...samples.base import BaseSample

def _spec_cmds(sources: Union[BaseSource, BaseSample], outer_radius: Union[str, Quantity],
               inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'),
               num_cores: int = NUM_CORES, disable_progress: bool = False, force_gen: bool = False):
    """
    An internal function to generate all the commands necessary to produce a srctool spectrum, but is not
    decorated by the esass_call function, so the commands aren't immediately run. This means it can be used for
    srctool functions that generate custom sets of spectra (like a set of annular spectra for instance), as well
    as for things like the standard srctool_spectrum function which produce relatively boring spectra. At the moment 
    each spectra will also generate a background spectra by default. 

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str/Quantity outer_radius: The name or value of the outer radius to use for the generation of
        the spectrum (for instance 'r200' would be acceptable for a GalaxyCluster, or Quantity(1000, 'kpc')). If
        'region' is chosen (to use the regions in region files), then any inner radius will be ignored.
    :param str/Quantity inner_radius: The name or value of the inner radius to use for the generation of
        the spectrum (for instance 'r500' would be acceptable for a GalaxyCluster, or Quantity(300, 'kpc')). By
        default this is zero arcseconds, resulting in a circular spectrum.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param int num_cores: The number of cores to use, default is set to 90% of available.
    :param bool disable_progress: Setting this to true will turn off the eSASS generation progress bar.
    :param bool force_gen: This boolean flag will force the regeneration of spectra, even if they already exist.
    """
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    if outer_radius != 'region':
        from_region = False
        sources, inner_radii, outer_radii = region_setup(sources, outer_radius, inner_radius, disable_progress, '')
    else:
        # This is used in the extra information dictionary for when the XGA spectrum object is defined
        from_region = True
    
    # Just make sure these values are the expect data type, this matters when the information is
    #  added to the storage strings and file names
    if over_sample is not None:
        over_sample = int(over_sample)
    if min_counts is not None:
        min_counts = int(min_counts)
    if min_sn is not None:
        min_sn = float(min_sn)

    #TODO corresponding to issue #1058, need to adding in rebinning functions. relate to min_counts and min_sn parameters in SAS version

    #TODO check with David about oversampling and group spectra, and the one_rmf parameter (think for erosita you want an RMF for each obsid-inst combo)

    # Defining the various eSASS commands that need to be populated
    # There will be a different command for extended and point sources
    ext_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF"' \
                ' srcreg="mask {em}" backreg=NONE exttype=MAP extmap="{em}" insts="{i}" tstep={ts} xgrid={xg}' \
                ' psftype=NONE'
    #TODO check the point source command in esass with some EDR obs
    pnt_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srcoord="{sc}" todo="SPEC ARF RMF" insts="{i}"' \
                      ' srcreg="{reg}" backreg="{breg}" exttype="POINT" tstep={ts} xgrid={xg}' \
                      ' psftype="2D_PSF"'
    
    # To deal with the object scanning across the telescopes, you need a detection map of the source
    #TODO how to make a detection/extent map
    ext_map_cmd = ""

    # For extended sources, it is best to make a background spectra with a separate command
    bckgr_srctool_cmd = 'cd {d}; srctool eventfiles="{ef}" srccoord="{sc}" todo="SPEC ARF RMF"' \
                        ' srcreg="{breg}" backreg=NONE insts="{i}"' \
                        ' tstep={ts} xgrid={xg} psftype=NONE'