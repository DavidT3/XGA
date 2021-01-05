#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 05/01/2021, 13:00. Copyright (c) David J Turner

import os
import warnings
from shutil import rmtree
from typing import Union

import numpy as np
from tqdm import tqdm

from .misc import cifbuild
from .run import sas_call
from .. import OUTPUT, NUM_CORES
from ..samples.base import BaseSample
from ..sources import BaseSource, ExtendedSource, GalaxyCluster
from ..sources.base import NullSource
from ..utils import xmm_sky


def _spec_setup(sources, reg_type, allowed_bounds, disable_progress):
    # This function supports passing both individual sources and sets of sources
    if isinstance(sources, BaseSource):
        sources = [sources]

    # NullSources are not allowed to have spectra, as they can have any observations associated and thus won't
    #  necessarily overlap
    if isinstance(sources, NullSource):
        raise TypeError("You cannot create spectra of a NullSource")

    if not all([type(src) != BaseSource for src in sources]):
        raise TypeError("You cannot generate spectra from a BaseSource object, really you shouldn't be using "
                        "them at all, they are mostly useful as a superclass.")
    elif not all([src.detected for src in sources]):
        warnings.warn("Not all of these sources have been detected, the spectra generated may not be helpful.")
    elif reg_type not in allowed_bounds:
        raise ValueError("The only valid choices for reg_type are:\n {}".format(", ".join(allowed_bounds)))
    elif reg_type in ["r2500", "r500", "r200"] and not all([type(src) == GalaxyCluster for src in sources]):
        raise TypeError("You cannot use ExtendedSource classes with {}, "
                        "they have no overdensity radii.".format(reg_type))

    # Have to make sure that all observations have an up to date cif file.
    cifbuild(sources, disable_progress=disable_progress)

    return sources


def _spec_cmds():
    pass


# TODO Add an option to generate core-excised spectra.
@sas_call
def evselect_spectrum(sources: Union[BaseSource, BaseSample], reg_type: str, group_spec: bool = True,
                      min_counts: int = 5, min_sn: float = None, over_sample: float = None, one_rmf: bool = True,
                      num_cores: int = NUM_CORES, disable_progress: bool = False):
    """
    A wrapper for all of the SAS processes necessary to generate an XMM spectrum that can be analysed
    in XSPEC. Every observation associated with this source, and every instrument associated with that
    observation, will have a spectrum generated using the specified region type as as boundary. It is possible
    to generate both grouped and ungrouped spectra using this function, with the degree of grouping set
    by the min_counts, min_sn, and oversample parameters.

    :param BaseSource/BaseSample sources: A single source object, or a sample of sources.
    :param str reg_type: Tells the method what region source you want to use, for instance r500 or r200.
    :param bool group_spec: A boolean flag that sets whether generated spectra are grouped or not.
    :param float min_counts: If generating a grouped spectrum, this is the minimum number of counts per channel.
        To disable minimum counts set this parameter to None.
    :param float min_sn: If generating a grouped spectrum, this is the minimum signal to noise in each channel.
        To disable minimum signal to noise set this parameter to None.
    :param float over_sample: The minimum energy resolution for each group, set to None to disable.
    :param bool one_rmf: This flag tells the method whether it should only generate one RMF for a particular
        ObsID-instrument combination - this is much faster in some circumstances, however the RMF does depend
    slightly on position on the detector.
    :param int num_cores: The number of cores to use (if running locally), default is set to
        90% of available.
    :param bool disable_progress: Setting this to true will turn off the SAS generation progress bar.
    """
    allowed_bounds = ["region", "r2500", "r500", "r200", "custom"]
    sources = _spec_setup(sources, reg_type, allowed_bounds, disable_progress)

    # Define the various SAS commands that need to be populated, for a useful spectrum you also need ARF/RMF
    spec_cmd = "cd {d}; cp ../ccf.cif .; export SAS_CCF={ccf}; evselect table={e} withspectrumset=yes " \
               "spectrumset={s} energycolumn=PI spectralbinsize=5 withspecranges=yes specchannelmin=0 " \
               "specchannelmax={u} {ex}"

    rmf_cmd = "rmfgen rmfset={r} spectrumset='{s}' detmaptype=flat extendedsource={es}"

    # Don't need to run backscale separately, as this arfgen call will do it automatically
    arf_cmd = "arfgen spectrumset='{s}' arfset={a} withrmfset=yes rmfset='{r}' badpixlocation={e} " \
              "extendedsource={es} detmaptype=flat setbackscale=yes"

    # If the user wants to group spectra, then we'll need this template command:
    grp_cmd = "specgroup spectrumset={s} overwrite=yes backgndset={b} arfset={a} rmfset={r} addfilenames=no"

    stack = False  # This tells the sas_call routine that this command won't be part of a stack
    execute = True  # This should be executed immediately

    sources_cmds = []
    sources_paths = []
    sources_extras = []
    sources_types = []

    # TODO Give myself cause to remove this by speeding up region string generation
    # This progress bar is being placed here because it can take QUITE a while to generate the SAS region strings
    spec_prep = tqdm(desc="Preparing evselect spectrum commands", total=len(sources), disable=disable_progress)
    for source in sources:
        # rmfgen and arfgen both take arguments that describe if something is an extended source or not,
        #  so we check the source type
        if isinstance(source, (ExtendedSource, GalaxyCluster)):
            ex_src = "yes"
        else:
            ex_src = "no"
        cmds = []
        final_paths = []
        extra_info = []
        # Check which event lists are associated with each individual source
        for pack in source.get_products("events", just_obj=False):
            obs_id = pack[0]
            inst = pack[1]

            if not os.path.exists(OUTPUT + obs_id):
                os.mkdir(OUTPUT + obs_id)

            # Got to check if this spectrum already exists
            exists = [match for match in source.get_products("spectrum", obs_id, inst, just_obj=False)
                      if reg_type in match]
            if len(exists) == 1 and exists[0][-1].usable:
                continue

            # If there is no match to a region, the source region returned by this method will be None,
            #  and if the user wants to generate spectra from region files, we have to ignore that observations
            if reg_type == "region" and source.source_back_regions("region", obs_id)[0] is None:
                continue

            # This method returns a SAS expression for the source and background regions - excluding interlopers
            reg, b_reg = source.get_sas_region(reg_type, obs_id, inst, xmm_sky)

            # Some settings depend on the instrument, XCS uses different patterns for different instruments
            if "pn" in inst:
                # Also the upper channel limit is different for EPN and EMOS detectors
                spec_lim = 20479
                expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0) && {s}'".format(s=reg)
                b_expr = "expression='#XMMEA_EP && (PATTERN <= 4) && (FLAG .eq. 0) && {s}'".format(s=b_reg)
            elif "mos" in inst:
                spec_lim = 11999
                expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0) && {s}'".format(s=reg)
                b_expr = "expression='#XMMEA_EM && (PATTERN <= 12) && (FLAG .eq. 0) && {s}'".format(s=b_reg)
            else:
                raise ValueError("You somehow have an illegal value for the instrument name...")

            # Some of the SAS tasks have issues with filenames with a '+' in them for some reason, so this
            #  replaces any + symbols that may be in the source name with another character
            source_name = source.name.replace("+", "x")

            # Just grabs the event list object
            evt_list = pack[-1]
            # Sets up the file names of the output files
            dest_dir = OUTPUT + "{o}/{i}_{n}_temp/".format(o=obs_id, i=inst, n=source_name)
            spec = "{o}_{i}_{n}_{bt}_spec.fits".format(o=obs_id, i=inst, n=source_name, bt=reg_type)
            b_spec = "{o}_{i}_{n}_{bt}_backspec.fits".format(o=obs_id, i=inst, n=source_name, bt=reg_type)
            arf = "{o}_{i}_{n}_{bt}.arf".format(o=obs_id, i=inst, n=source_name, bt=reg_type)
            b_arf = "{o}_{i}_{n}_{bt}_back.arf".format(o=obs_id, i=inst, n=source_name, bt=reg_type)
            ccf = dest_dir + "ccf.cif"

            # Fills out the evselect command to make the main and background spectra
            s_cmd_str = spec_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, s=spec, u=spec_lim, ex=expr)
            sb_cmd_str = spec_cmd.format(d=dest_dir, ccf=ccf, e=evt_list.path, s=b_spec, u=spec_lim, ex=b_expr)

            # This chunk adds rmfgen commands depending on whether we're using a universal RMF or
            #  an individual one for each spectrum. Also adds arfgen commands on the end, as they depend on
            #  the rmf.
            if one_rmf:
                rmf = "{o}_{i}_{n}_{bt}.rmf".format(o=obs_id, i=inst, n=source_name, bt="universal")
                b_rmf = rmf
            else:
                rmf = "{o}_{i}_{n}_{bt}.rmf".format(o=obs_id, i=inst, n=source_name, bt=reg_type)
                b_rmf = "{o}_{i}_{n}_{bt}_back.rmf".format(o=obs_id, i=inst, n=source_name, bt=reg_type)

            if one_rmf and not os.path.exists(dest_dir + rmf):
                cmd_str = ";".join([s_cmd_str, rmf_cmd.format(r=rmf, s=spec, es=ex_src),
                                    arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path, es=ex_src), sb_cmd_str,
                                    arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path, es=ex_src)])
            elif not one_rmf and not os.path.exists(dest_dir + rmf):
                cmd_str = ";".join([s_cmd_str, rmf_cmd.format(r=rmf, s=spec, es=ex_src),
                                    arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path, es=ex_src)]) + ";"
                cmd_str += ";".join([sb_cmd_str, rmf_cmd.format(r=b_rmf, s=b_spec, es=ex_src),
                                    arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path, es=ex_src)])
            else:
                cmd_str = ";".join([s_cmd_str, arf_cmd.format(s=spec, a=arf, r=rmf, e=evt_list.path,
                                                              es=ex_src)]) + ";"
                cmd_str += ";".join([sb_cmd_str, arf_cmd.format(s=b_spec, a=b_arf, r=b_rmf, e=evt_list.path,
                                                                es=ex_src)])

            # If the user wants to produce grouped spectra, then this if statement is triggered and adds a specgroup
            #  command at the end. The groupspec command will replace the ungrouped spectrum.
            if group_spec:
                new_grp = grp_cmd.format(s=spec, b=b_spec, r=rmf, a=arf)
                if min_counts is not None:
                    new_grp += " mincounts={mc}".format(mc=min_counts)
                if min_sn is not None:
                    new_grp += " minSN={msn}".format(msn=min_sn)
                if over_sample is not None:
                    new_grp += " oversample={os}".format(os=over_sample)
                cmd_str += "; " + new_grp

            # Adds clean up commands to move all generated files and remove temporary directory
            cmd_str += "; mv * ../; cd ..; rm -r {d}".format(d=dest_dir)
            cmds.append(cmd_str)  # Adds the full command to the set
            # If something got interrupted and the temp directory still exists, this will remove it
            if os.path.exists(dest_dir):
                rmtree(dest_dir)
            # Makes sure the whole path to the temporary directory is created
            os.makedirs(dest_dir)

            final_paths.append(os.path.join(OUTPUT, obs_id, spec))
            extra_info.append({"reg_type": reg_type, "rmf_path": os.path.join(OUTPUT, obs_id, rmf),
                               "arf_path": os.path.join(OUTPUT, obs_id, arf),
                               "b_spec_path": os.path.join(OUTPUT, obs_id, b_spec),
                               "b_rmf_path": os.path.join(OUTPUT, obs_id, b_rmf),
                               "b_arf_path": os.path.join(OUTPUT, obs_id, b_arf),
                               "obs_id": obs_id, "instrument": inst})

        sources_cmds.append(np.array(cmds))
        sources_paths.append(np.array(final_paths))
        # This contains any other information that will be needed to instantiate the class
        #  once the SAS cmd has run
        sources_extras.append(np.array(extra_info))
        sources_types.append(np.full(sources_cmds[-1].shape, fill_value="spectrum"))

        spec_prep.update(1)
    spec_prep.close()

    return sources_cmds, stack, execute, num_cores, sources_types, sources_paths, sources_extras, disable_progress


def evselect_annular_spectrum():
    raise NotImplementedError("Haven't quite got around to doing this bit yet")




