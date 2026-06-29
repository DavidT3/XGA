#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 6/23/26, 1:59 PM. Copyright (c) The Contributors.

import os
import sys
from functools import wraps
from multiprocessing.dummy import Pool
from subprocess import Popen, PIPE
from typing import Tuple, Union, List, Callable
from warnings import warn

import numpy as np
from astropy.units import Quantity, UnitBase, deg
from regions import EllipseSkyRegion
from tqdm import tqdm

from xga.exceptions import XGADeveloperError, InvalidTelescopeError, ProductGenerationError
from ..products import BaseProduct, Image, ExpMap, Spectrum, PSFGrid, EventList, AnnularSpectra
from ..products.lightcurve import LightCurve
from ..samples.base import BaseSample
from ..sources import BaseSource
from ..sources.base import NullSource


def mission_software_call(mission_name: str, avail_check: Callable[[], None]):
    """
    This is a generic decorator for functions that produce command strings for mission-specific software (like SAS,
    eSASS, or CIAO). It handles the setup of multiprocessing pools, the execution of commands, and the assignment
    of resulting product objects to the relevant source objects.

    :param str mission_name: The name of the mission/software being called (e.g., 'xmm', 'erosita', 'chandra').
    :param Callable avail_check: A function that checks if the necessary software and environment variables are
        available. Should raise an appropriate error if not.
    """
    # Using the standard XGA pretty names for the progress bar
    from xga import PRETTY_TELESCOPE_NAMES
    disp_name = PRETTY_TELESCOPE_NAMES[mission_name]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Checking software is installed and available on the system
            avail_check()

            # The first argument of all of these mission software functions will be the source object (or a list of),
            # so rather than return them from the function I'll just access them like this.
            if isinstance(args[0], (BaseSource, NullSource)):
                sources = [args[0]]
            elif isinstance(args[0], (BaseSample, list)):
                sources = args[0]
            else:
                raise TypeError("Please pass a source, NullSource, or sample object.")

            # This is the output from whatever function this is a decorator for
            cmd_list, to_stack, to_execute, cores, p_type, paths, extra_info, disable = func(*args, **kwargs)

            src_lookup = {}
            all_run = []  # Combined command list for all sources
            all_type = []  # Combined expected type list for all sources
            all_path = []  # Combined expected path list for all sources
            all_extras = []  # Combined extra information list for all sources
            source_rep = []  # For repr calls of each source object, needed for assigning products to sources
            for ind in range(len(cmd_list)):
                source = sources[ind]
                if len(cmd_list[ind]) > 0:
                    src_lookup[repr(source)] = ind
                    # If there are commands to add to a source queue, then do it
                    source.update_queue(cmd_list[ind], p_type[ind], paths[ind], extra_info[ind], to_stack)

                # If we do want to execute the commands this time round, we read them out for all sources
                # and add them to these master lists
                if to_execute:
                    to_run, expected_type, expected_path, extras = source.get_queue()
                    all_run += to_run
                    all_type += expected_type
                    all_path += expected_path
                    all_extras += extras
                    source_rep += [repr(source)] * len(to_run)

            # This is what the returned products get stored in before they're assigned to sources
            results = {s: [] for s in src_lookup}
            # Any errors raised shouldn't be software-specific, as they are stored within the product object.
            raised_errors = []
            # Making sure something is defined for this variable
            prod_type_str = ""
            if to_execute and len(all_run) > 0:
                # Will run the commands locally in a pool
                # This flattening logic is necessary for CIAO but safe for all
                prod_type_str = ", ".join(set(np.array(all_type).flatten().tolist()))
                with tqdm(total=len(all_run), desc="Generating {} products of type(s) ".format(disp_name) +
                                                   prod_type_str, disable=disable) as gen, Pool(cores) as pool:
                    def callback(results_in: Tuple[Union[BaseProduct, List[BaseProduct]], str]):
                        """
                        Callback function for the apply_async pool method, gets called when a task finishes
                        and something is returned.
                        :param Tuple[Union[BaseProduct, List[BaseProduct]], str] results_in: Results of the command call.
                        """
                        nonlocal gen  # The progress bar will need updating
                        nonlocal results  # The dictionary the command call results are added to
                        if results_in[0] is None:
                            gen.update(1)
                            return
                        else:
                            prod_obj, rel_src = results_in
                            if isinstance(prod_obj, list):
                                results[rel_src] += prod_obj
                            else:
                                results[rel_src].append(prod_obj)
                            gen.update(1)

                    def err_callback(err):
                        """
                        The callback function for errors that occur inside a task running in the pool.
                        :param err: An error that occurred inside a task.
                        """
                        nonlocal raised_errors
                        nonlocal gen
                        nonlocal src_lookup
                        nonlocal sources

                        if err is not None:
                            # We used a memory address laden source name representation when we adjusted the error
                            # message in execute_cmd, so we'll replace it with an actual name here
                            if len(err.args) == 1 and isinstance(err.args[0], str) and \
                                    ' is the associated source' in err.args[0]:
                                err_src_rep = err.args[0].split(' is the associated source')[0].split('- ')[-1].strip()
                                act_src_name = sources[src_lookup[err_src_rep]].name
                                err.args = (err.args[0].replace(err_src_rep, act_src_name),)
                            elif len(err.args) > 1 and isinstance(err.args[1], str) and \
                                    ' is the associated source' in err.args[1]:
                                err_src_rep = err.args[1].split(' is the associated source')[0].split('- ')[-1].strip()
                                act_src_name = sources[src_lookup[err_src_rep]].name
                                err.args = (err.args[0], err.args[1].replace(err_src_rep, act_src_name))

                            # Rather than throwing an error straight away I append them all to a list for later.
                            raised_errors.append(err)
                        gen.update(1)

                    for cmd_ind, cmd in enumerate(all_run):
                        # These are just the relevant entries in all these lists for the current command
                        exp_type = all_type[cmd_ind]
                        exp_path = all_path[cmd_ind]
                        ext = all_extras[cmd_ind]
                        src = source_rep[cmd_ind]

                        pool.apply_async(execute_cmd, args=(str(cmd), exp_type, exp_path, ext, src),
                                         error_callback=err_callback, callback=callback)
                    pool.close()  # No more tasks can be added to the pool
                    pool.join()  # Joins the pool, the code will only move on once the pool is empty.

            elif to_execute and len(all_run) == 0:
                # It is possible to call a wrapped function and find that the products already exist.
                pass

            # Now we assign products to source objects
            all_to_raise = []
            # This is for the special case of generating an AnnularSpectra product
            ann_spec_comps = {k: [] for k in results}
            for entry in results:
                # Made this lookup list earlier, using string representations of source objects.
                # Finds the ind of the list of sources that we should add this set of products to
                ind = src_lookup[entry]
                to_raise = []
                for product in results[entry]:
                    product: BaseProduct
                    ext_info = "- {s} is the associated source, the specific data used is " \
                               "{t} {o}-{i}.".format(s=sources[ind].name, t=product.telescope,
                                                     o=product.obs_id, i=product.instrument)
                    if len(product.gen_errors) == 1:
                        to_raise.append(ProductGenerationError(product.gen_errors[0] + ext_info))
                    elif len(product.gen_errors) > 1:
                        errs = [ProductGenerationError(e + ext_info) for e in product.gen_errors]
                        to_raise += errs

                    if len(product.errors) == 1:
                        to_raise.append(ProductGenerationError(product.errors[0] + "-" + ext_info))
                    elif len(product.errors) > 1:
                        errs = [ProductGenerationError(e + "-" + ext_info) for e in product.errors]
                        to_raise += errs

                    # If the product type is None we don't store it
                    if product.type is not None and product.usable and prod_type_str != "annular spectrum set components":
                        # For each product produced for this source, we add it to the storage hierarchy
                        sources[ind].update_products(product)
                    elif product.type is not None and product.usable and prod_type_str == "annular spectrum set components":
                        # Really we're just re-creating the results dictionary here, but I want these products
                        # to go through the error checking stuff like everything else does
                        ann_spec_comps[entry].append(product)
                    # In case they are components of an annular spectrum but they are either none or not usable
                    elif prod_type_str == "annular spectrum set components":
                        warn("An annular spectrum component ({a}) for {t} {o}{i} has not been generated properly (not "
                             "usable reason - {nur}). The std_err entry is:\n\n {se}\n\n The std_out entry is:\n\n "
                             "{so}".format(a=product.storage_key, t=product.telescope, o=product.obs_id,
                                           i=product.instrument, nur=product.not_usable_reasons,
                                           se=product.unprocessed_stderr, so=product.unprocessed_stdout), stacklevel=2)
                    # Here the generated product was a cross-arf, and needs to be added to the right annular spectrum
                    # object that already exists in our source
                    elif prod_type_str == "cross arfs":
                        ei = product._extra_info
                        ann_spec = sources[ind].get_annular_spectra(set_id=ei['ann_spec_set_id'])
                        ann_spec.add_cross_arf(product, ei['obs_id'], ei['inst'], ei['src_ann_id'], ei['cross_ann_id'],
                                               ei['ann_spec_set_id'])

                if len(to_raise) != 0:
                    all_to_raise.append(to_raise)

            # We raise the errors BEFORE setting up AnnularSpectra instances, as if a product destined to be in an
            # AnnularSpectra instance failed to generate properly then the init of that class could fail and we'd
            # never see the generation errors.
            if len(raised_errors) != 0:
                raise Exception(raised_errors)

            if len(all_to_raise) != 0:
                raise ProductGenerationError(all_to_raise)

            if prod_type_str == "annular spectrum set components":
                for entry in ann_spec_comps:
                    if len(ann_spec_comps[entry]) == 0:
                        warn(f"Entry {entry} - no annular spectrum components were successfully "
                             f"generated - contact the developers if you see this warning.", stacklevel=2)
                        continue

                    # So now we pass the list of spectra to a AnnularSpectra definition
                    ann_spec = AnnularSpectra(ann_spec_comps[entry])
                    ind = src_lookup[entry]
                    if sources[ind].redshift is not None:
                        # If we know the redshift we will add the radii to the annular spectra in proper distance units
                        ann_spec.proper_radii = sources[ind].convert_radius(ann_spec.radii, 'kpc')
                    # And adding our exciting new set of annular spectra into the storage structure
                    sources[ind].update_products(ann_spec)

            # If only one source was passed, turn it back into a source object rather than a list.
            if len(sources) == 1:
                sources = sources[0]
            return sources
        return wrapper
    return decorator


def execute_cmd(cmd: str, p_type: Union[str, List[str]], p_path: list, extra_info: dict,
                src: str) -> Tuple[Union[BaseProduct, List[BaseProduct]], str]:
    """
    This function is called for the local compute option, and runs the passed command in a Popen shell.
    It then creates an appropriate product object, and passes it back to the callback function of the Pool
    it was called from.

    :param str cmd: Command to be executed on the command line.
    :param str p_type: The product type(s) that will be produced by this command - this can be a string, or a list
        of strings in the case where an XGA interface-with-telescope-software function will produce multiple
        different products.
    :param list p_path: The final output path(s) of the product.
    :param dict extra_info: Any extra information required to define the product object.
    :param str src: A string representation of the source object that this product is associated with.
    :return: The product object(s), and the string representation of the associated source object.
    :rtype: Tuple[Union[BaseProduct, List[BaseProduct]], str]
    """

    # This chunk is a fix for problems with eSASS (eROSITA package) finding the correct libraries on Apple ARM based
    #  systems, and just creates a new environment variable so it can locate them, if necessary
    sys_env = os.environ.copy()
    if sys.platform == 'darwin':
        if "LD_LIBRARY_PATH" in sys_env:
            cmd = f"export LD_LIBRARY_PATH={sys_env['LD_LIBRARY_PATH']} && {cmd}"
        if "DYLD_LIBRARY_PATH" in sys_env:
            cmd = f"export DYLD_LIBRARY_PATH={sys_env['DYLD_LIBRARY_PATH']} && {cmd}"

    # This runs the passed command - it captures the stdout and stderr as well
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    # Captured out/err are byte type, will decode that into str for easier use
    out = out.decode("UTF-8", errors='ignore')
    err = err.decode("UTF-8", errors='ignore')

    # We will assume that the first entry in the product path argument is the
    #  primary output file, and will name the log files after it
    log_stem = p_path[0][:p_path[0].rfind('.')]
    # One file for out and one for error
    std_out_file = log_stem + "_stdout.log"
    std_err_file = log_stem + "_stderr.log"

    # Write the standard out to a log file
    with open(std_out_file, "w") as std_outo:
        std_outo.write(cmd + "\n\n" + out)
    # If there are standard error entries, write them to a file as well
    if len(err) > 0:
        with open(std_err_file, "w") as std_erro:
            std_erro.write(cmd + "\n\n" + err)

    # Trying to make sure that any passed arrays get smoothed out into the list type we want
    if type(p_path) == np.ndarray:
        p_path = list(p_path)

    if type(p_type) == np.ndarray:
        p_type = list(p_type)

    # This is becoming a bit of a hodge podge of fixes to avoid making absolutely sure the inputs are right,
    #  but it should work fine and its likely we rewrite the command management system in sources which
    #  is causing some of these woes (at some point) so I don't feel too bad about it
    # This is because the way the XMM call is set up, the passed paths are also in a list of one
    if isinstance(p_path, list) and len(p_path) == 1:
        # In some instances the type(p_path[0]) can be a numpy string, so Im making it be a normal 
        # string so that its type can match type(p_type)
        p_path = str(p_path[0])
    
    # In some instances the type(p_type) can be a numpy string, so Im making it be a normal 
    # string so that its type can match type(p_path)
    if isinstance(p_type, (str, np.str_)):
        p_type = str(p_type)

    # Catch any mistakes that will be easy to make in developing new interface functions with
    #  backend telescope software
    if type(p_type) != type(p_path):
        raise XGADeveloperError("Both the p_type and p_path arguments must be of the same type (both string or "
                                "both list).")
    elif isinstance(p_type, str):
        p_type = [p_type]
        p_path = [p_path]
    # Check again for mistakes made during development, where there are multiple product paths, but they don't each
    #  have a product type as is required
    if len(p_type) != len(p_path):
        raise XGADeveloperError("Product generation products that produce multiple products to be loaded into "
                                "different product classes must have one product type entry for each.")

    # We'll now be iterating through the product paths and their matching types
    prods = []
    for p_ind, cur_p_path in enumerate(p_path):
        cur_p_type = p_type[p_ind]

        # This part for defining an image object used to make sure that the src wasn't a NullSource, as defining product
        #  objects is wasteful considering the purpose of a NullSource, but generating exposure maps requires a
        #  pre-existing image
        if cur_p_type == "image":
            # Maybe let the user decide not to raise errors detected in stderr
            prod = Image(cur_p_path, extra_info["obs_id"], extra_info["instrument"], out, err, cmd, extra_info["lo_en"],
                         extra_info["hi_en"], telescope=extra_info["telescope"])
            if "psf_corr" in extra_info and extra_info["psf_corr"]:
                prod.psf_corrected = True
                prod.psf_bins = extra_info["psf_bins"]
                prod.psf_model = extra_info["psf_model"]
                prod.psf_iterations = extra_info["psf_iter"]
                prod.psf_algorithm = extra_info["psf_algo"]
        elif cur_p_type == "expmap":
            prod = ExpMap(cur_p_path, extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                          extra_info["lo_en"], extra_info["hi_en"], telescope=extra_info["telescope"])
        elif cur_p_type == "ccf" and "NullSource" not in src:
            # ccf files may not be destined to spend life as product objects, but that doesn't mean
            # I can't take momentarily advantage of the error parsing I built into the product classes
            prod = BaseProduct(cur_p_path, "", "", out, err, cmd, telescope='xmm')
        elif (cur_p_type == "spectrum" or cur_p_type == "annular spectrum set components") and "NullSource" not in src:
            prod = Spectrum(cur_p_path, extra_info["rmf_path"], extra_info["arf_path"], extra_info["b_spec_path"],
                            extra_info['central_coord'], extra_info["inner_radius"], extra_info["outer_radius"],
                            extra_info["obs_id"], extra_info["instrument"], extra_info["grouped"],
                            extra_info["min_counts"], extra_info["min_sn"], extra_info["over_sample"], out, err, cmd,
                            extra_info["from_region"], extra_info["b_rmf_path"], extra_info["b_arf_path"],
                            telescope=extra_info["telescope"])
        elif cur_p_type == "psf" and "NullSource" not in src:
            prod = PSFGrid(extra_info["files"], extra_info["chunks_per_side"], extra_info["model"],
                           extra_info["x_bounds"], extra_info["y_bounds"], extra_info["obs_id"],
                           extra_info["instrument"], out, err, cmd, telescope=extra_info['telescope'])
        elif cur_p_type == 'light curve' and "NullSource" not in src:
            prod = LightCurve(cur_p_path, extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                              extra_info['central_coord'], extra_info["inner_radius"], extra_info["outer_radius"],
                              extra_info["lo_en"], extra_info["hi_en"], extra_info['time_bin'], extra_info['pattern'],
                              extra_info["from_region"], telescope=extra_info['telescope'])
        elif cur_p_type == "cross arfs":
            prod = BaseProduct(cur_p_path, extra_info['obs_id'], extra_info['inst'], out, err, cmd, extra_info,
                               telescope=extra_info["telescope"])
        elif cur_p_type == 'events' or cur_p_type == 'combined events':
            prod = EventList(cur_p_path, extra_info['obs_id'], extra_info['instrument'], out, err, cmd,
                             telescope=extra_info['telescope'], obs_ids=extra_info['obs_ids'])
        elif cur_p_type == 'ratemap':
            # The count-rate map files produced by Chandra software (for instance) cannot yet be read into XGA
            #  ratemap class instances - though we will include this at some point
            prod = None
        elif "NullSource" in src:
            # Even if we're not storing the product instances, the prod variable needs
            #  to be set, otherwise there will be an error
            prod = None
        else:
            raise NotImplementedError("Not implemented yet")

        # An extra step is required for annular spectrum set components
        if cur_p_type == "annular spectrum set components":
            prod.annulus_ident = extra_info["ann_ident"]
            prod.set_ident = extra_info["set_ident"]

        # Put the current prod in the prods list
        prods.append(prod)

    # This should make it easier to keep this compatible for the telescopes without functions that generate multiple
    #  different products at once without changing how their call decorators work right now
    if len(prods) == 1:
        prods = prods[0]

    return prods, src


def _interloper_esass_string(reg: EllipseSkyRegion) -> str:
    """
    Converts ellipse sky regions into eSASS region strings for use in eSASS tasks.

    :param EllipseSkyRegion reg: The interloper region to generate an eSASS string for
    :return: The eSASS string region for this interloper
    :rtype: str
    """

    w = reg.width.to('deg').value
    h = reg.height.to('deg').value
    cen = Quantity([reg.center.ra.value, reg.center.dec.value], 'deg')

    if w == h:
        shape_str = "-circle {cx} {cy} {r}d"
        shape_str = shape_str.format(cx=cen[0].value, cy=cen[1].value, r=(h / 2))
    else:
        # The rotation angle from the region object is in degrees already
        shape_str = "-ellipse {cx} {cy} {w}d {h}d {rot}"
        shape_str = shape_str.format(cx=cen[0].value, cy=cen[1].value, w=w,
                                     h=h, rot=reg.angle.value)

    return shape_str


def get_annular_esass_region(source: BaseSource, inner_radius: Quantity, outer_radius: Quantity, obs_id: str, telescope: str,
                             output_unit: Union[UnitBase, str] = deg, rot_angle: Quantity = Quantity(0, 'deg'),
                             interloper_regions: np.ndarray = None, central_coord: Quantity = None,
                             bkg_reg: bool = False, rand_ident: int = None, out_root_path: str = None) -> str:
    """
    A method to generate an eSASS region string for an arbitrary circular or elliptical annular region, with
    interloper sources removed.

    :param BaseSource source: The source object for which we wish to generate an eSASS-compatible annular region
        string/region file.
    :param Quantity inner_radius: The inner radius/radii of the region you wish to generate in eSASS, if the
        quantity has multiple elements then an elliptical region will be generated, with the first element
        being the inner radius on the semi-major axis, and the second on the semi-minor axis.
    :param Quantity outer_radius: The outer radius/radii of the region you wish to generate in eSASS, if the
        quantity has multiple elements then an elliptical region will be generated, with the first element
        being the outer radius on the semi-major axis, and the second on the semi-minor axis.
    :param str obs_id: The ObsID of the observation you wish to generate the eSASS region for.
    :param str telescope: The particular eROSITA mission skew to create a region for; 'erosita' or 'erass'.
    :param UnitBase/str output_unit: The desired units for the region string/file to be written in.
    :param Quantity rot_angle: The rotation angle of the source region, default is zero degrees.
    :param np.ndarray interloper_regions: The interloper regions to remove from the source region,
        default is None, in which case the function will run self.regions_within_radii.
    :param Quantity central_coord: The coordinate on which to centre the source region, default is
        None in which case the function will use the default_coord of the source object.
    :param bool bkg_reg: Specifies whether the region string/file to be generated is for a background annulus. If
        True, and contaminating regions are being removed (i.e. a file needs to be written out), then a back_ prefix
        will be added to the file names. Default is False.
    :param int rand_ident: A random identifier to be inserted in the 'temp_regs' directory name, if temporary
        region files need to be written out.
    :param str out_root_path: The root path within which we can create a temporary directory of region files if
        they need to be written to disk.
    :return: Either a string representation of the requested region (if no contaminating sources are being
        removed), or a path to a region file that can be used with eSASS that specifies the source and
        contaminating regions.
    :rtype: str
    """
    if telescope not in ['erosita', 'erass']:
        raise InvalidTelescopeError("Pass either 'erosita' or 'erass' to this function's telescope argument.")

    if central_coord is None:
        central_coord = source.default_coord

    inner_radius = source.convert_radius(inner_radius, 'deg')
    outer_radius = source.convert_radius(outer_radius, 'deg')

    # Then we can check to make sure that the outer radius is larger than the inner radius
    if inner_radius.isscalar and inner_radius >= outer_radius:
        raise ValueError("An eSASS circular region for {s} cannot have an inner_radius larger than or equal to its "
                         "outer_radius".format(s=source.name))
    elif not inner_radius.isscalar and (inner_radius[0] >= outer_radius[0] or inner_radius[1] >= outer_radius[1]):
        raise ValueError("An eSASS elliptical region for {s} cannot have inner radii larger than or equal to its "
                         "outer radii".format(s=source.name))
    
    if output_unit != deg:
        raise NotImplementedError("Only degree coordinates are currently supported "
                                  " for generating eSASS region strings.")
            
    # And just to make sure the central coordinates are in degrees
    # TODO do a conversion instead of raising an error
    if central_coord[0].unit != deg and central_coord[1].unit != deg:
        raise ValueError("Need to convert central coordinates into degrees.")
    
    # If the user doesn't pass any regions, then we have to find them ourselves. I decided to allow this
    #  so that within_radii can just be called once externally for a set of ObsID-instrument combinations,
    #  like in evselect_spectrum for instance.
    if interloper_regions is None and inner_radius.isscalar:
        interloper_regions = source.regions_within_radii(inner_radius, outer_radius, telescope, central_coord)
    elif interloper_regions is None and not inner_radius.isscalar:
        interloper_regions = source.regions_within_radii(min(inner_radius), max(outer_radius), telescope, central_coord)

    # So now we convert our interloper regions into their eSASS equivalents
    esass_interloper = [_interloper_esass_string(i) for i in interloper_regions]
    # TODO I have assumed that the eSASS versions of the regions are in the correct format

    if inner_radius.isscalar and inner_radius.value != 0:
        esass_source_area = "annulus {cx} {cy} {ri}d {ro}d"
        esass_source_area = esass_source_area.format(cx=central_coord[0].value,
                                                     cy=central_coord[1].value,
                                                     ri=inner_radius.value, ro=outer_radius.value)

    elif inner_radius.isscalar and inner_radius.value == 0:
        esass_source_area = "circle {cx} {cy} {r}d"
        esass_source_area = esass_source_area.format(cx=central_coord[0].value,
                                                     cy=central_coord[1].value, r=outer_radius.value)
        
    elif not inner_radius.isscalar and inner_radius[0].value != 0:
        esass_source_area = "ellipse {cx} {cy} {wi} {hi} {wo} {ho} {rot}"
        esass_source_area = esass_source_area.format(cx=central_coord[0].value,
                                                     cy=central_coord[1].value,
                                                     wi=inner_radius[0].value,
                                                     hi=inner_radius[1].value,
                                                     wo=outer_radius[0].value,
                                                     ho=outer_radius[1].value,
                                                     rot=rot_angle.to('deg').value)

    elif not inner_radius.isscalar and inner_radius[0].value == 0:
        esass_source_area = "ellipse {cx} {cy} {w} {h} {rot}"
        esass_source_area = esass_source_area.format(cx=central_coord[0].value,
                                                     cy=central_coord[1].value,
                                                     w=outer_radius[0].value,
                                                     h=outer_radius[1].value,
                                                     rot=rot_angle.to('deg').value)

    # Combining the source region with regions we need to cut out
    if len(esass_interloper) == 0:
        final_src = esass_source_area
    else:
        # Multiple regions must be passed to eSASS via an ASCII
        #  file, so we will need somewhere to write them
        reg_file_path = os.path.join(out_root_path, f'/temp_regs_{rand_ident}')

        reg_str = esass_source_area.replace(" ", "_")  # replacing spaces with underscores for file naming purposes
        reg_str = reg_str.replace(".", "-")  # replacing any dots with dashes

        if bkg_reg: 
            # adding backround prefix for background region files
            reg_str = 'BACK_' + reg_str
        
        reg_file_name = f"{reg_str}.reg"

        # Making a temporary directory to write files into.
        #  Extra argument means no error is raised if directories already exist
        os.makedirs(reg_file_path, exist_ok=True)

        final_reg_path = os.path.join(reg_file_path, reg_file_name)
        # Making the file
        # TODO Decide whether FK5 or ICRS is more appropriate here
        with open(final_reg_path, 'w') as file:
            file.write('fk5; ' + esass_source_area + "\n" + "\n".join(esass_interloper))
        
        final_src = final_reg_path

    return final_src


def check_pattern(pattern: Union[str, int], telescope: str = 'xmm') -> Tuple[str, str]:
    """
    A very simple (and not exhaustive) checker for pattern expressions.

    :param str/int pattern: The pattern selection expression to be checked.
    :param str telescope: The telescope for which we wish to check the validity of a pattern expression.
    :return: A string pattern selection expression, and a pattern representation that should be safe for naming
        files with.
    :rtype: Tuple[str, str]
    """

    if telescope == 'xmm':
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

    elif telescope == 'erosita' or telescope == 'erass':
        # TODO Add a pattern checker when I actually understand what patterns can be for eROSITA
        patt_file_name = str(pattern)
    elif telescope == 'chandra':
        # TODO Add a pattern checker when I actually understand what patterns can be for Chandra
        patt_file_name = str(pattern)
    else:
        warn("Support for the {t} telescope has not yet been added to this function.".format(t=telescope),
             stacklevel=2)
        patt_file_name = str(pattern)

    return pattern, patt_file_name
