#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 18/01/2024, 16:02. Copyright (c) The Contributors

import os
from subprocess import Popen, PIPE
from typing import Tuple, Union

import numpy as np
from astropy.units import Quantity, UnitBase, deg
from regions import EllipseSkyRegion

from ..products import BaseProduct, Image, ExpMap, Spectrum, PSFGrid
from ..products.lightcurve import LightCurve
from ..sources import BaseSource
from ..utils import OUTPUT


def execute_cmd(cmd: str, p_type: str, p_path: list, extra_info: dict, src: str) -> Tuple[BaseProduct, str]:
    """
    This function is called for the local compute option, and runs the passed command in a Popen shell.
    It then creates an appropriate product object, and passes it back to the callback function of the Pool
    it was called from.

    :param str cmd: Command to be executed on the command line.
    :param str p_type: The product type that will be produced by this command.
    :param str p_path: The final output path of the product.
    :param dict extra_info: Any extra information required to define the product object.
    :param str src: A string representation of the source object that this product is associated with.
    :return: The product object, and the string representation of the associated source object.
    :rtype: Tuple[BaseProduct, str]
    """
    out, err = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE).communicate()
    out = out.decode("UTF-8", errors='ignore')
    err = err.decode("UTF-8", errors='ignore')

    # This part for defining an image object used to make sure that the src wasn't a NullSource, as defining product
    #  objects is wasteful considering the purpose of a NullSource, but generating exposure maps requires a
    #  pre-existing image
    if p_type == "image":
        # Maybe let the user decide not to raise errors detected in stderr
        prod = Image(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                     extra_info["lo_en"], extra_info["hi_en"], telescope=extra_info["telescope"])
        if "psf_corr" in extra_info and extra_info["psf_corr"]:
            prod.psf_corrected = True
            prod.psf_bins = extra_info["psf_bins"]
            prod.psf_model = extra_info["psf_model"]
            prod.psf_iterations = extra_info["psf_iter"]
            prod.psf_algorithm = extra_info["psf_algo"]
    elif p_type == "expmap":
        prod = ExpMap(p_path[0], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                      extra_info["lo_en"], extra_info["hi_en"], telescope=extra_info["telescope"])
    elif p_type == "ccf" and "NullSource" not in src:
        # ccf files may not be destined to spend life as product objects, but that doesn't mean
        # I can't take momentarily advantage of the error parsing I built into the product classes
        prod = BaseProduct(p_path[0], "", "", "", out, err, cmd)
    elif (p_type == "spectrum" or p_type == "annular spectrum set components") and "NullSource" not in src:
        prod = Spectrum(p_path[0], extra_info["rmf_path"], extra_info["arf_path"], extra_info["b_spec_path"],
                        extra_info['central_coord'], extra_info["inner_radius"], extra_info["outer_radius"],
                        extra_info["obs_id"], extra_info["instrument"], extra_info["grouped"], extra_info["min_counts"],
                        extra_info["min_sn"], extra_info["over_sample"], out, err, cmd, extra_info["from_region"],
                        extra_info["b_rmf_path"], extra_info["b_arf_path"], telescope=extra_info["telescope"])
    elif p_type == "psf" and "NullSource" not in src:
        prod = PSFGrid(extra_info["files"], extra_info["chunks_per_side"], extra_info["model"],
                       extra_info["x_bounds"], extra_info["y_bounds"], extra_info["obs_id"],
                       extra_info["instrument"], out, err, cmd)
    elif p_type == 'light curve' and "NullSource" not in src:
        prod = LightCurve(p_path[0],  extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                          extra_info['central_coord'], extra_info["inner_radius"], extra_info["outer_radius"],
                          extra_info["lo_en"], extra_info["hi_en"], extra_info['time_bin'], extra_info['pattern'],
                          extra_info["from_region"], telescope=extra_info['telescope'])
    elif p_type == "cross arfs":
        prod = BaseProduct(p_path[0], extra_info['obs_id'], extra_info['inst'], out, err, cmd, extra_info,
                           telescope=extra_info["telescope"])
    elif "NullSource" in src:
        prod = None
    else:
        raise NotImplementedError("Not implemented yet")

    # An extra step is required for annular spectrum set components
    if p_type == "annular spectrum set components":
        prod.annulus_ident = extra_info["ann_ident"]
        prod.set_ident = extra_info["set_ident"]

    return prod, src


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


def get_annular_esass_region(source: BaseSource, inner_radius: Quantity, outer_radius: Quantity, obs_id: str,
                             output_unit: Union[UnitBase, str] = deg, rot_angle: Quantity = Quantity(0, 'deg'),
                             interloper_regions: np.ndarray = None, central_coord: Quantity = None,
                             bkg_reg: bool = False, rand_ident: int = None) -> str:
    """
    A method to generate an eSASS region string for an arbitrary circular or elliptical annular region, with
    interloper sources removed.

    :param BaseSource source: The source object for which we wish to generate an eSASS-compatible annular region
        string/region file.
    :param Quantity inner_radius: The inner radius/radii of the region you wish to generate in SAS, if the
        quantity has multiple elements then an elliptical region will be generated, with the first element
        being the inner radius on the semi-major axis, and the second on the semi-minor axis.
    :param Quantity outer_radius: The inner outer_radius/radii of the region you wish to generate in SAS, if the
        quantity has multiple elements then an elliptical region will be generated, with the first element
        being the outer radius on the semi-major axis, and the second on the semi-minor axis.
    :param str obs_id: The ObsID of the observation you wish to generate the SAS region for.
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
    :return: Either a string representation of the requested region (if no contaminating sources are being
        removed), or a path to a region file that can be used with eSASS that specifies the source and
        contaminating regions.
    :rtype: str
    """
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
        interloper_regions = source.regions_within_radii(inner_radius, outer_radius, "erosita", central_coord)
    elif interloper_regions is None and not inner_radius.isscalar:
        interloper_regions = source.regions_within_radii(min(inner_radius), max(outer_radius), "erosita", central_coord)

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
        # Multiple regions must be passed to eSASS via an ASCII file, so I will write this here
        reg_file_path = OUTPUT + 'erosita/' + obs_id + '/temp_regs_{i}'.format(i=rand_ident)
        reg_str = esass_source_area.replace(" ", "_")  # replacing spaces with underscores for file naming purposes
        reg_str = reg_str.replace(".", "-")  # replacing any dots with dashes

        if bkg_reg: 
            # adding backround prefix for background region files
            reg_str = 'BACK_' + reg_str
        
        reg_file_name = f"{reg_str}.reg"

        # Making a temporary directory to write files into.
        #  Extra argument means no error is raised if directories already exist
        os.makedirs(reg_file_path, exist_ok=True)

        # Making the file
        # TODO Decide whether FK5 or ICRS is more appropriate here
        with open(reg_file_path + '/' + reg_file_name, 'w') as file:
            file.write('fk5; ' + esass_source_area + "\n" + "\n".join(esass_interloper))
        
        final_src = reg_file_path + '/' + reg_file_name

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

    elif telescope == 'erosita':
        # TODO Add a pattern checker when I actually understand what patterns can be for eROSITA
        patt_file_name = str(pattern)
    else:
        raise NotImplementedError("Support for the {t} telescope has not yet been added to this "
                                  "function.".format(t=telescope))

    return pattern, patt_file_name
