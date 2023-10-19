from subprocess import Popen, PIPE
from typing import Tuple, Union

import numpy as np
from astropy.units import Quantity, UnitBase, deg

from ..utils import erosita_sky
from ..products import BaseProduct, Image, ExpMap, Spectrum, PSFGrid

#ASSUMPTION7 that the telescope agnostic region_setup will go here
def region_setup():
    pass

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
        # ASSUMPTION1 - tscope attribute in BaseProduct
        prod = Image(p_path[0], extra_info["telescope"], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                     extra_info["lo_en"], extra_info["hi_en"])
        if "psf_corr" in extra_info and extra_info["psf_corr"]:
            prod.psf_corrected = True
            prod.psf_bins = extra_info["psf_bins"]
            prod.psf_model = extra_info["psf_model"]
            prod.psf_iterations = extra_info["psf_iter"]
            prod.psf_algorithm = extra_info["psf_algo"]
    elif p_type == "expmap":
        # ASSUMPTION1 - tscope attribute in BaseProduct
        prod = ExpMap(p_path[0], extra_info["telescope"], extra_info["obs_id"], extra_info["instrument"], out, err, cmd,
                      extra_info["lo_en"], extra_info["hi_en"])
    elif p_type == "ccf" and "NullSource" not in src:
        # ccf files may not be destined to spend life as product objects, but that doesn't mean
        # I can't take momentarily advantage of the error parsing I built into the product classes
        # ASSUMPTION1 - tscope attribute in BaseProduct
        prod = BaseProduct(p_path[0], "", "", "", out, err, cmd)
    elif (p_type == "spectrum" or p_type == "annular spectrum set components") and "NullSource" not in src:
        # ASSUMPTION1 - tscope attribute in BaseProduct
        prod = Spectrum(p_path[0], extra_info["telescope"], extra_info["rmf_path"], extra_info["arf_path"], extra_info["b_spec_path"],
                        extra_info['central_coord'], extra_info["inner_radius"], extra_info["outer_radius"],
                        extra_info["obs_id"], extra_info["instrument"], extra_info["grouped"], extra_info["min_counts"],
                        extra_info["min_sn"], extra_info["over_sample"], out, err, cmd, extra_info["from_region"],
                        extra_info["b_rmf_path"], extra_info["b_arf_path"])
    elif p_type == "psf" and "NullSource" not in src:
        prod = PSFGrid(extra_info["files"], extra_info["chunks_per_side"], extra_info["model"],
                       extra_info["x_bounds"], extra_info["y_bounds"], extra_info["obs_id"],
                       extra_info["instrument"], out, err, cmd)
    elif p_type == "cross arfs":
        # ASSUMPTION1 - tscope attribute in BaseProduct
        prod = BaseProduct(p_path[0], extra_info["telescope"], extra_info['obs_id'], extra_info['inst'], out, err, cmd, extra_info)
    elif "NullSource" in src:
        prod = None
    else:
        raise NotImplementedError("Not implemented yet")

    # An extra step is required for annular spectrum set components
    if p_type == "annular spectrum set components":
        prod.annulus_ident = extra_info["ann_ident"]
        prod.set_ident = extra_info["set_ident"]

    return prod, src

def get_annular_esass_region(self, inner_radius: Quantity, outer_radius: Quantity, obs_id: str, inst: str,
                               output_unit: Union[UnitBase, str] = deg, rot_angle: Quantity = Quantity(0, 'deg'),
                               interloper_regions: np.ndarray = None, central_coord: Quantity = None) -> str:
    """
    A method to generate an eSASS region string for an arbitrary circular or elliptical annular region, with
    interloper sources removed.
    """
    # DAVID_QUESTION what system is this in? Need to convert to ICRS for srctool
    if central_coord is None:
        central_coord = self._default_coord

    inner_radius = self.convert_radius(inner_radius, 'deg')
    outer_radius = self.convert_radius(outer_radius, 'deg')

    # Then we can check to make sure that the outer radius is larger than the inner radius
    if inner_radius.isscalar and inner_radius >= outer_radius:
        raise ValueError("An eSASS circular region for {s} cannot have an inner_radius larger than or equal to its "
                            "outer_radius".format(s=self.name))
    elif not inner_radius.isscalar and (inner_radius[0] >= outer_radius[0] or inner_radius[1] >= outer_radius[1]):
        raise ValueError("An eSASS elliptical region for {s} cannot have inner radii larger than or equal to its "
                            "outer radii".format(s=self.name))
    
    if output_unit != deg:
        raise NotImplementedError("Only degree coordinates are currently supported "
                                  " for generating eSASS region strings.")
            
    # And just to make sure the central coordinates are in degrees
    # TODO do a conversion instead of raising an error
    if central_coord.unit != deg:
        raise ValueError("Need to convert central coordinates into degrees.")
    
    # If the user doesn't pass any regions, then we have to find them ourselves. I decided to allow this
    #  so that within_radii can just be called once externally for a set of ObsID-instrument combinations,
    #  like in evselect_spectrum for instance.
    #ASSUMPTION8 telescope agnostic version of the regions_within_radii will have telescope argument
    if interloper_regions is None and inner_radius.isscalar:
        interloper_regions = self.regions_within_radii(inner_radius, outer_radius, central_coord, telescope="erosita")
    #ASSUMPTION8 telescope agnostic version of the regions_within_radii will have telescope argument
    elif interloper_regions is None and not inner_radius.isscalar:
        interloper_regions = self.regions_within_radii(min(inner_radius), max(outer_radius), central_coord, telescope="erosita")
    



    



    pass