from subprocess import Popen, PIPE
from typing import Tuple

from ..products import BaseProduct, Image, ExpMap, Spectrum, PSFGrid

#TODO need to see if the current version of region_setup needs editing for eROSITA implementation
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