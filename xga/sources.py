#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 29/04/2020, 21:44. Copyright (c) David J Turner
import os
import warnings
from abc import ABC
from itertools import product
from typing import Tuple

import numpy as np
from astropy.cosmology import Planck15

from xga import xga_conf
from xga.exceptions import NotAssociatedError, UnknownProductTypeError
from xga.sourcetools import simple_xmm_match, nhlookup
from xga.utils import ENERGY_BOUND_PRODUCTS, ALLOWED_PRODUCTS


class BaseSource:
    def __init__(self, ra, dec, redshift=None, cosmology=Planck15):
        self.ra_dec = np.array([ra, dec])
        # Only want ObsIDs, not pointing coordinates as well
        # Don't know if I'll always use the simple method
        self.obs = simple_xmm_match(ra, dec)["ObsID"].values
        # Check in a box of half-side 5 arcminutes, should give an idea of which are on-axis
        on_axis_match = simple_xmm_match(ra, dec, 5)["ObsID"].values
        self.onaxis = np.isin(self.obs, on_axis_match)
        # nhlookup returns average and weighted average values, so just take the first
        self.nH = nhlookup(ra, dec)[0]
        self.redshift = redshift
        self._products = self.initial_products()
        if redshift is not None:
            self.lum_dist = cosmology.luminosity_distance(self.redshift)

    def initial_products(self) -> dict:
        """
        Assembles the initial dictionary structure of existing XMM data products associated with this source.
        :return: A dictionary structure detailing the data products available at initialisation
        :rtype: dict
        """
        def gen_file_names(en_lims: tuple) -> Tuple[str, dict]:
            """
            This nested function takes pairs of energy limits defined in the config file and runs
            through the XMM products (also defined in the config file), filling in the energy limits and
            checking if the file paths exist. Those that do exist are returned in a dictionary.
            :param tuple en_lims: A tuple containing a lower and upper energy limit to generate file names for,
            the first entry should be the lower limit, the second the upper limit.
            :return: A dictionary key based on the energy limits for the file paths to be stored under, and the
            dictionary of file paths.
            :rtype: tuple[str, dict]
            """
            not_these = ["root_xmm_dir", "lo_en", "hi_en", "region_file", evt_key]
            files = {k.split('_')[1]: v.format(lo_en=en_lims[0], hi_en=en_lims[1], obs_id=obs_id)
                     for k, v in xga_conf["XMM_FILES"].items() if k not in not_these and inst in k}

            existing_files = {key: file for key, file in files.items() if os.path.exists(file)}
            return "{l}-{u}".format(l=float(en_lims[0]), u=float(en_lims[1])), existing_files

        # This dictionary structure will contain paths to all available data products associated with this
        # source instance, both pre-generated and made with XGA.
        obs_dict = {obs: {} for obs in self.obs}

        # Use itertools to create iterable and avoid messy nested for loop
        # product makes iterable of tuples, with all combinations of the events files and ObsIDs
        for oi in product(obs_dict, ["pn", "mos1", "mos2"]):
            # Produces a list of the combinations of upper and lower energy bounds from the config file.
            en_comb = zip(xga_conf["XMM_FILES"]["lo_en"], xga_conf["XMM_FILES"]["hi_en"])

            # This is purely to make the code easier to read
            obs_id = oi[0]
            inst = oi[1]
            evt_key = "clean_{}_evts".format(inst)
            evt_file = xga_conf["XMM_FILES"][evt_key].format(obs_id=obs_id)
            reg_file = xga_conf["XMM_FILES"]["region_file"].format(obs_id=obs_id)

            if os.path.exists(evt_file):
                # An instrument subsection of an observation will ONLY be populated if the events file exists
                # Otherwise nothing can be done with it.
                obs_dict[obs_id][inst] = {"events": evt_file}
                # Dictionary updated with derived product names
                obs_dict[obs_id][inst].update({gen_return[0]: gen_return[1] for gen_return
                                               in map(gen_file_names, en_comb)})
                if os.path.exists(reg_file):
                    # Dictionary updated with path to region file, if it exists
                    obs_dict[obs_id][inst]["regions"] = reg_file

        return obs_dict

    def _update_products(self, obs_id: str, inst: str, p_type: str, p_path: str,
                         lo: float = None, hi: float = None):
        """
        Setter method for the products attribute of source objects. Cannot delete existing products,
        but will overwrite existing products with a warning. Raises errors if the ObsID is not associated
        with this source or the instrument is not associated with the ObsID.
        :param str obs_id: A valid XMM ObsID.
        :param str inst: XMM Instrument identifier. Allowed values are pn, mos1, or mos2.
        :param str p_type: Product type identifier. e.g. image or expmap.
        :param str p_path: Product file path.
        :param float lo: The lower energy bound used to create this product. This only needs to
        be supplied for image based products such as images or exposure maps.
        :param float hi: The upper energy bound used to create this product. This only needs to
        be supplied for image based products images or exposure maps.
        """
        # Allowed products are defined in the utils code, lay down the law about what types of products
        # are supported by XGA
        if p_type not in ALLOWED_PRODUCTS:
            raise UnknownProductTypeError()
        elif lo is not None and hi is not None:
            en_key = "{l}-{u}".format(l=float(lo), u=float(hi))
        elif lo is None or hi is None and p_type in ENERGY_BOUND_PRODUCTS:
            raise TypeError("Energy limits cannot be None when writing an energy bound product.")
        else:
            en_key = None

        # Just to be sure
        inst = inst.lower()

        # Don't want to point a source object at a file that doesn't actually exist.
        if not os.path.exists(p_path):
            raise FileNotFoundError("The path to the XMM product does not exist.")

        # Double check that something is trying to add spectra from another source to the current one.
        if obs_id not in self._products:
            raise NotAssociatedError("{o} is not associated with this X-ray source.".format(o=obs_id))
        elif inst not in self._products[obs_id]:
            raise NotAssociatedError("{i} is not associated with XMM observation {o}".format(i=inst, o=obs_id))
        elif en_key is not None and en_key in self._products[obs_id][inst] \
                and p_type in self._products[obs_id][inst][en_key]:
            warnings.warn("You are replacing an existing product associated with this X-ray source.")

        if p_type in ENERGY_BOUND_PRODUCTS:
            self._products[obs_id][inst][en_key][p_type] = p_path
        else:
            # TODO This implementation will have to change when I figure out how to implement
            #  writing groups of spectra.
            self._products[obs_id][inst][p_type] = p_path

    def info(self):

        raise NotImplementedError("Haven't written the summary function yet.")


class ExtendedSource(BaseSource, ABC):
    def __init__(self, ra, dec, redshift=None):
        super().__init__(ra, dec, redshift)


class PointSource(BaseSource):
    def __init__(self, ra, dec, redshift=None):
        super().__init__(ra, dec, redshift)


class GalaxyCluster(ExtendedSource):
    def __init__(self, ra, dec, redshift=None):
        super().__init__(ra, dec, redshift)





