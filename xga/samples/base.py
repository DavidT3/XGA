#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 07/07/2021, 17:50. Copyright (c) David J Turner

from typing import Union, List, Dict
from warnings import warn

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity
from numpy import ndarray
from tqdm import tqdm

from ..exceptions import NoMatchFoundError, ModelNotAssociatedError, ParameterNotAssociatedError, \
    NoValidObservationsError
from ..exceptions import NoValidObservationsError
from ..sources.base import BaseSource
from ..sourcetools.misc import coord_to_name


class BaseSample:
    """
    The superclass for all sample classes. These store whole samples of sources, to make bulk analysis of
    interesting X-ray sources easy.
    """
    def __init__(self, ra: ndarray, dec: ndarray, redshift: ndarray = None, name: ndarray = None, cosmology=Planck15,
                 load_products: bool = True, load_fits: bool = False, no_prog_bar: bool = False):
        if len(ra) == 0:
            raise ValueError("You have passed an empty array for the RA values.")

        # There used to be a set of attributes storing the basic information (ra, dec, and redshifts) about
        #  the sources in this sample, but for subclasses its actually way more convenient for the properties
        #  to just pull the information out of the sources.
        # This is an empty list so that if no name is passed the automatically generated names can be added during
        #  declaration
        self._names = []
        self._cosmo = cosmology
        self._sources = {}
        # This stores the indexes of the sources that work and will be stored in this object, for use by things
        #  like the PointSample declaration, where names aren't strongly required but there are arguments that
        #  aren't passed to this object and stored by it.
        self._accepted_inds = []

        # A dictionary of the names of sources that could not be declared, and a basic reason why
        self._failed_sources = {}

        # Just checking that, if names are being supplied, then they are all unique
        if name is not None and len(set(name)) != len(name):
            raise ValueError("Names supplied to samples must be unique.")

        with tqdm(desc="Declaring BaseSource Sample", total=len(ra), disable=no_prog_bar) as dec_base:
            for ind, r in enumerate(ra):
                d = dec[ind]
                if name is not None:
                    n = name[ind]
                else:
                    n = None

                if redshift is not None:
                    z = redshift[ind]
                else:
                    z = None

                try:
                    temp = BaseSource(r, d, z, n, cosmology, load_products, load_fits)
                    n = temp.name
                    self._sources[n] = temp
                    self._names.append(n)
                    self._accepted_inds.append(ind)
                except (NoMatchFoundError, NoValidObservationsError):
                    if n is not None:
                        # We don't be liking spaces in source names
                        # n = n.replace(" ", "")
                        pass
                    else:
                        ra_dec = Quantity(np.array([r, d]), 'deg')
                        n = coord_to_name(ra_dec)

                    warn("Source {n} does not appear to have any XMM data, and will not be included in the "
                         "sample.".format(n=n))
                    self._failed_sources[n] = "NoMatch"
                dec_base.update(1)

        # It is possible (especially if someone is using the Sample classes as a way to check whether things have
        #  XMM data) that no sources will have been declared by this point, in which case it should fail now
        if len(self._sources) == 0:
            raise NoValidObservationsError("No sources have been declared, likely meaning that none of the sample have"
                                           " valid XMM data.")

    # These next few properties are all quantities passed in by the user on init, then used to
    #  declare source objects - as such they cannot ever be set by the user.
    @property
    def names(self) -> ndarray:
        """
        Property getter for the list of source names in this sample.

        :return: List of source names.
        :rtype: list
        """
        return np.array(self._names)

    @property
    def ra_decs(self) -> Quantity:
        """
        Property getter for the list of RA-DEC positions of the sources in this sample.

        :return: List of source RA-DEC positions as supplied at sample initialisation.
        :rtype: Quantity
        """

        return Quantity([s.ra_dec.value for s in self._sources.values()], 'deg')

    @property
    def redshifts(self) -> ndarray:
        """
        Property getter for the list of redshifts of the sources in this
        sample (if available). If no redshifts were supplied, None will be returned.

        :return: List of redshifts.
        :rtype: ndarray
        """
        return np.array([s.redshift for s in self._sources.values()])

    @property
    def nHs(self) -> Quantity:
        """
        Property getter for the list of nH values of the sources in this sample.

        :return: List of nH values.
        :rtype: Quantity
        """
        return np.Quantity([s.nH for s in self._sources.values()])

    @property
    def cosmo(self):
        """
        Property getter for the cosmology defined at initialisation of the sample. This cosmology is what
        is used for all analyses performed on the sample.

        :return: The chosen cosmology.
        """
        return self._cosmo

    @property
    def obs_ids(self) -> dict:
        """
        Property meant to inform the user about the number (and identities) of ObsIDs associated with the sources
        in a given sample.

        :return: A dictionary (where the top level keys are the source names) of the ObsIDs associated with the
        individual sources in this sample.
        :rtype: dict
        """
        return {n: s.obs_ids for n, s in self._sources.items()}

    @property
    def instruments(self) -> dict:
        """
        Property meant to inform the user about the number (and identities) of instruments associated with ObsIDs
        associated with the sources in a given sample.

        :return: A dictionary (where the top level keys are the source names) of the instruments associated with
        ObsIDs associated with the individual sources in this sample.
        :rtype: dict
        """
        return {n: s.instruments for n, s in self._sources.items()}

    @property
    def failed_names(self) -> List[str]:
        """
        Yields the names of those sources that could not be declared for some reason.

        :return: A list of source names that could not be declared.
        :rtype: List[str]
        """
        return list(self._failed_sources)

    @property
    def failed_reasons(self) -> Dict[str, str]:
        """
        Returns a dictionary containing sources that failed to be declared successfully, and a
        simple reason why they couldn't be.

        :return: A dictionary of source names as keys, and reasons as values.
        :rtype: Dict[str, str]
        """
        return self._failed_sources

    def Lx(self, outer_radius: Union[str, Quantity], model: str,
           inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = Quantity(0.5, 'keV'),
           hi_en: Quantity = Quantity(2.0, 'keV'), group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
           over_sample: float = None, quality_checks: bool = True):
        """
        A get method for luminosities measured for the constituent sources of this sample. An error will be
        thrown if luminosities haven't been measured for the given region and model, no default model has been
        set, unlike the Tx method of ClusterSample. An extra condition that aims to only return 'good' data has
        been included, so that any Lx measurement with an uncertainty greater than value will be set to NaN, and
        a warning will be issued.

        :param str model: The name of the fitted model that you're requesting the luminosities
            from (e.g. constant*tbabs*apec).
        :param str/Quantity outer_radius: The name or value of the outer radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r200' would be acceptable
            for a GalaxyCluster, or Quantity(1000, 'kpc')). You may also pass a quantity containing radius values,
            with one value for each source in this sample.
        :param str/Quantity inner_radius: The name or value of the inner radius that was used for the generation of
            the spectra which were fitted to produce the desired result (for instance 'r500' would be acceptable
            for a GalaxyCluster, or Quantity(300, 'kpc')). By default this is zero arcseconds, resulting in a
            circular spectrum. You may also pass a quantity containing radius values, with one value for each
            source in this sample.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param bool group_spec: Whether the spectra that were fitted for the desired result were grouped.
        :param float min_counts: The minimum counts per channel, if the spectra that were fitted for the
            desired result were grouped by minimum counts.
        :param float min_sn: The minimum signal to noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal to noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param bool quality_checks: Whether the quality checks to make sure a returned value is good enough
            to use should be performed.
        :return: An Nx3 array Quantity where N is the number of sources. First column is the luminosity, second
            column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN
        :rtype: Quantity
        """
        # Has to be here to prevent circular import unfortunately
        from ..sas.spec import region_setup

        if outer_radius != 'region':
            # This just parses the input inner and outer radii into something predictable
            inn_rads, out_rads = region_setup(self, outer_radius, inner_radius, True, '')[1:]
        else:
            raise NotImplementedError("Sorry region fitting is currently well supported")

        lums = []
        for src_ind, src in enumerate(self._sources.values()):
            try:
                # Fetch the luminosity from a given source using the dedicated method
                lx_val = src.get_luminosities(out_rads[src_ind], model, inn_rads[src_ind], lo_en, hi_en, group_spec,
                                              min_counts, min_sn, over_sample)
                frac_err = lx_val[1:] / lx_val[0]
                # We check that no error is larger than the measured value, if quality checks are on
                if quality_checks and len(frac_err[frac_err >= 1]) != 0:
                    raise ValueError("{s} luminosity measurement's uncertainty greater than value.".format(s=src.name))
                else:
                    lums.append(lx_val)

            except (ValueError, ModelNotAssociatedError, ParameterNotAssociatedError) as err:
                # If any of the possible errors are thrown, we print the error as a warning and replace
                #  that entry with a NaN
                warn(str(err))
                lums.append(np.array([np.NaN, np.NaN, np.NaN]))

        # Turn the list of 3 element arrays into an Nx3 array which is then turned into an astropy Quantity
        lums = Quantity(np.array(lums), 'erg / s')

        # We're going to throw an error if all the luminosities are NaN, because obviously something is wrong
        check_lums = lums[~np.isnan(lums)]
        if len(check_lums) == 0:
            raise ValueError("All luminosities appear to be NaN.")

        return lums

    def check_spectra(self):
        """
        This method checks through the spectra associated with each source in the sample, printing a summary of which
        aren't usable and the reasons.
        """
        triggered = False
        print("\n-----------------------------------------------------")
        for s in self._sources:
            src = self._sources[s]
            src: BaseSource
            spectra = src.get_products("spectrum")
            spec_check = [(spec.obs_id, spec.instrument, spec.not_usable_reasons) for spec in spectra
                          if not spec.usable]
            if len(spec_check) > 0:
                print(src.name, spec_check)
                triggered = True

        if not triggered:
            print("All available spectra are okay")
        print("-----------------------------------------------------\n")

    def info(self):
        """
        Simple function to show basic information about the sample.
        """
        print("\n-----------------------------------------------------")
        print("Number of Sources - {}".format(len(self)))
        print("Redshift Information - {}".format(self.redshifts[0] is not None))
        print("-----------------------------------------------------\n")

    # The length of the sample object will be the number of associated sources.
    def __len__(self):
        """
        The result of using the Python len() command on this sample.

        :return: Number of sources in this sample.
        :rtype: int
        """
        return len(self._sources)

    def __iter__(self):
        """
        Called when initiating iterating through a BaseSample based object. Resets the counter _n.
        """
        self._n = 0
        return self

    def __next__(self):
        """
        Iterates the counter _n uses it to find the name of the corresponding source, then retrieves
        that source from the _sources dictionary. Sources are accessed using their name as a key, just like
        in dictionaries.
        """
        if self._n < self.__len__():
            result = self.__getitem__(self._names[self._n])
            self._n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, key: Union[int, str]) -> BaseSource:
        """
        This returns the relevant source when a sample is addressed using the name of a source as the key,
        or using an integer index.

        :param int/str key: The index or name of the source to fetch.
        :return: The relevant Source object.
        :rtype: BaseSource
        """
        if isinstance(key, (int, np.integer)):
            src = self._sources[self._names[key]]
        elif isinstance(key, str):
            src = self._sources[key]
        else:
            src = None
            raise ValueError("Only a source name or integer index may be used to address a sample object")
        return src

    def __delitem__(self, key: Union[int, str]):
        """
        This deletes a source from the sample, along with all accompanying data, using the index or
        name of the source.

        :param int/str key: The index or name of the source to delete.
        """
        if isinstance(key, (int, np.integer)):
            del self._sources[self._names[key]]
        elif isinstance(key, str):
            del self._sources[key]
            key = self._names.index(key)
        else:
            raise ValueError("Only a source name or integer index may be used to address a sample object")

        # Now the standard stored values
        del self._names[key]
        del self._accepted_inds[key]

        # This function is specific to the Sample type, as some Sample classes have extra information stored
        #  that will need to be deleted.
        self._del_data(key)

    def _del_data(self, key: int):
        """
        This function will be replaced in subclasses that store more information about sources
        in internal attributes.

        :param int key: The index or name of the source to delete.
        """
        pass



