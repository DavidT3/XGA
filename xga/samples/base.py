#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 21/10/2020, 10:11. Copyright (c) David J Turner

from warnings import warn

import numpy as np
from astropy.cosmology import Planck15
from astropy.units import Quantity
from numpy import ndarray
from tqdm import tqdm

from ..exceptions import NoMatchFoundError
from ..sources.base import BaseSource


class BaseSample:
    def __init__(self, ra: ndarray, dec: ndarray, redshift: ndarray = None, name: ndarray = None, cosmology=Planck15,
                 load_products: bool = True, load_fits: bool = False, no_prog_bar: bool = False):
        # Slight duplication of data here, but I'm going to save the inputted information into attributes,
        #  that way I don't have to iterate through my sources everytime the user might want to pull it out
        self._ra_dec = []
        self._redshifts = []
        # This is an empty list so that if no name is passed the automatically generated names can be added during
        #  declaration
        self._names = []
        self._cosmo = cosmology
        self._sources = {}

        # Just checking that, if names are being supplied, then they are all unique
        if name is not None and len(set(name)) != len(name):
            raise ValueError("Names supplied to samples must be unique.")

        dec_base = tqdm(desc="Declaring BaseSource Sample", total=len(ra), disable=no_prog_bar)
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
                self._ra_dec.append((r, d))
                self._redshifts.append(z)
            except NoMatchFoundError:
                warn("Source {n} does not appear to have any XMM data, and will not be included in the "
                     "sample.".format(n=n))
            dec_base.update(1)
        dec_base.close()

        self._redshifts = np.array(self._redshifts)

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
        return Quantity(self._ra_dec, 'deg')

    @property
    def redshifts(self) -> ndarray:
        """
        Property getter for the list of redshift positions of the sources in  this
        sample (if available). If no redshifts were supplied, None will be returned.
        :return: List of redshifts.
        :rtype: ndarray
        """
        return self._redshifts

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
        print("Redshift Information - {}".format(self._redshifts is not None))
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
        :return:
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

    def __getitem__(self, key):
        """
        This returns the relevant source when a sample is addressed using the name of a source as the key,
        or using an integer index.
        """
        if isinstance(key, int):
            src = self._sources[self._names[key]]
        elif isinstance(key, str):
            src = self._sources[key]
        else:
            ValueError("Only a source name or integer index may be used to address a sample object")
        return src



