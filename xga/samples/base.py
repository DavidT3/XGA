#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 04/06/2025, 13:31. Copyright (c) The Contributors

from typing import Union, List, Dict
from warnings import warn

import numpy as np
from astropy.cosmology import Cosmology
from astropy.units import Quantity, Unit, arcmin, UnitConversionError
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from .. import DEFAULT_COSMO
from ..exceptions import NoMatchFoundError, ModelNotAssociatedError, ParameterNotAssociatedError, \
    FitConfNotAssociatedError
from ..exceptions import NoValidObservationsError
from ..sources.base import BaseSource
from ..sourcetools.misc import coord_to_name


class BaseSample:
    """
    The superclass for all sample classes. These store whole samples of sources, to make bulk analysis of
    interesting X-ray sources easy. This in particular creates samples of BaseSource object. It doesn't seem
    likely that users should need to declare one of these, they should use one of the general ExtendedSample or
    PointSample classes if they are doing exploratory analyses, or a more specific subclass like ClusterSample.

    :param ndarray ra: The right-ascensions of the sources, in degrees.
    :param ndarray dec: The declinations of the sources, in degrees.
    :param ndarray redshift: The redshifts of the sources, optional. Default is None
    :param ndarray name: The names of the sources, optional. Default is None, in which case the names will be
        constructed from the coordinates.
    :param Cosmology cosmology: An astropy cosmology object to be used in distance calculations and analyses.
    :param bool load_products: Whether existing products should be loaded from disk.
    :param bool load_fits: Whether existing fits should be loaded from disk.
    :param bool no_prog_bar: Whether a progress bar should be shown as sources are declared.
    """
    def __init__(self, ra: ndarray, dec: ndarray, redshift: ndarray = None, name: ndarray = None,
                 cosmology: Cosmology = DEFAULT_COSMO, load_products: bool = True, load_fits: bool = False,
                 no_prog_bar: bool = False):

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
                    # We declare the source object, making sure to tell it that its part of a sample
                    #  using in_sample=True
                    temp = BaseSource(r, d, z, n, cosmology, load_products, load_fits, True)
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

                    # We record that a particular source name was not successfully declared
                    self._failed_sources[n] = "NoMatch"
                dec_base.update(1)

        # It is possible (especially if someone is using the Sample classes as a way to check whether things have
        #  XMM data) that no sources will have been declared by this point, in which case it should fail now
        if len(self._sources) == 0:
            raise NoValidObservationsError("No sources have been declared, likely meaning that none of the sample have"
                                           " valid XMM data.")

        # Put all the warnings for there being no XMM data in one - I think it's neater. Wait until after the check
        #  to make sure that are some sources because in that case this warning is redundant.
        # HOWEVER - I only want this warning to appear in certain circumstances. For instance I wouldn't want it
        #  to be triggered here for a ClusterSample declaration that has called the super init (this method), as that
        #  class declaration does its own (somewhat different) check on which sources have data
        no_data = [name for name in self._failed_sources if self._failed_sources[name] == 'NoMatch']
        # If there are names in that list, then we do the warning
        if len(no_data) != 0 and type(self) == BaseSample:
            warn("The following do not appear to have any XMM data, and will not be included in the "
                 "sample (can also check .failed_names); {n}".format(n=', '.join(no_data)))

        # This calls the method that checks for suppressed source-level warnings that occurred during declaration, but
        #  only if this init has been called for a BaseSample declaration, rather than by a sub-class
        if type(self) == BaseSample:
            self._check_source_warnings()

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
    def peaks(self) -> Quantity:
        """
        This property getter will fetch peak coordinates for the sources in this sample. An exception will
        be raised if the source objects do not have a peak attribute, and a warning will be presented if all
        user supplied ra-dec values are the same as all peak values.

        :return: A quantity containing the peak coordinates measured for the sources in the sample.
        :rtype: Quantity
        """
        if not hasattr(self[0], 'peak'):
            raise AttributeError("The sources making up this sample do not have a peak property.")

        if all([np.array_equal(s.ra_dec.value, s.peak.value) for s in self._sources.values()]):
            warn("All user supplied ra-dec values are the same as the peak ra-dec values, likely means that peak "
                 "finding was not run for this sample.")

        return Quantity([s.peak.value for s in self._sources.values()], 'deg')

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
        return Quantity([s.nH for s in self._sources.values()])

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

    @property
    def suppressed_warnings(self) -> Dict[str, List[str]]:
        """
        A property getter for a dictionary of the suppressed warnings that occurred during the declaration of
        sources for this sample.

        :return: A dictionary with source name as keys, and lists of warning text as values. Sources are
            only included if they have had suppressed warnings.
        :rtype: Dict[str, List[str]]
        """
        return {n: s.suppressed_warnings for n, s in self._sources.items() if len(s.suppressed_warnings) > 0}

    def Lx(self, outer_radius: Union[str, Quantity], model: str,
           inner_radius: Union[str, Quantity] = Quantity(0, 'arcsec'), lo_en: Quantity = Quantity(0.5, 'keV'),
           hi_en: Quantity = Quantity(2.0, 'keV'), group_spec: bool = True, min_counts: int = 5, min_sn: float = None,
           over_sample: float = None, quality_checks: bool = True, fit_conf: Union[str, dict] = None):
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
        :param float min_sn: The minimum signal-to-noise per channel, if the spectra that were fitted for the
            desired result were grouped by minimum signal-to-noise.
        :param float over_sample: The level of oversampling applied on the spectra that were fitted.
        :param bool quality_checks: Whether the quality checks to make sure a returned value is good enough
            to use should be performed.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: An Nx3 array Quantity where N is the number of sources. First column is the luminosity, second
            column is the -err, and 3rd column is the +err. If a fit failed then that entry will be NaN
        :rtype: Quantity
        """
        # Has to be here to prevent circular import unfortunately
        from ..sas._common import region_setup

        if outer_radius != 'region':
            # This just parses the input inner and outer radii into something predictable
            inn_rads, out_rads = region_setup(self, outer_radius, inner_radius, True, '')[1:]
        else:
            raise NotImplementedError("Sorry region fitting is currently well supported")

        lums = []
        warns = []
        for src_ind, src in enumerate(self._sources.values()):
            src: BaseSource
            try:
                # Fetch the luminosity from a given source using the dedicated method
                lx_val = src.get_luminosities(out_rads[src_ind], model, inn_rads[src_ind], lo_en, hi_en, group_spec,
                                              min_counts, min_sn, over_sample, fit_conf)
                frac_err = lx_val[1:] / lx_val[0]
                # We check that no error is larger than the measured value, if quality checks are on
                if quality_checks and len(frac_err[frac_err >= 1]) != 0:
                    raise ValueError("{s} luminosity measurement's uncertainty greater than value.".format(s=src.name))
                else:
                    lums.append(lx_val)

            except (ValueError, ModelNotAssociatedError, ParameterNotAssociatedError, FitConfNotAssociatedError) as err:
                # If any of the possible errors are thrown, we grab the name of the source and replace
                #  that entry with a NaN - the names will be included in a warning at the end
                # warn(str(err))
                warns.append(src.name)
                lums.append(np.array([np.nan, np.nan, np.nan]))

        # Turn the list of 3 element arrays into an Nx3 array which is then turned into an astropy Quantity
        lums = Quantity(np.array(lums), 'erg / s')

        # We're going to throw an error if all the luminosities are NaN, because obviously something is wrong
        check_lums = lums[~np.isnan(lums)]
        if len(check_lums) == 0:
            raise ValueError("All luminosities appear to be NaN.")

        # If there were any errors for any of the sources, we include their names in a warning here
        if len(warns) > 0:
            warn_text = ("Problems occurred (there may have been no successful model fit) while attempting to retrieve "
                         "luminosities for the following sources; {}".format(",".join(warns)))
            warn(warn_text, stacklevel=2)

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

    def offsets(self, off_unit: Union[Unit, str] = arcmin) -> Quantity:
        """
        Uses the offset method built into the sources to fetch the offsets between ra_dec and peak for all
        sources in the sample.

        :param Unit/str off_unit: The desired unit for the offsets to be in.
        :return: The offsets.
        :rtype: Quantity
        """
        # This call fetches peaks which we then never use, but it triggers a check that will trigger a warning if
        #  all the ra_dec values are the same as all the peak values
        ps = self.peaks
        # Use list comprehension to grab all the offsets, then convert to a single quantity
        offsets = Quantity([s.offset(off_unit) for s in self._sources.values()])

        return offsets

    def view_offset_dist(self, off_unit: Union[Unit, str] = arcmin, figsize: tuple = (6, 6),
                         bins: Union[str, np.ndarray, int] = 'auto', x_lims: Quantity = None, x_scale: str = 'log',
                         y_scale: str = 'log', colour: str = "cadetblue", alpha: float = 0.5, title: str = '',
                         font_size: int = 13, data_label: str = '', y_label: str = "N", save_path: str = None):
        """
        A method to create a histogram of the offsets of user from peak coordinates for the objects in
        this sample. A range of options to customise the plot are supplied.

        :param Unit/str off_unit: The desired output unit of separation, default is arcmin.
        :param tuple figsize: The size of the figure produced.
        :param str/np.ndarray/int bins: This is passed directly through to the plt.hist bins argument, default is auto.
        :param Quantity x_lims: Set the limits for the x-axis, first element should be lower, second element
            upper. Default is None in which case matplotlib decides.
        :param str x_scale: The scale for the x-axis, default is log.
        :param str y_scale: The scale for the y-axis, default is log.
        :param str colour: The colour of the bars, default is cadetblue.
        :param float alpha: The alpha (transparency) value of the the bars, default is 0.5.
        :param str title: A title to be added to the plot. Default is empty, which means no title will be
            added. Fontsize will be 1.2 times the font_size argument.
        :param int font_size: The font_size argument sets the font_size of the axis labels. Default is 13.
        :param str data_label: Whether the data should be labelled, default is empty. If this is set a legend will
            be added.
        :param str y_label: The y-axis label, default is N.
        :param str save_path: A path to save the figure on, optional. Default is None in which case the figure is
            not saved to disk.
        """

        # Uses the convenience method for calculating separation that is built into BaseSource to grab
        #  all the offsets
        seps = self.offsets(off_unit)

        # x_lims is allowed to be None, in which case it will be automatic, but if it isn't then we check it
        #  is structured as we expect it to be
        if x_lims is not None and x_lims.unit != seps.unit:
            raise UnitConversionError("The x_lims unit must be the same as off_unit.")
        elif x_lims is not None and (x_lims.isscalar or len(x_lims) != 2):
            raise ValueError("x_lims must have one entry for lower limit, and one for upper limit.")

        # Set up the figure, with minorticks on and ticks facing inwards
        plt.figure(figsize=figsize)
        plt.minorticks_on()
        plt.tick_params(which='both', top=True, right=True, direction='in')

        # Plot the histogram, stepfilled makes it look good when saved as a pdf
        plt.hist(seps.value, color=colour, label=data_label, alpha=alpha, bins=bins, histtype='stepfilled')

        # Set the y_label as the argument value
        plt.ylabel(y_label, fontsize=font_size)

        # If x-axis limits were passed, then set them.
        if x_lims is not None:
            plt.xlim(*x_lims.value)

        # Setup the x-axis label, using the separation value units, then add to figure
        off_label = r"Offset [{u}]".format(u=seps.unit.to_string())
        plt.xlabel(off_label, fontsize=font_size)

        # Set the axis scales to the arguments.
        plt.xscale(x_scale)
        plt.yscale(y_scale)

        # If the data_label argument was not empty, we must add a legend.
        if data_label != '':
            plt.legend(fontsize=font_size)

        # If the title argument was not empty, we must add a legend
        if title != '':
            plt.title(title, fontsize=font_size*1.2)

        # Turn on tight layout to remove white space
        plt.tight_layout()

        # Check whether the user wants this plot save or not, if so then call savefig
        if save_path is not None:
            plt.savefig(save_path)

        # Show the figure
        plt.show()

    def info(self):
        """
        Simple function to show basic information about the sample.
        """
        # Finding the number of sources in the sample that have been detected in AT LEAST one ObsID
        num_det = sum(np.array([sum(s.detected.values()) for s in self._sources.values()]) >= 1)
        perc_det = int(round(num_det / len(self._sources), 2) * 100)
        print("\n-----------------------------------------------------")
        print("Number of Sources - {}".format(len(self)))
        print("Redshift Information - {}".format(self.redshifts[0] is not None))
        print("Sources with ≥1 detection - {n} [{p}%]".format(n=num_det, p=perc_det))
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

    def _check_source_warnings(self):
        """
        This method checks the suppressed_warnings property of the member sources, and if any have had warnings
        suppressed then it itself raises a warning that instructs the user to look at the suppressed_warnings
        property of the sample. It doesn't print them all because that could lead to a confusing mess. This method
        is to be called at the end of every sub-class init.
        """
        if any([len(src.suppressed_warnings) > 0 for src in self._sources.values()]):
            warn("Non-fatal warnings occurred during the declaration of some sources, to access them please use the "
                 "suppressed_warnings property of this sample.", stacklevel=2)

    def _del_data(self, key: int):
        """
        This function will be replaced in subclasses that store more information about sources
        in internal attributes.

        :param int key: The index or name of the source to delete.
        """
        pass
