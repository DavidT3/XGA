#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (turne540@msu.edu) 11/03/2025, 22:44. Copyright (c) The Contributors

import os
from copy import deepcopy
from typing import Tuple, Union, List, Dict
from warnings import warn

import numpy as np
from astropy.io import fits
from astropy.units import Quantity, Unit, UnitConversionError
from fitsio import hdu, FITS, read, read_header, FITSHDR
from matplotlib import legend_handler
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D

from . import BaseProduct, BaseAggregateProduct, BaseProfile1D
from ..exceptions import ModelNotAssociatedError, ParameterNotAssociatedError, XGASetIDError, NotAssociatedError, \
    FailedProductError
from ..products.profile import ProjectedGasTemperature1D, ProjectedGasMetallicity1D, Generic1D, APECNormalisation1D
from ..utils import dict_search


class Spectrum(BaseProduct):
    """
    This class is the XGA product responsible for storing an individual spectrum. Various qualities that can be
    measured from it (X-ray luminosity for example) can be associated with an instance of this object, as well as
    conversion factors that can be calculated from XSPEC. If a model has been fitted then the data and model
    can be viewed.

    :param str path: The path to the spectrum file.
    :param str rmf_path: The path to the RMF generated for the spectrum file.
    :param str arf_path: The path to the ARF generated for the spectrum file.
    :param str b_path: The path to the background spectrum generated for the spectrum file.
    :param Quantity central_coord: The central coordinate of the spectrum region.
    :param Quantity inn_rad: The inner radius of the spectrum region.
    :param Quantity out_rad: The outer radius of the spectrum region.
    :param str obs_id: The ObsID from which this spectrum was generated.
    :param str instrument: The instrument which this spectrum was generated.
    :param bool grouped: Was this spectrum grouped?
    :param int min_counts: The minimum counts applied for the grouping.
    :param float min_sn: The minimum signal to noise applied for the grouping.
    :param int over_sample: Level of oversampling applied to spectrum grouping.
    :param str stdout_str: The stdout from the generation process.
    :param str stderr_str: The stderr for the generation process.
    :param str gen_cmd: The generation command for the spectrum stack.
    :param bool region: Was this spectrum generated from a region in a region file?
    :param str b_rmf_path: The path to the RMF generated for the background spectrum (if applicable, XGA no longer
        generates these by default, as XSPEC does not make use of them).
    :param str b_arf_path: The path to the ARF generated for the background spectrum (if applicable, XGA no longer
        generates these by default, as XSPEC does not make use of them).
    """
    def __init__(self, path: str, rmf_path: str, arf_path: str, b_path: str,
                 central_coord: Quantity, inn_rad: Quantity, out_rad: Quantity, obs_id: str, instrument: str,
                 grouped: bool, min_counts: int, min_sn: float, over_sample: int, stdout_str: str,
                 stderr_str: str, gen_cmd: str, region: bool = False, b_rmf_path: str = '', b_arf_path: str = ''):
        """
        The init of the Spectrum class, sets up both the base product behind the Spectrum and the specific
        information/abilities that a spectrum needs.
        """
        super().__init__(path, obs_id, instrument, stdout_str, stderr_str, gen_cmd)
        self._prod_type = "spectrum"

        if os.path.exists(rmf_path):
            self._rmf = rmf_path
        else:
            self._rmf = ''
            self._usable = False
            self._why_unusable.append("RMFPathDoesNotExist")

        if os.path.exists(arf_path):
            self._arf = arf_path
        else:
            self._arf = ''
            self._usable = False
            self._why_unusable.append("ARFPathDoesNotExist")

        if os.path.exists(b_path):
            self._back_spec = b_path
        else:
            self._back_spec = ''
            self._usable = False
            self._why_unusable.append("BackSpecPathDoesNotExist")

        if b_rmf_path != '' and os.path.exists(b_rmf_path):
            self._back_rmf = b_rmf_path
        elif b_rmf_path == '':
            self._back_rmf = None
        else:
            self._back_rmf = ''
            self._usable = False
            self._why_unusable.append("BackRMFPathDoesNotExist")

        if b_arf_path != '' and os.path.exists(b_arf_path):
            self._back_arf = b_arf_path
        elif b_arf_path == '':
            self._back_arf = None
        else:
            self._back_arf = ''
            self._usable = False
            self._why_unusable.append("BackARFPathDoesNotExist")

        # Storing the central coordinate of this spectrum
        self._central_coord = central_coord

        # Storing the region information
        self._inner_rad = inn_rad
        self._outer_rad = out_rad
        # And also the shape of the region
        if self._inner_rad.isscalar:
            self._shape = 'circular'
        else:
            self._shape = 'elliptical'

        # If this spectrum has just been generated by XGA then we'll set the headers, otherwise its
        #  too slow and must be avoided. I am assuming here that the gen_cmd will be "" if the object
        #  hasn't just been generated - which is true of XGA's behaviour
        if gen_cmd != "":
            try:
                self._update_spec_headers("main")
                self._update_spec_headers("back")
            except OSError as err:
                self._usable = False
                self._why_unusable.append("FITSIOOSError")

        # Here we store the fit information
        self._exp = None
        self._plot_data = {}
        self._luminosities = {}
        self._count_rate = {}
        # self._fit_stat = {}
        # self._test_stat = {}

        # This is specifically for fakeit runs (for cntrate - lum conversions) on the ARF/RMF
        #  associated with this Spectrum
        self._conv_factors = {}

        # This set of properties describe the configuration of evselect/specgroup during generation
        self._grouped = grouped
        self._min_counts = min_counts
        self._min_sn = min_sn
        if self._grouped and self._min_counts is not None:
            self._grouped_on = 'counts'
        elif self._grouped and self._min_sn is not None:
            self._grouped_on = 'signal to noise'
        else:
            self._grouped_on = None

        # Not to do with grouping, but this states the level of oversampling requested from evselect
        self._over_sample = over_sample

        # This describes whether this spectrum was generated directly from a region present in a region file
        self._region = region

        # Here we generate the storage key for this object, its just convenient to do it in here
        # Sets up the extra part of the storage key name depending on if grouping is enabled
        if grouped and min_counts is not None:
            extra_name = "_mincnt{}".format(min_counts)
        elif grouped and min_sn is not None:
            extra_name = "_minsn{}".format(min_sn)
        else:
            extra_name = ''

        # And if it was oversampled during generation then we need to include that as well
        if over_sample is not None:
            extra_name += "_ovsamp{ov}".format(ov=over_sample)

        spec_storage_name = "ra{ra}_dec{dec}_ri{ri}_ro{ro}_grp{gr}"
        if not self._region and self.inner_rad.isscalar:
            spec_storage_name = spec_storage_name.format(ra=self.central_coord[0].value,
                                                         dec=self.central_coord[1].value,
                                                         ri=self._inner_rad.value, ro=self._outer_rad.value,
                                                         gr=grouped)
        elif not self._region and not self._inner_rad.isscalar:
            inn_rad_str = 'and'.join(self._inner_rad.value.astype(str))
            out_rad_str = 'and'.join(self._outer_rad.value.astype(str))
            spec_storage_name = spec_storage_name.format(ra=self.central_coord[0].value,
                                                         dec=self.central_coord[1].value, ri=inn_rad_str,
                                                         ro=out_rad_str, gr=grouped)
        else:
            spec_storage_name = "region_grp{gr}".format(gr=grouped)

        spec_storage_name += extra_name
        # And we save the completed key to an attribute
        self._storage_key = spec_storage_name

        # This attribute is set via the property, ONLY if this spectrum is considered to be a member of a set
        #  of annular spectra. It describes which position in the set this spectrum has
        self._ann_ident = None
        # This holds a unique random identifier for the set itself, and again will only be set from outside
        self._set_ident = None

        # We also setup empty attributes (akin to the data attribute in the Image class) to store the actual
        #  data from the spectrum and background spectrum should they ever need to be read into memory from the file
        #  by this object
        # First the raw counts per channel of the source spectrum
        self._spec_counts = None
        # Secondly the channels with which each count data point is associated
        self._spec_channels = None
        # And finally attributes to store the group and quality of each data entry (group showing how they were
        #  grouped from the original raw spectrum and quality contains a quality flag 0=good 1=not good)
        self._spec_group = None
        self._spec_quality = None
        # Also add an attribute to store the overall header of the spectrum
        self._prim_spec_header = None
        # And an attribute for the SPECTRUM table header
        self._spec_spec_header = None

        # Now all of the same but for the background spectrum
        self._back_counts = None
        self._back_channels = None
        self._back_group = None
        self._back_quality = None
        self._prim_back_header = None
        self._spec_back_header = None

        # Attributes to store ARF information
        # The actual effective area information will live in this attribute
        self._arf_eff_area = None
        # The corresponding energy bounds will be stored in these attributes
        self._arf_lo_en = None
        self._arf_hi_en = None

        # Attributes to store RMF information
        # This is a one is a bit of a cop out for now, I'm going to store the entire redist matrix table because
        #  I don't know if I'm actually going to do anything with it
        self._redist_matrix_info = None
        # Here I will store lookup information to convert channels to a lower and upper energy bound
        self._rmf_channels = None
        self._rmf_channels_lo_en = None
        self._rmf_channels_hi_en = None

    def _update_spec_headers(self, which_spec: str):
        """
        An internal method that will 'push' the current class attributes that hold the paths to data products
        (like ARF and RMF) to the relevant spectrum file.

        :param str which_spec: A flag that tells the method whether to update the header of
            the main or background spectrum.
        """
        # This function is meant for internal use only, so I won't check that the passed-in file paths
        #  actually exist. This will have been checked already
        if which_spec == "main" and self.usable:
            # Currently having to use astropy's fits interface, I don't really want to because of risk of segfaults
            with fits.open(self._path, mode='update') as spec_fits:
                # I delete the headers first, as I've found issues with XSPEC not being able to read the
                #  path if the SAS version I'm using adds entries for the headers before I do. See issue #745
                # Do have to check that the offending headers are actually present, as they won't be introduced by
                #  all versions of SAS
                if "RESPFILE" in spec_fits["SPECTRUM"].header:
                    del spec_fits["SPECTRUM"].header["RESPFILE"]
                if "ANCRFILE" in spec_fits["SPECTRUM"].header:
                    del spec_fits["SPECTRUM"].header["ANCRFILE"]
                if "BACKFILE" in spec_fits["SPECTRUM"].header:
                    del spec_fits["SPECTRUM"].header["BACKFILE"]

                # This writes the new response file paths to the headers.
                spec_fits["SPECTRUM"].header["RESPFILE"] = self._rmf
                spec_fits["SPECTRUM"].header["ANCRFILE"] = self._arf
                spec_fits["SPECTRUM"].header["BACKFILE"] = self._back_spec

        elif which_spec == "back" and self.usable:
            with fits.open(self._back_spec, mode='update') as spec_fits:
                if self._back_rmf is not None:
                    if 'RESPFILE' in spec_fits["SPECTRUM"].header:
                        del spec_fits["SPECTRUM"].header["RESPFILE"]
                    spec_fits["SPECTRUM"].header["RESPFILE"] = self._back_rmf
                if self._back_arf is not None:
                    if 'ANCRFILE' in spec_fits["SPECTRUM"].header:
                        del spec_fits["SPECTRUM"].header["ANCRFILE"]
                    spec_fits["SPECTRUM"].header["ANCRFILE"] = self._back_arf

    def _read_on_demand(self, src_spec: bool = True):
        """
        Internal method to read the spectrum (or background spectrum) associated with this Spectrum object into
        memory when it is requested by another method. Doing it on-demand saves on wasting memory.

        :param bool src_spec: This parameter controls whether it is the source or background spectrum that
            is read into memory. If True (the default) then the source spectrum is read, otherwise the background
            spectrum is read.
        """

        # Usable flag to check that nothing went wrong in the spectrum generation
        if self.usable:
            try:
                # Populate the source spectrum relevant attributes if we're asked to load the source
                if src_spec:
                    # Make this variable so the FileNotFoundError can work
                    rel_path = self.path
                    all_dat = read(rel_path)
                    self._spec_counts = all_dat['COUNTS']
                    self._spec_channels = all_dat['CHANNEL']
                    # If the spectrum has not been grouped it may not have this column
                    if "GROUPING" in all_dat.dtype.names:
                        self._spec_group = all_dat['GROUPING']
                    self._spec_quality = all_dat['QUALITY']

                # And if not then the only other option is to populate the background spectrum attributes
                else:
                    rel_path = self.background
                    all_dat = read(rel_path)
                    self._back_counts = all_dat['COUNTS']
                    self._back_channels = all_dat['CHANNEL']

                    # Background spectra do not necessarily have these entries
                    if "GROUPING" in all_dat.dtype.names:
                        self._back_group = all_dat['GROUPING']
                    if "QUALITY" in all_dat.dtype.names:
                        self._back_quality = all_dat['QUALITY']

            except OSError:
                raise FileNotFoundError("FITSIO read cannot open {f}, possibly because there is a problem with "
                                        "the file, it doesn't exist, or maybe an SFTP problem? This product is "
                                        "associated with {s}.".format(f=rel_path, s=self.src_name))

        else:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot access "
                                     "data from it; reason give is {}. Check the usable attribute next "
                                     "time".format(reasons))

    def _read_header_on_demand(self, src_spec: bool = True, primary_header: bool = True):
        """
        Internal method to read the spectrum (or background spectrum) header associated with this Spectrum object into
        memory when it is requested by another method. Doing it on-demand saves on wasting memory. Either the header of

        :param bool src_spec: This parameter controls whether it is the source or background spectrum header that
            is read into memory. If True (the default) then the source spectrum is read, otherwise the background
            spectrum is read.
        :param bool primary_header: Whether the header from the primary table (the default) should be read. The
            alternative is to read the header of the SPECTRUM table.
        """

        # Usable flag to check that nothing went wrong in the spectrum generation
        if self.usable:
            try:
                # Here we read in the particular header that has been requested to the relevant attribute. The
                #  source or background spectrum headers can be chosen, as well as whether the primary header (for the
                #  whole file) or the SPECTRUM header (for the spectrum table) are read in.
                if src_spec and primary_header:
                    rel_path = self.path
                    self._prim_spec_header = read_header(rel_path)
                elif src_spec and not primary_header:
                    rel_path = self.path
                    self._spec_spec_header = read_header(rel_path, 'SPECTRUM')
                elif not src_spec and primary_header:
                    rel_path = self.background
                    self._prim_back_header = read_header(rel_path)
                elif not src_spec and not primary_header:
                    rel_path = self.background
                    self._spec_back_header = read_header(rel_path, 'SPECTRUM')

            except OSError:
                raise FileNotFoundError("FITSIO read cannot open {f}, possibly because there is a problem with "
                                        "the file, it doesn't exist, or maybe an SFTP problem? This product is "
                                        "associated with {s}.".format(f=rel_path, s=self.src_name))

        else:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot access "
                                     "data from it; reason give is {}. Check the usable attribute next "
                                     "time".format(reasons))

    def _read_response_on_demand(self, rmf: bool = True):
        """
        Internal method to read the response information for this spectrum into memory. Either the redistribution
        matrix and channel-to-energy conversion information from the RMF, or the effective area as a discrete function
        of energy from the ARF.

        :param bool rmf: Whether the RMF information should be read into memory, default True. If False, the ARF
            information will be read into memory.
        """

        # Usable flag to check that nothing went wrong in the spectrum generation
        if self.usable:
            try:
                # Populate the relevant response attributes if we're asked to load the rmf
                if rmf:
                    # This so that the error message if it can't be read works properly
                    rel_path = self.rmf
                    # I'm reading the whole fits file because there are two tables I'm interested in
                    all_dat = FITS(rel_path)
                    # This stores the matrix table as a numpy array with named columns
                    self._redist_matrix_info = all_dat['MATRIX'].read().copy()
                    # Reading out the second table containing the conversions between energy and channel
                    chan_dat = all_dat['EBOUNDS'].read()
                    # Setting up arrays of RMF channels with their equivelant lower and upper energy bounds
                    self._rmf_channels = chan_dat['CHANNEL']
                    self._rmf_channels_lo_en = Quantity(chan_dat['E_MIN'], 'keV')
                    self._rmf_channels_hi_en = Quantity(chan_dat['E_MAX'], 'keV')
                    all_dat.close()
                # And if not then the only other option is to populate the ARF attributes
                else:
                    rel_path = self.arf
                    # Read in the ARF fits file from the arf property
                    arf_read = FITS(rel_path)
                    # Read out the data from the ARF table into attributes
                    self._arf_lo_en = Quantity(arf_read[1]['ENERG_LO'].read(), 'keV')
                    self._arf_hi_en = Quantity(arf_read[1]['ENERG_HI'].read(), 'keV')
                    self._arf_eff_area = Quantity(arf_read[1]['SPECRESP'].read(), 'cm^2')

                    # And make sure to close the arf file after reading
                    arf_read.close()

            except OSError:
                raise FileNotFoundError("FITSIO read cannot open {f}, possibly because there is a problem with "
                                        "the file, it doesn't exist, or maybe an SFTP problem? This product is "
                                        "associated with {s}.".format(f=rel_path, s=self.src_name))

        else:
            reasons = ", ".join(self.not_usable_reasons)
            raise FailedProductError("SAS failed to generate this product successfully, so you cannot access "
                                     "data from it; reason give is {}. Check the usable attribute next "
                                     "time".format(reasons))

    @property
    def header(self) -> FITSHDR:
        """
        The SPECTRUM table header of the source spectrum. This property was called header because I suspect
        this will be more the more useful of the two headers that a Spectrum instance allows you to access.

        :rtype: FITSHDR
        :return: The SPECTRUM fits table header.
        """
        # Check whether the specific header has been read in yet, if not then trigger that
        if self._spec_spec_header is None:
            self._read_header_on_demand(src_spec=True, primary_header=False)
        return self._spec_spec_header

    @property
    def primary_header(self) -> FITSHDR:
        """
        The PRIMARY (overall) header of the source spectrum file.

        :rtype: FITSHDR
        :return: The PRIMARY (overall) fits file header.
                """
        # Check whether the specific header has been read in yet, if not then trigger that
        if self._prim_spec_header is None:
            self._read_header_on_demand(src_spec=True, primary_header=True)
        return self._prim_spec_header

    @property
    def counts(self) -> Quantity:
        """
        The array of counts associated with each channel of the spectrum, with a second column containing the
        Poisson error.

        :rtype: Quantity
        :return: The counts quantity in units of 'ct'.
        """
        # Checks whether the initial value of the spec counts attribute has been overwritten, if not then I run
        #  the read on demand method to grab the information from the file.
        if self._spec_counts is None:
            self._read_on_demand(True)

        return Quantity([self._spec_counts, np.sqrt(self._spec_counts)], 'ct').T

    @property
    def exposure(self) -> Quantity:
        """
        The weighted (from individual exposure times of different CCD chips) exposure time of the
        source spectrum, as used by XSPEC.

        :rtype: Quantity
        :return: The exposure time for the source spectrum, in units of seconds.
        """
        # Fetch the exposure time from the header and create a quantity
        exp_time = Quantity(float(self.header['EXPOSURE']), 's')
        return exp_time

    @property
    def count_rates(self) -> Quantity:
        """
        The array of counts/second associated with each channel of the spectrum. This takes the counts property
        and divides it by the EXPOSURE entry in the spectrum header. A second column containing uncertainty is
        included.

        :rtype: Quantity
        :return: The counts/second quantity in units of 'ct/s'.
        """

        return self.counts/self.exposure

    @property
    def channels(self) -> Quantity:
        """
        The array of instrument channels in the spectrum.

        :rtype: Quantity
        :return: An array of channels.
        """
        # Checks whether the initial value of the channels attribute has been overwritten, if not then I run
        #  the read on demand method to grab the information from the file.
        if self._spec_channels is None:
            self._read_on_demand(True)

        return Quantity(self._spec_channels)

    @property
    def energies(self) -> Quantity:
        """
        The array of instrument channels, converted to energy using information from the RMF.

        :rtype: Quantity
        :return: An array of channel energy midpoints, in keV.
        """

        return self.conv_channel_energy(self.channels)

    @property
    def grouping(self) -> np.ndarray:
        """
        The grouping information from the spectrum. A 1 entry indicates the first channel in a group and -1
        indicates a member of the current group.

        :rtype: np.ndarray
        :return: An array of group IDs.
        """
        if not self.grouped:
            raise ValueError("This spectrum was generated without grouping, and so you cannot "
                             "retrieve grouping information.")

        # Checks whether the initial value of the attribute has been overwritten, if not then I run
        #  the read on demand method to grab the information from the file.
        if self._spec_group is None:
            self._read_on_demand(True)

        return self._spec_group

    @property
    def quality(self) -> np.ndarray:
        """
        The quality information from the spectrum. 0 = good quality, 1 is bad quality and 2 means dubious.

        :rtype: np.ndarray
        :return: An array of quality flags.
        """
        # Checks whether the initial value of the attribute has been overwritten, if not then I run
        #  the read on demand method to grab the information from the file.
        if self._spec_quality is None:
            self._read_on_demand(True)

        return self._spec_quality

    @property
    def back_header(self) -> FITSHDR:
        """
        The SPECTRUM table header of the background spectrum. This property was called header because I suspect
        this will be more the more useful of the two headers that a Spectrum instance allows you to access.

        :rtype: FITSHDR
        :return: The SPECTRUM fits table header.
        """
        # Check whether the specific header has been read in yet, if not then trigger that
        if self._spec_back_header is None:
            self._read_header_on_demand(src_spec=False, primary_header=False)
        return self._spec_back_header

    @property
    def back_primary_header(self) -> FITSHDR:
        """
        The PRIMARY (overall) header of the background spectrum file.

        :rtype: FITSHDR
        :return: The PRIMARY (overall) fits file header.
                """
        # Check whether the specific header has been read in yet, if not then trigger that
        if self._prim_back_header is None:
            self._read_header_on_demand(src_spec=False, primary_header=True)
        return self._prim_back_header

    @property
    def back_counts(self) -> Quantity:
        """
        The array of counts associated with each channel of the background spectrum, with a second column containing
        the Poisson error.

        :rtype: Quantity
        :return: The counts quantity in units of 'ct'.
        """
        # Checks whether the initial value of the background spec counts attribute has been overwritten, if not
        #  then I run the read on demand method to grab the information from the file.
        if self._back_counts is None:
            # Passing false means it won't read the source spectrum, but instead the background spectrum
            self._read_on_demand(False)

        return Quantity([self._back_counts, np.sqrt(self._back_counts)], 'ct').T

    @property
    def back_exposure(self) -> Quantity:
        """
        The weighted (from individual exposure times of different CCD chips) exposure time of the
        background spectrum, as used by XSPEC.

        :rtype: Quantity
        :return: The exposure time for the background spectrum, in units of seconds.
        """
        # Fetch the exposure time from the header and create a quantity
        exp_time = Quantity(float(self.back_header['EXPOSURE']), 's')
        return exp_time

    @property
    def back_count_rates(self) -> Quantity:
        """
        The array of counts/second associated with each channel of the background spectrum. This takes the
        back_counts property and divides it by the EXPOSURE entry in the background spectrum header. A second column
        containing uncertainty is included.

        :rtype: Quantity
        :return: The counts/second quantity in units of 'ct/s'.
        """

        return self.back_counts / self.back_exposure

    @property
    def back_channels(self) -> Quantity:
        """
        The array of instrument channels in the background spectrum.

        :rtype: Quantity
        :return: An array of channels.
        """
        # Checks whether the initial value of the background channels attribute has been overwritten, if not then I run
        #  the read on demand method to grab the information from the file.
        if self._back_channels is None:
            # Passing false means it won't read the source spectrum, but instead the background spectrum
            self._read_on_demand(False)

        return Quantity(self._back_channels)

    @property
    def back_grouping(self) -> np.ndarray:
        """
        The grouping information from the background spectrum

        :rtype: np.ndarray
        :return: An array of group IDs.
        """
        # Checks whether the initial value of the attribute has been overwritten, if not then I run
        #  the read on demand method to grab the information from the file.
        if self._back_group is None:
            # Passing false means it won't read the source spectrum, but instead the background spectrum
            self._read_on_demand(False)

        return self._back_group

    @property
    def back_quality(self) -> np.ndarray:
        """
        The quality information from the background spectrum. A 0 flag value means good quality, 1 means
        not good quality.

        :rtype: np.ndarray
        :return: An array of quality flags.
        """
        # Checks whether the initial value of the attribute has been overwritten, if not then I run
        #  the read on demand method to grab the information from the file.
        if self._back_quality is None:
            self._read_on_demand(False)

        return self._back_quality

    @property
    def eff_area(self) -> Quantity:
        """
        The discrete effective area curve of the telescope. These area values correspond to upper and lower energy
        bounds that can be accessed using the eff_area_lo_en and eff_area_hi_en properties

        :rtype: Quantity
        :return: A quantity containing the effective area values in units of cm^-2.
        """
        # Checking whether the data have been read in yet, if not then do so
        if self._arf_eff_area is None:
            self._read_response_on_demand(rmf=False)

        return self._arf_eff_area

    @property
    def eff_area_lo_en(self) -> Quantity:
        """
        The lower energy bounds for the effective area curve.

        :rtype: Quantity
        :return: A quantity containing the lower energy bounds for effective area values in units keV.
        """
        # Checking whether the data have been read in yet, if not then do so
        if self._arf_lo_en is None:
            self._read_response_on_demand(rmf=False)

        return self._arf_lo_en

    @property
    def eff_area_hi_en(self) -> Quantity:
        """
        The upper energy bounds for the effective area curve.

        :rtype: Quantity
        :return: A quantity containing the upper energy bounds for effective area values in units keV.
        """
        # Checking whether the data have been read in yet, if not then do so
        if self._arf_hi_en is None:
            self._read_response_on_demand(rmf=False)

        return self._arf_hi_en

    @property
    def rmf_channels(self) -> np.ndarray:
        """
        The channels present in the RMF EBOUND table, used for converting between channel and energy when used in
        conjunction with rmf_channels_lo_en and rmf_channels_hi_en.

        :rtype: np.ndarray
        :return: A numpy array containing the channels in the EBOUND table.
        """
        # Checking whether the data have been read in yet, if not then do so
        if self._rmf_channels is None:
            self._read_response_on_demand(rmf=True)

        return self._rmf_channels

    @property
    def rmf_channels_lo_en(self) -> Quantity:
        """
        The lower energy bounds for the RMF EBOUND channels.

        :rtype: Quantity
        :return: A quantity containing the lower energy bounds for RMF channels, in units keV.
        """
        # Checking whether the data have been read in yet, if not then do so
        if self._rmf_channels_lo_en is None:
            self._read_response_on_demand(rmf=True)

        return self._rmf_channels_lo_en

    @property
    def rmf_channels_hi_en(self) -> Quantity:
        """
        The upper energy bounds for the RMF EBOUND channels.

        :rtype: Quantity
        :return: A quantity containing the upper energy bounds for RMF channels, in units keV.
        """
        # Checking whether the data have been read in yet, if not then do so
        if self._rmf_channels_hi_en is None:
            self._read_response_on_demand(rmf=True)

        return self._rmf_channels_hi_en

    @property
    def rmf_redist_matrix(self) -> np.ndarray:
        """
        The contents of the RMF MATRIX table.

        :rtype: np.ndarray
        :return: A numpy array with names columns.
        """
        return self._redist_matrix_info

    @property
    def path(self) -> str:
        """
        This method returns the path to the spectrum file of this object.

        :return: The path to the spectrum file associated with this object.
        :rtype: str
        """
        return self._path

    @path.setter
    def path(self, new_path: str):
        """
        This setter updates the path to the spectrum file, and then updates that file with the current values of
        the RMF, ARF, and background spectrum paths. WARNING: This does permanently alter the file, so use your
        own spectrum file with caution.

        :param str new_path: The updated path to the spectrum file.
        """
        if os.path.exists(new_path):
            self._path = new_path
            # Call this here because it'll replace any existing arf and rmf file paths with the ones
            #  currently loaded in the instance of this object.
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new spectrum file does not exist")

    @property
    def rmf(self) -> str:
        """
        This method returns the path to the RMF file of the main spectrum of this object.

        :return: The path to the RMF file associated with the main spectrum of this object.
        :rtype: str
        """
        return self._rmf

    @rmf.setter
    def rmf(self, new_path: str):
        """
        This setter updates the path to the main RMF file, then writes that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!

        :param str new_path: The path to the new RMF file.
        """
        if os.path.exists(new_path):
            self._rmf = new_path
            # Push to the actual file
            self._update_spec_headers("main")
            # If the data from an RMF have been loaded already then we make them go away
            self._redist_matrix_info = None
            self._rmf_channels = None
            self._rmf_channels_lo_en = None
            self._rmf_channels_hi_en = None
        else:
            raise FileNotFoundError("The new RMF file does not exist")

    @property
    def arf(self) -> str:
        """
        This method returns the path to the ARF file of the main spectrum of this object.

        :return: The path to the ARF file associated with the main spectrum of this object.
        :rtype: str
        """
        return self._arf

    @arf.setter
    def arf(self, new_path: str):
        """
        This setter updates the path to the main ARF file, then writes that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!

        :param str new_path: The path to the new ARF file.
        """
        if os.path.exists(new_path):
            self._arf = new_path
            self._update_spec_headers("main")
            # If ARF data has been loaded into memory then we must remove it
            self._arf_eff_area = None
            self._arf_lo_en = None
            self._arf_hi_en = None
        else:
            raise FileNotFoundError("The new ARF file does not exist")

    @property
    def background(self) -> str:
        """
        This method returns the path to the background spectrum.

        :return: Path of the background spectrum.
        :rtype: str
        """
        return self._back_spec

    @background.setter
    def background(self, new_path: str):
        """
        This method is the setter for the background spectrum. It can be used to change the background
        spectrum file associated with this object, and will write that change to the actual spectrum file.
        WARNING: This permanently alters the file, use with caution!

        :param str new_path: The path to the new background spectrum.
        """
        if os.path.exists(new_path):
            self._back_spec = new_path
            self._update_spec_headers("main")
        else:
            raise FileNotFoundError("The new background spectrum file does not exist")

    @property
    def background_rmf(self) -> str:
        """
        This method returns the path to the background spectrum's RMF file.

        :return: The path the the background spectrum's RMF.
        :rtype: str
        """
        return self._back_rmf

    @background_rmf.setter
    def background_rmf(self, new_path: str):
        """
        This setter method will change the RMF associated with the background spectrum, then write
        that change to the background spectrum file.

        :param str new_path: The path to the background spectrum's new RMF.
        """
        if os.path.exists(new_path):
            self._back_rmf = new_path
            self._update_spec_headers("back")
        else:
            raise FileNotFoundError("That new background RMF file does not exist")

    @property
    def background_arf(self) -> str:
        """
        This method returns the path to the background spectrum's ARF file.

        :return: The path the the background spectrum's ARF.
        :rtype: str
        """
        return self._back_arf

    @background_arf.setter
    def background_arf(self, new_path: str):
        """
        This setter method will change the ARF associated with the background spectrum, then write
        that change to the background spectrum file.

        :param str new_path: The path to the background spectrum's new ARF.
        """
        if os.path.exists(new_path):
            self._back_arf = new_path
            self._update_spec_headers("back")
        else:
            raise FileNotFoundError("That new background ARF file does not exist")

    @property
    def storage_key(self) -> str:
        """
        This property returns the storage key which this object assembles to place the Spectrum in
        an XGA source's storage structure. The key is based on the properties of the spectrum, and
        some of the configuration options, and is basically human readable.

        :return: String storage key.
        :rtype: str
        """
        return self._storage_key

    @property
    def central_coord(self) -> Quantity:
        """
        This property provides the central coordinates (RA-Dec) of the region that this spectrum
        was generated from.

        :return: Astropy quantity object containing the central coordinate in degrees.
        :rtype: Quantity
        """
        return self._central_coord

    @property
    def shape(self) -> str:
        """
        Returns the shape of the outer edge of the region this spectrum was generated from.

        :return: The shape (either circular or elliptical).
        :rtype: str
        """
        return self._shape

    @property
    def inner_rad(self) -> Quantity:
        """
        Gives the inner radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this spectrum was generated.

        :return: The inner radius(ii) of the region.
        :rtype: Quantity
        """
        return self._inner_rad

    @property
    def outer_rad(self):
        """
        Gives the outer radius (if circular) or radii (if elliptical - semi-major, semi-minor) of the
        region in which this spectrum was generated.

        :return: The outer radius(ii) of the region.
        :rtype: Quantity
        """
        return self._outer_rad

    @property
    def grouped(self) -> bool:
        """
        A property stating whether SAS was told to group this spectrum during generation or not.

        :return: Boolean variable describing whether the spectrum is grouped or not
        :rtype: bool
        """
        return self._grouped

    @property
    def grouped_on(self) -> str:
        """
        A property stating what metric this spectrum was grouped on.

        :return: String representation of the metric this spectrum was grouped on (None if not grouped).
        :rtype: str
        """
        return self._grouped_on

    @property
    def min_counts(self) -> int:
        """
        A property stating the minimum number of counts allowed in a grouped channel.

        :return: The integer minimum number of counts per grouped channel (if this spectrum was grouped on
            minimum numbers of counts).
        :rtype: int
        """
        return self._min_counts

    @property
    def min_sn(self) -> Union[float, int]:
        """
        A property stating the minimum signal to noise allowed in a grouped channel.

        :return: The minimum signal to noise per grouped channel (if this spectrum was grouped on
            minimum signal to noise).
        :rtype: Union[float, int]
        """
        return self._min_sn

    @property
    def over_sample(self) -> float:
        """
        A property string stating the amount of oversampling applied by evselect during the spectrum
        generation process.

        :return: Oversampling applied during generation
        :rtype: float
        """
        return self._over_sample

    @property
    def region(self) -> bool:
        """
        This property states whether this spectrum was generated directly from a region file
        region or not. If true then this isn't from any arbitrary radii or an overdensity radius, but
        instead directly from a source finder.

        :return: A boolean flag describing if this is a region spectrum or not.
        :rtype: bool
        """
        return self._region

    @property
    def annulus_ident(self) -> int:
        """
        This property returns the integer identifier of which annulus in a set this Spectrum is, if it
        is part of a set.

        :return: Integer annulus identifier, None if not part of a set.
        :rtype: object
        """
        return self._ann_ident

    @annulus_ident.setter
    def annulus_ident(self, new_ident: int):
        """
        This property sets the annulus identifier of this object.

        :param int new_ident: The annulus integer identifier of this spectrum.
        """
        if not isinstance(new_ident, int):
            raise TypeError("Spectrum annulus identifiers may ONLY be positive integers")
        self._ann_ident = new_ident

    @property
    def set_ident(self) -> int:
        """
        This property returns the random id of the spectrum set this is a part of.

        :return: Set identifier, None if not part of a set.
        :rtype: int
        """
        return self._set_ident

    @set_ident.setter
    def set_ident(self, new_ident: int):
        """
        This property sets the set identifier of this object.

        :param int new_ident: The set identifier of this spectrum.
        """
        if not isinstance(new_ident, int):
            raise TypeError("Spectrum set identifiers may ONLY be positive integers")
        self._set_ident = new_ident

    @property
    def fitted_models(self) -> list:
        """
        A property that gets the list of spectral models that have been fit to this spectrum instance.

        :return: A list of models fit to this spectrum.
        :rtype: list
        """
        return list(self._plot_data.keys())

    @property
    def fitted_model_configurations(self) -> dict:
        """
        Property that returns a dictionary with model names as keys and lists of fit configuration identifiers as
        values, for the models that have been fit to this spectrum.

        :return: Dictionary with model names as keys, and lists of model configuration identifiers as values.
        :rtype: dict
        """
        return {m: list(self._plot_data[m].keys()) for m in self.fitted_models}

    @property
    def fitted_model_configuration_diffs(self) -> dict:
        """
        Property that returns the difference of each fitted model configuration from the default for that particular
        model - making it easier to identify only those parameters that were altered.

        :return: Dictionary with model names as keys, fit configuration identifiers as lower level keys, and
            dictionaries of parameters-changed-from-default as values.
        :rtype: dict
        """
        from ..xspec.fit import FIT_FUNC_MODEL_NAMES
        from ..xspec.fitconfgen import fit_conf_from_function, FIT_FUNC_ARGS

        diffs = {}
        for mod in self.fitted_models:
            diffs.setdefault(mod, {})
            fit_func = FIT_FUNC_MODEL_NAMES[mod]
            def_fit_conf = fit_conf_from_function(fit_func)

            mod_args = [in_arg for in_arg in FIT_FUNC_ARGS[fit_func.__name__]
                        if FIT_FUNC_ARGS[fit_func.__name__][in_arg]]

            for cur_fit_conf in self.fitted_model_configurations[mod]:
                diffs[mod][cur_fit_conf] = {}
                for par in cur_fit_conf.split('_'):
                    if par not in def_fit_conf:
                        # We are trying to split the fitconf key into parname and value - but there is no easy way to
                        #  do that without knowing the parname. Thus we identify candidates (candidates because it
                        #  is conceivable that there are parnames for the function that are substrings of each other),
                        #  and then split on those names, determining which results in the shortest string value (which
                        #  would be the correct name)
                        cands = {in_arg: par.split(in_arg.replace("_", ''))[-1] for in_arg in mod_args
                                 if in_arg.replace("_", '') in par}
                        if len(cands) != 0:
                            chos_arg = np.argmin(np.array([len(val) for val in list(cands.values())]))
                            final_par = np.array(list(cands.keys()))[chos_arg]
                            final_val = np.array(list(cands.values()))[chos_arg]

                            diffs[mod][cur_fit_conf][final_par] = final_val

        return diffs

    def add_fit_data(self, model: str, tab_line, plot_data: hdu.table.TableHDU, fit_conf: str):
        """
        Method that adds information specific to a spectrum from an XSPEC fit to this object. This includes
        individual spectrum exposure and count rate, as well as calculated luminosities, and plotting
        information for data and model.

        :param str model: String representation of the XSPEC model fitted to the data.
        :param tab_line: The line of the SPEC_INFO table produced by xga_extract.tcl that is relevant to this
            spectrum object.
        :param hdu.table.TableHDU plot_data: The PLOT{N} table in the file produced by xga_extract.tcl that is
            relevant to this spectrum object.
        :param str fit_conf: In order to be able to store results for different fit configurations (e.g. different
            starting pars, abundance tables, all that), we need to have a key that identifies the configuration. We
            do not expect the user to be adding fit data, so this will be a key generated by the fit function
        """
        # This stores the exposure time that XSPEC uses for this specific spectrum.
        if self._exp is None:
            self._exp = float(tab_line["EXPOSURE"])

        # This is the count rate and error for this spectrum.
        self._count_rate.setdefault(model, {})
        self._count_rate[model][fit_conf] = [float(tab_line["COUNT_RATE"]), float(tab_line["COUNT_RATE_ERR"])]

        # Searches for column headers with 'Lx' in them (this has to be dynamic as the user can calculate
        #  luminosity in as many bands as they like)
        lx_inds = np.where(np.char.find(tab_line.dtype.names, "Lx") == 0)[0]
        lx_cols = np.array(tab_line.dtype.names)[lx_inds]

        # Constructs a dictionary of luminosities and their errors for the different energy bands
        #  in this XSPEC fit.
        lx_dict = {}
        for col in lx_cols:
            lx_info = col.split("_")
            if lx_info[2][-1] == "-" or lx_info[2][-1] == "+":
                en_band = "bound_{l}-{u}".format(l=lx_info[1], u=lx_info[2][:-1])
                err_type = lx_info[-1][-1]
            else:
                en_band = "bound_{l}-{u}".format(l=lx_info[1], u=lx_info[2])
                err_type = ""

            if en_band not in lx_dict:
                lx_dict[en_band] = [0, 0, 0]

            if err_type == "":
                lx_dict[en_band][0] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")
            elif err_type == "-":
                lx_dict[en_band][1] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")
            elif err_type == "+":
                lx_dict[en_band][2] = Quantity(float(tab_line[col])*(10**44), "erg s^-1")

        self._luminosities.setdefault(model, {})
        self._luminosities[model][fit_conf] = lx_dict
        self._plot_data.setdefault(model, {})
        self._plot_data[model][fit_conf] = {"x": plot_data["X"][:], "x_err": plot_data["XERR"][:],
                                            "y": plot_data["Y"][:], "y_err": plot_data["YERR"][:],
                                            "model": plot_data["YMODEL"][:]}

    def _get_fit_checks(self, model: str = None, fit_conf: Union[str, dict] = None) -> Tuple[str, str]:
        """
        An internal function to perform input checks and pre-processing for get methods that access fit results, or
        other related information such as fit statistic.

        :param str model: The name of the fitted model that you're requesting the results from
            (e.g. constant*tbabs*apec).
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: The model name and fit configuration.
        :rtype: Tuple[str, str]
        """
        from ..xspec.fit import FIT_FUNC_MODEL_NAMES
        from ..xspec.fitconfgen import fit_conf_from_function

        # It is possible to pass a null value for the 'model' parameter, but we'll only accept that if a single model
        #  has been fit to this spectrum - otherwise how are we to know which model they want?
        if len(self.fitted_models) == 0:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this Spectrum object.")
        elif model is None and len(self.fitted_models) != 1:
            av_mods = ", ".join(self.fitted_models)
            raise ValueError("Multiple models have been fit to this spectrum, so model=None is not "
                             "valid; available models are {a}".format(m=model, a=av_mods))
        elif model is None:
            # In this case there is ONE model fit, and the user didn't pass a model parameter value, so we'll just
            #  automatically select it for them
            model = self.fitted_models[0]
        elif model is not None and model not in self.fitted_models:
            av_mods = ", ".join(self.fitted_models)
            raise ModelNotAssociatedError("{m} has not been fitted to this Spectrum; available "
                                          "models are {a}".format(m=model, a=av_mods))

        # Checks the input fit configuration values - if they are completely illegal we throw an error
        if fit_conf is not None and not isinstance(fit_conf, (str, dict)):
            raise TypeError("'fit_conf' must be a string fit configuration key, or a dictionary with "
                            "changed-from-default fit function arguments as keys and changed values as items.")
        # If the input is a dictionary then we need to construct the key, as opposed to it being passed in whole
        #  as a string
        elif isinstance(fit_conf, dict):
            fit_conf = fit_conf_from_function(FIT_FUNC_MODEL_NAMES[model], fit_conf)
        elif isinstance(fit_conf, str) and fit_conf not in self.fitted_model_configurations[model]:
            av_fconfs = ", ".join(self.fitted_model_configurations[model])
            raise ModelNotAssociatedError("The {fc} fit configuration has not been used for any {m} fit to this "
                                          "spectrum; available fit configurations are "
                                          "{a}".format(fc=fit_conf, m=model, a=av_fconfs))
        # In this case the user passed no fit configuration key, but there are multiple fit configurations stored here
        elif fit_conf is None and len(self.fitted_model_configurations[model]) != 1:
            av_fconfs = ", ".join(self.fitted_model_configurations[model])
            raise ValueError("The {m} model has been fit with multiple configuration, so fit_conf=None is not "
                             "valid; available fit configurations are {a}".format(m=model, a=av_fconfs))
        # However here they passed no fit configuration, and only one has been used for the model, so we're all good
        #  and will select it for them
        elif fit_conf is None and len(self.fitted_model_configurations[model]) == 1:
            fit_conf = self.fitted_model_configurations[model][0]

        # # We also check that
        # if par is not None and par not in self._fit_results[annulus_ident][model]:
        #     av_pars = ", ".join(self._fit_results[annulus_ident][model].keys())
        #     raise ParameterNotAssociatedError("{p} was not a free parameter in the {m} fit to this AnnularSpectra; "
        #                                       "available parameters are {a}".format(p=par, m=model, a=av_pars))

        return model, fit_conf

    def get_luminosities(self, model: str = None, lo_en: Quantity = None, hi_en: Quantity = None,
                         fit_conf: Union[str, dict] = None):
        """
        Returns the luminosities measured for this spectrum from a given model.

        If no model name is supplied, but only one model has been fit to this spectrum, then that model
        will be automatically selected - this behavior also applies to the fit configuration (fit_conf) parameter; if
        a model was only fit with one fit configuration then that will be automatically selected.

        :param model: Name of model to fetch luminosities for.
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: Luminosity measurement, either for all energy bands, or the one requested with the energy
            limit parameters. Luminosity measurements are presented as three column numpy arrays, with column 0
            being the value, column 1 being err-, and column 2 being err+.
        """
        # Use the internal method to check the model name and fit configuration - populating them if they are None
        #  and only one model and/or one configuration of that model has been fit
        model, fit_conf = self._get_fit_checks(model, fit_conf)

        # Checking the input energy limits are valid, and assembles the key to look for lums in those energy
        #  bounds. If the limits are none then so is the energy key
        if lo_en is not None and hi_en is not None and lo_en > hi_en:
            raise ValueError("The low energy limit cannot be greater than the high energy limit")
        elif lo_en is not None and hi_en is not None:
            en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        else:
            en_key = None

        # Checks that the requested energy band actually exists
        if en_key is not None and en_key not in self._luminosities[model]:
            av_bands = ", ".join([en.split("_")[-1] + "keV" for en in self._luminosities[model].keys()])
            raise ParameterNotAssociatedError("{l}-{u}keV was not an energy band for the fit with {m}; available "
                                              "energy bands are {b}".format(l=lo_en.to("keV").value,
                                                                            u=hi_en.to("keV").value,
                                                                            m=model, b=av_bands))

        if en_key is None:
            return self._luminosities[model][fit_conf]
        else:
            return self._luminosities[model][fit_conf][en_key]

    def get_rate(self, model: str = None, fit_conf: Union[str, dict] = None) -> Quantity:
        """
        Fetches the count rate for a particular model fitted to this spectrum.

        If no model name is supplied, but only one model has been fit to this spectrum, then that model
        will be automatically selected - this behavior also applies to the fit configuration (fit_conf) parameter; if
        a model was only fit with one fit configuration then that will be automatically selected.

        :param model: The model to fetch count rate for.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: Count rate in counts per second.
        :rtype: Quantity
        """
        model, fit_conf = self._get_fit_checks(model, fit_conf)
        rate = Quantity(self._count_rate[model][fit_conf], 'ct/s')

        return rate

    # TODO Should this take parameter values as arguments too? - It definitely should
    def add_conv_factors(self, lo_ens: np.ndarray, hi_ens: np.ndarray, rates: np.ndarray,
                         lums: np.ndarray, model: str):
        """
        Method used to store countrate to luminosity conversion factors derived from fakeit spectra, as well as
        the actual countrate and luminosity measured in case the user wants to create a combined factor for multiple
        observations.

        :param np.ndarray lo_ens: A numpy array of string representations of the lower energy bounds for the cntrate
            and luminosity measurements.
        :param np.ndarray hi_ens: A numpy array of string representations of the upper energy bounds for the cntrate
            and luminosity measurements.
        :param np.ndarray rates: A numpy array of the rates measured for this arf/rmf combination for the energy
            ranges specified in lo_ens and hi_end.
        :param np.ndarray lums: A numpy array of the luminosities measured for this arf/rmf combination
            for the energy ranges specified in lo_ens and hi_end.
        :param str model: The name of the model used to calculate this factor.
        """
        for row_ind, lo_en in enumerate(lo_ens):
            # Define the key with energy information under which to store this information
            hi_en = hi_ens[row_ind]
            en_key = "bound_{l}-{u}".format(l=lo_en, u=hi_en)

            # Split out the rate and lum for this particular set of energy limits
            rate = Quantity(rates[row_ind], "ct/s")
            lum = Quantity(lums[row_ind], "10^44 erg/s")

            # Will be storing the individual components, but will also store the factor for this spectrum
            factor = lum / rate

            if model not in self._conv_factors:
                self._conv_factors[model] = {}

            self._conv_factors[model][en_key] = {"rate": rate, "lum": lum, "factor": factor}

    def get_conv_factor(self, lo_en: Quantity, hi_en: Quantity, model: str) -> Tuple[Quantity, Quantity, Quantity]:
        """
        Retrieves a conversion factor between count rate and luminosity for a given energy range, if one
        has been calculated.

        :param Quantity lo_en: The lower energy bound for the desired conversion factor.
        :param Quantity hi_en: The upper energy bound for the desired conversion factor.
        :param str model: The model used to generate the desired conversion factor.
        :return: The conversion factor, luminosity, and rate for the supplied model-energy combination.
        :rtype: Tuple[Quantity, Quantity, Quantity]
        """
        en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        if model not in self._conv_factors:
            mods = ", ".join(list(self._conv_factors.keys()))
            raise ModelNotAssociatedError("{0} is not associated with this spectrum, only {1} "
                                          "are available.".format(model, mods))
        elif en_key not in self._conv_factors[model]:
            raise ParameterNotAssociatedError("The conversion factor for {m} in {l}-{u}keV has not been "
                                              "calculated".format(m=model, l=lo_en.to("keV").value,
                                                                  u=hi_en.to("keV").value))

        rel_vals = self._conv_factors[model][en_key]
        return rel_vals["factor"], rel_vals["lum"], rel_vals["rate"]

    def get_plot_data(self, model: str = None, fit_conf: Union[str, dict] = None) -> dict:
        """
        Simply grabs the plot data dictionary for a given model, if the spectrum has had a fit performed on it.

        If no model name is supplied, but only one model has been fit to this spectrum, then that model
        will be automatically selected - this behavior also applies to the fit configuration (fit_conf) parameter; if
        a model was only fit with one fit configuration then that will be automatically selected.

        :param str model: The model for which the plotting data is to be retrieved. Default is None, which will
            automatically be set to the model name IF only one model has been fit.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: All information required to plot the data and model.
        :rtype: dict
        """
        model, fit_conf = self._get_fit_checks(model, fit_conf)

        return self._plot_data[model][fit_conf]

    def conv_channel_energy(self, to_convert: Quantity) -> Quantity:
        """
        This method will use RMF information to convert between channels and energy, and vice versa. If converting
        from channel to energy, the return will be the midpoint of the energy bin associated with that channel.

        :param Quantity to_convert: The input Quantity, either with units of energy, or no units (which will be
            taken to mean channels). Any channel input will be converted to an integer, floats are not valid.
        :rtype: Quantity
        :return: The converted value, either as a channel (for energy input) or an energy (for channel input).
        """
        if not to_convert.unit.is_equivalent("keV") and to_convert.unit != "":
            raise UnitConversionError("The to_convert argument must be in units of energy (e.g. convertible to keV) or "
                                      "unitless (for instance Quantity(50) would be a unitless quantity).")
        elif to_convert.unit.is_equivalent('keV'):
            # If only one value is passed in then we make it iterable for now because its easier to write the
            #  rest of the code - I'll change it back at the end
            if to_convert.isscalar:
                ens = Quantity([to_convert])
            else:
                ens = to_convert

            # Checking that energies are within a valid range
            if not np.all((ens > self.rmf_channels_lo_en.min()) & (ens <= self.rmf_channels_hi_en.max())):
                raise UnitConversionError("Some energies in to_convert are outside the range of this spectrum's "
                                          "instrument, please enter values between {l} "
                                          "and {u}".format(l=self.rmf_channels_lo_en.min(),
                                                           u=self.rmf_channels_hi_en.max()))
            # Finding the index of the energy bin that brackets the input energy value(s)
            rel_inds = np.array([np.where((e > self.rmf_channels_lo_en) & (e <= self.rmf_channels_hi_en))[0][0]
                                 for e in ens])
            # Getting the equivelant channel
            converted_vals = Quantity(self.rmf_channels[rel_inds])

            # Returning the result to a single value if it was passed in as one
            if len(converted_vals) == 1:
                converted_vals = converted_vals[0]

        elif to_convert.unit == "":
            # It is unphysical for channels to be anything other than integers, also want it to be iterable
            if to_convert.isscalar:
                int_chans = np.array([to_convert.value.astype(int)])
            else:
                int_chans = to_convert.value.astype(int)

            # Checking that channels are within a valid range
            test_in = np.isin(int_chans, self.rmf_channels)
            if not all(test_in):
                raise UnitConversionError("Some channels in to_convert are outside the range of this spectrum's "
                                          "instrument, please enter values between {l} and "
                                          "{u}".format(l=self.rmf_channels.min(), u=self.rmf_channels.max()))

            # I would rather use some clever numpy solution, but this will do for now. Although I am almost certain
            #  that channels and indices of self.rmf_channels will always be the same I can't guarantee that channels
            #  will never start at 1 or something, so I'm doing it with a list comprehension as the easiest way to
            #  preserve the order of the input channels
            rel_inds = np.array([np.where(self.rmf_channels == c)[0][0] for c in int_chans])
            rel_lo_ens = self.rmf_channels_lo_en[rel_inds]
            rel_hi_ens = self.rmf_channels_hi_en[rel_inds]
            # Calculate the midpoint of the energy bins
            converted_vals = (rel_hi_ens + rel_lo_ens)/2
            # If only one value was passed in then we change our array back into a single quantity
            if len(converted_vals) == 1:
                converted_vals = converted_vals[0]

        return converted_vals

    def get_grouped_data(self, count_rate: bool = True) -> Tuple[Quantity]:
        """
        In many cases a spectrum is 'grouped' after generation, which involves combining sequential channels to
        increase the signal-to-noise. This method reads any grouping information in the spectrum associated with
        this object and returns the grouped data, along with everything necessary to use it. The properties that
        return counts, energy bins etc all give the raw data, unlike this method.

        The spectrum quality information is used to filter the spectrum before grouping is applied, only channels with
        a quality flag of 0 are accepted.

        :param bool count_rate: Should the grouped spectrum data be returned as a count-rate, default is True. If
            set to False then grouped data will be returned as counts.
        :rtype: Tuple[Quantity, Quantity, Quantity, Quantity, Quantity, Quantity]
        :return: The source count-rates (or counts) with uncertainties, the background count-rates (or counts) with
            uncertainties, the lower energy bounds of the groups, the upper energy bounds of the groups, the channel
            midpoints of the groups (with width in a second column), the energy midpoints of the groups (with width
            in a second column).
        """
        # Check whether this spectrum was actually grouped on generation
        if not self.grouped:
            raise ValueError("This spectrum was generated without grouping, and so you cannot "
                             "retrieve grouped data.")

        # This is the mask which accepts only those channels with a quality flag of zero.
        notice = self.quality == 0
        # This copies the raw counts, and then masks them to take only those with an acceptable quality
        start_cnt = self.counts.copy()[notice]
        # Then the same process is followed for background counts, grouping information, channels, and energies
        start_bck_cnt = self.back_counts.copy()[notice]
        grouping = self.grouping.copy()[notice]
        start_chans = self.channels.copy()[notice]
        lo_energies = self.rmf_channels_lo_en.copy()[notice]
        hi_energies = self.rmf_channels_hi_en.copy()[notice]

        # As the beginning of a group is indicated by a 1, we assemble a list of indexes where the grouping
        #  value is 1. After that -1 values indicate that the channel belongs to a group, but we won't actually need
        #  to use them
        grp_bnds = np.where(grouping == 1)[0]
        # As I am setting these up as boundaries, I append the length of the grouping array as a final boundary
        grp_bnds = np.append(grp_bnds, len(grouping))

        # Setting up empty quantities to store the grouped counts in later, both source and background
        src_grpd_cnts = Quantity(np.zeros(len(grp_bnds) - 1), 'ct')
        bck_grpd_cnts = Quantity(np.zeros(len(grp_bnds) - 1), 'ct')

        # Setting up empty quantities to store the lower and upper energy bounds for the groups
        bins_lo_en = Quantity(np.zeros((len(grp_bnds) - 1)), 'keV')
        bins_hi_en = Quantity(np.zeros((len(grp_bnds) - 1)), 'keV')
        # Empty quantities (with two columns now) to store central energy and channel values, along with
        #  an 'uncertainty' column that gives the width of the group
        en_cens = Quantity(np.zeros((len(grp_bnds) - 1, 2)), 'keV')
        chans = Quantity(np.zeros((len(grp_bnds) - 1, 2)))

        # I really did try to figure out a more efficient way to do this, but this is what I'm resorting to for now
        for grp_start_ind, grp_start in enumerate(grp_bnds[:-1]):
            # We have the index of the start of the current group in the grp_start variable, but we also need to be
            #  able to define the end of the group. As we are going to be slicing arrays in this loop (for which the
            #  second index is non-inclusive), I just define the start of the next group to use as the end of this one
            next_grp_start = grp_bnds[grp_start_ind + 1]

            # Simply grab the channels which are a part of the current group (both for the source and background
            #  spectra), and then add the raw counts together to retrieve the start and background counts for
            #  this group
            src_grpd_cnts[grp_start_ind] = start_cnt[grp_start: next_grp_start].sum()
            bck_grpd_cnts[grp_start_ind] = start_bck_cnt[grp_start: next_grp_start].sum()

            # Finding the lower and upper energy bounds of the group
            bins_lo_en[grp_start_ind] = lo_energies[grp_start]
            bins_hi_en[grp_start_ind] = hi_energies[next_grp_start - 1]

            # Finding the start and end channel values in the group
            cur_chan = start_chans[grp_start]
            next_chan = start_chans[next_grp_start - 1]

            # If the 'group' is actually just made up of one channel then we just put that current channel
            #  in the quantity, with a width of 0
            if cur_chan == next_chan:
                chans[grp_start_ind] = Quantity([cur_chan, 0.5])
            # Otherwise we calculate the midpoint of the group, and the width
            else:
                mid = (cur_chan + next_chan) / 2
                width = (next_chan - cur_chan) / 2
                chans[grp_start_ind, :] = Quantity([mid, width])

            # Always calculate the energy midpoint and width of a group, even if there is only one entry, because
            #  we know the upper and lower energy bounds of every channel from the RMF
            en_mid = (bins_lo_en[grp_start_ind] + bins_hi_en[grp_start_ind])/2
            en_width = (bins_hi_en[grp_start_ind] - bins_lo_en[grp_start_ind])/2
            en_cens[grp_start_ind] = Quantity([en_mid, en_width])

        src_grpd_cnts = Quantity([src_grpd_cnts, Quantity(np.sqrt(src_grpd_cnts.value), 'ct')]).T
        bck_grpd_cnts = Quantity([bck_grpd_cnts, Quantity(np.sqrt(bck_grpd_cnts.value), 'ct')]).T

        # Simple enough, if the user wants a count rate then divide by exposure, if just counts then don't
        if count_rate:
            src_grpd = src_grpd_cnts / self.exposure
            bck_grpd = bck_grpd_cnts / self.exposure
        else:
            src_grpd = src_grpd_cnts
            bck_grpd = bck_grpd_cnts

        # Return all the useful information that we have calculated.
        return src_grpd, bck_grpd, bins_lo_en, bins_hi_en, chans, en_cens

    def view_arf(self, figsize: Tuple = (8, 6), xscale: str = 'log', yscale: str = 'linear',
                 lo_en: Quantity = Quantity(0.01, 'keV'), hi_en: Quantity = Quantity(16.0, 'keV')):
        """
        Plots the response curve for this spectrum.

        :param tuple figsize: The desired size of the output figure.
        :param str xscale: The xscale to use for the plot.
        :param str yscale: The yscale to use for the plot.
        :param Quantity lo_en: The lower energy limit for the x-axis. The default is 0.01 keV. This will be altered
            to reflect the minimum value of the energy scale for this curve if lo_en is smaller than the lowest
            energy bin.
        :param Quantity hi_en: The upper energy limit for the x-axis. The default is 16.0 keV. This will be altered
            to reflect the maximum value of the energy scale for this curve if hi_en is greater than the highest
            energy bin.
        """
        # Calculate the energy values by finding the midpoints of the bins - this is done up here so that the minimum
        #  and maximum values of energy (lo_en and hi_en) passed by the user can be set to minimum and maximum
        #  values available if they are higher or lower
        ens = (self.eff_area_hi_en+self.eff_area_lo_en)/2

        # Have to check the input energy bounds to make sure that they are sensible
        if lo_en >= hi_en:
            raise ValueError("The 'hi_en' argument cannot be greater than or equal to the 'lo_en' argument.")
        else:
            lo_en = lo_en.to("keV").value
            hi_en = hi_en.to("keV").value

        # We dynamically alter the upper and lower energy bounds passed by the user to reflect the actual maxima and
        #  minima of the energy scale, so there isn't a ton of blank space in on either side if they made a bad
        #  choice, or my default value choices were bad
        if lo_en < ens.value.min():
            lo_en = ens.value.min()
        if hi_en > ens.value.max():
            hi_en = ens.value.max()

        # As we know, log scales don't like zero values, so I make a change if the user has set the minimum
        #  energy to be zero and set an x-axis log scale
        if xscale == 'log' and lo_en == 0:
            warn("The x-axis scale has been set to log, and 'lo_en' cannot be zero - it has been set to 0.01 "
                 "keV.", stacklevel=2)
            lo_en = 0.01

        plt.figure(figsize=figsize)
        # Set the plot up to look nice and professional.
        ax = plt.gca()
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # Get the data and plot it - we select the data to plot first because then matplotlib selects a nice upper
        #  y-axis limit for us
        sel_ens = (ens.value >= lo_en) & (ens.value <= hi_en)
        plt.plot(ens[sel_ens], self.eff_area[sel_ens], color='black')

        # Set the lower y-lim to be 1, and then the user supplied x-lims (supplementing the fact that we've already
        #  used those limits to select the data to plot
        plt.ylim(1)
        plt.xlim(lo_en, hi_en)

        # Set the user defined x and y scales
        plt.xscale(xscale)
        plt.yscale(yscale)

        # Title and axis labels
        plt.ylabel("Effective Area [cm$^{2}$]", fontsize=12)
        plt.xlabel("Energy [keV]", fontsize=12)
        plt.title("{o}-{i} Sensitivity Curve".format(o=self.obs_id, i=self.instrument.upper()), fontsize=14)

        # This makes sure that the tick labels are formatted as 0.1, 1, 10, etc. keV on the x-axis (if logged) and
        #  10, 100, 100, 1000, etc. cm^2 on the y-axis if logged
        ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

        # Aaaand finally actually plot it
        plt.tight_layout()
        plt.show()

    def get_view(self, ax: Axes, lo_lim: Quantity = Quantity(0.3, "keV"), hi_lim: Quantity = Quantity(7.9, "keV"),
                 back_sub: bool = True, energy: bool = True, src_colour: str = 'black', bck_colour: str = 'firebrick',
                 grouped: bool = True, xscale: str = "log", yscale: str = "linear", fontsize: Union[int, float] = 14,
                 show_model_fits: bool = True, model: str = None, fit_conf: Union[str, dict] = None) -> Axes:
        """
        The method that creates and populates the view axes, separate from actual view so outside methods
        can add a view to other matplotlib axes.

        A spectrum can be viewed prior to fitting, and this method will produce plots that should be the same as the
        XSPEC count/s/keV (or channel) spectrum views. If a model has been fit, and the user wishes to display it, then
        the 'normalised count/s/keV' that are plotted are extracted from the XSPEC data, rather than assembled in this
        method.

        :param Axes ax: The matplotlib axes on which to show the spectrum.
        :param Quantity lo_lim: The lower limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 0.3 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param Quantity hi_lim: The upper limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 7.9 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param bool back_sub: Whether the plotted data should have their background subtracted, default is True.
        :param bool energy: Controls whether the x-axis is in units of energy, default is True. If False then
            channels are plotted instead.
        :param str src_colour: The colour in which to plot the source spectrum. Default is 'black'.
        :param str bck_colour: The colour in which to plot the background spectrum. Default is 'firebrick' red.
        :param bool grouped: Whether the grouped spectrum should be plotted, default is True. If the spectrum has not
            been grouped then this be automatically set to False.
        :param str xscale: The scaling to be applied to the x-axis, default is 'log'.
        :param str yscale: The scaling to be applied to the y-axis, default is 'linear'.
        :param int/float fontsize: The fontsize for axis labels. The legend fontsize will be fontsize - 1. The title
            fontsize will be fontsize + 1. Default is 14.
        :param bool show_model_fits: Whether any models fit to the spectrum by XSPEC should be shown. Default is
            True, but will be set to False if no fits have been performed.
        :param str model: This parameter allows you to specify a particular model to plot (if show_model_fits is
            True). Default is None, in which case all models will be shown (if available).
        :param str/dict fit_conf: This parameter allows you to specify a particular fit configuration of a model to
            plot (if 'show_model_fits' is True and 'model' is set). Pass either a dictionary with keys being the names
            of parameters passed to the XGA XSPEC fit function that were changed from default, and values being the
            changed values, or a full string representation of the fit configuration that is being requested. Default
            is None, in which case all fit configurations of a model will be plotted.
        """
        from ..xspec.fit import FIT_FUNC_MODEL_NAMES
        from ..xspec.fitconfgen import fit_conf_from_function

        # This just checks whether the grouped argument to this method is compatible with whether the spectrum
        #  associated with this Spectrum instance has actually been grouped - if not then we automatically
        #  set the method argument to False
        if not self.grouped:
            grouped = False

        # This just ensures that everything works if someone has passed an integer for the channel limits
        lo_lim = Quantity(lo_lim)
        hi_lim = Quantity(hi_lim)

        # Performing checks on the limits
        if lo_lim >= hi_lim:
            raise ValueError("The hi_lim argument cannot be less than or equal to the lo_lim argument")

        # These just make sure that limits in units of either channel or energy are converted appropriately to what
        #  we're plotting on the x-axis, channels or energies.
        if not energy and lo_lim.unit != '':
            lo_lim = self.conv_channel_energy(lo_lim)
        if not energy and hi_lim.unit != '':
            hi_lim = self.conv_channel_energy(hi_lim)
        if energy and not lo_lim.unit.is_equivalent('keV'):
            lo_lim = self.conv_channel_energy(lo_lim)
        if energy and not hi_lim.unit.is_equivalent('keV'):
            hi_lim = self.conv_channel_energy(hi_lim)

        # Reads out the values of the limits as matplotlib sometimes gets upset by astropy quantities
        if energy:
            lo_lim = lo_lim.to("keV").value
            hi_lim = hi_lim.to("keV").value
        else:
            lo_lim = lo_lim.value
            hi_lim = hi_lim.value

        # If the call to this method requested that models be plotted, we need to just make sure that there are models
        #  available, because the default is True, but we can look at spectra prior to fitting now
        if show_model_fits and len(self._plot_data) == 0:
            # If there are no model data to plot, we set this to False
            show_model_fits = False
        elif show_model_fits and not energy:
            raise ValueError("As fitted spectra are extracted from XSPEC, and only spectra with energy x-axes are "
                             "extracted, plotting against channel is not supported.")

        # Now we deal with the different models/model fit configurations that can and cannot be specified
        if show_model_fits and model is None and fit_conf is not None:
            raise ValueError("Specifying a fit configuration ('fit_conf') is not supported without setting the "
                             "'model' argument; use the 'fitted_model_configurations' property of this Spectrum to "
                             "see which models and configurations are available.")
        elif show_model_fits and model is None and fit_conf is None:
            model = self.fitted_models
            fit_conf = self.fitted_model_configurations
        elif show_model_fits and model is not None and fit_conf is None:
            # I indent this check because it is just a bit easier for me that way
            if model not in self.fitted_models:
                av_mods = ", ".join(self.fitted_models)
                raise ModelNotAssociatedError("{m} has not been fitted to this Spectrum; available "
                                              "models are {a}".format(m=model, a=av_mods))

            # If we're here then the model is valid, and in this case no fit configuration has been specified, so
            #  we grab ALL OF THEM - making sure that the structure of the parameters is the same (model in a list,
            #  fit configs in a list in a dictionary with model name as key
            fit_conf = {model: self.fitted_model_configurations[model]}
            model = [model]
        elif show_model_fits and model is not None and fit_conf is not None:
            if model not in self.fitted_models:
                av_mods = ", ".join(self.fitted_models)
                raise ModelNotAssociatedError("{m} has not been fitted to this Spectrum; available "
                                              "models are {a}".format(m=model, a=av_mods))

            # If the configuration is a dictionary we need to try to turn that into a proper fit configuration key
            if isinstance(fit_conf, dict):
                fit_conf = fit_conf_from_function(FIT_FUNC_MODEL_NAMES[model], fit_conf)

            # And now we check if the fit configuration is available to this Spectrum instance
            if fit_conf not in self.fitted_model_configurations[model]:
                av_fconfs = ", ".join(self.fitted_model_configurations[model])
                raise ModelNotAssociatedError("The {fc} fit configuration has not been used for any {m} fit to this "
                                              "spectrum; available fit configurations are "
                                              "{a}".format(fc=fit_conf, m=model, a=av_fconfs))

            fit_conf = {model: [fit_conf]}
            model = [model]

        # Here we grab the count-rates of the channels in this spectrum - either straight from the property
        #  or the get_grouped_data() method
        if not grouped:
            sct = self.count_rates.copy()
            bct = self.back_count_rates.copy()
            if energy:
                x_dat = self.conv_channel_energy(self.channels.copy()).value
                x_wid = (self.rmf_channels_hi_en - self.rmf_channels_lo_en).value
            else:
                x_dat = self.channels.copy()
                x_wid = 1
        else:
            grp_info = self.get_grouped_data()
            sct = grp_info[0]
            bct = grp_info[1]
            if energy:
                # This entry is the middle energy of each bin
                x_dat = grp_info[-1][:, 0].value
                # This entry is the 'error' (but really just half the width) of each energy bin
                x_wid = grp_info[-1][:, 1].value
            else:
                # This entry is the middle channel of each bin
                x_dat = grp_info[4][:, 0].value
                # This entry is the 'error' (but really just half the width) of each channel bin
                x_wid = grp_info[4][:, 1].value

        # We check that the x limits are actually sensible values, if they are higher (for the top limit) or lower (
        #  (for the lower limit) than the data that are actually available then we nudge them to those values
        if lo_lim < x_dat.min():
            lo_lim = x_dat.min()
        if hi_lim > x_dat.max():
            hi_lim = x_dat.max()

        # We pre-select the data based on the passed lower and upper limits - first making a selection mask array
        sel_x = (x_dat <= hi_lim) & (x_dat >= lo_lim)
        # Then selecting the relevant source count, background count, and x-data (energy or channel) entries
        sct = sct[sel_x]
        bct = bct[sel_x]
        x_dat = x_dat[sel_x]
        x_wid = x_wid[sel_x]
        # This is what the y-data are divided by to make it per keV or per channel, the width of the bin essentially
        per_x = x_wid * 2

        # This uses the AREASCAL keyword (the product of EXPOSURE times AREASCAL is the exposure duration for any
        #  fully exposed pixels in each channel - my experience is that this is normally 1 for XMM products) to
        #  effectively scale the exposure time by dividing the count rate by it
        src_rate = sct / self.header['AREASCAL']

        # This scales the background count rates by the AREASCAL (as above), but also by the ratio of BACKSCAL
        #  values, which scales the background flux to the same area as the source
        bck_rate = (self.header['BACKSCAL'] / self.back_header['BACKSCAL']) * (bct / self.back_header['AREASCAL'])

        # And finally subtracting one from the other - they both have error columns which are also subtracted
        #  here (which is completely meaningless of course), but don't worry we'll fix that on the next line!
        src_sub_bck_rate = src_rate - bck_rate
        # Simple error propagation to replace the nonsense uncertainty column in src_sub_bck_rate
        src_sub_bck_rate[:, 1] = np.sqrt(src_rate[:, 1] ** 2 + bck_rate[:, 1] ** 2)

        # Ensure axis is limited to the chosen energy range
        ax.set_xlim(lo_lim, hi_lim)

        # Set the plot up to look nice and professional.
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # This is an ugly way of doing this, but I hope that in the future I'll be able to implement this 'properly'
        #  and just undo this
        if not show_model_fits:
            # Plotting the data, accounting for the different combinations of x-axis and y-axis
            if back_sub:
                # If we're going for background subtracted data, then that is all we plot
                ax.errorbar(x_dat, src_sub_bck_rate.value[:, 0] / per_x, xerr=x_wid,
                            yerr=src_sub_bck_rate.value[:, 1] / per_x, fmt="+", color=src_colour,
                            label="Background subtracted source data", zorder=1)
            else:
                # But if we're not wanting background subtracted, we need to plot the source and background spectra
                ax.errorbar(x_dat, src_rate.value[:, 0] / per_x, xerr=x_wid, yerr=src_rate.value[:, 1] / per_x,
                            fmt="+",
                            color=src_colour, label="Source data", zorder=1)
                ax.errorbar(x_dat, bck_rate.value[:, 0] / per_x, xerr=x_wid, yerr=bck_rate.value[:, 1] / per_x,
                            fmt="x",
                            color=bck_colour, label="Background data", zorder=1)

            # Energy vs channel has already been encoded in the x data, but we still need to plot different axis labels
            if energy:
                ax.set_ylabel("Counts s$^{-1}$ keV$^{-1}$", fontsize=fontsize)
                ax.set_xlabel("Energy [keV]", fontsize=fontsize)
            else:
                ax.set_ylabel("Counts s$^{-1}$ Channel$^{-1}$", fontsize=fontsize)
                ax.set_xlabel("Channel", fontsize=fontsize)

        # In this case the user wants the fitted spectra, and there ARE fits to plot, so rather than plot our own
        #  calculated values we plot the normalised counts/s/keV (or channel) that were extracted from XSPEC
        else:
            # Set the axis labels
            ax.set_ylabel("Normalised Counts s$^{-1}$ keV$^{-1}$", fontsize=fontsize)
            ax.set_xlabel("Energy [keV]", fontsize=fontsize)

            plot_cnt = 0
            for mod in model:
                # We also iterate through the different fit configurations for the current model, and plot them
                #  separately - currently with the only the model name in the legend
                for fc in fit_conf[mod]:
                    cur_fit_data = self.get_plot_data(mod, fc)

                    # Extract the x values which we gathered from XSPEC (they will be in keV)
                    x = cur_fit_data["x"]
                    # Cut the x dataset to just the energy range we want
                    sel_x = (x > lo_lim) & (x < hi_lim)
                    plot_x = x[sel_x]

                    if plot_cnt == 0:
                        # Read out the data just for line length reasons
                        # Make the cuts based on energy values supplied to the view method
                        plot_y = cur_fit_data["y"][sel_x]
                        plot_xerr = cur_fit_data["x_err"][sel_x]
                        plot_yerr = cur_fit_data["y_err"][sel_x]
                        plot_mod = cur_fit_data["model"][sel_x]

                        ax.errorbar(plot_x, plot_y, xerr=plot_xerr, yerr=plot_yerr, fmt="k+",
                                    label="Background subtracted source data", zorder=1)
                        plot_cnt += 1
                    else:
                        # Don't want to re-plot data points as they should be identical, so if there is another model
                        #  only it will be plotted
                        plot_mod = cur_fit_data["model"][sel_x]

                    # The model line is put on
                    changed = self.fitted_model_configuration_diffs[mod][fc]
                    fc_str = "; ".join([par + "=" + val for par, val in changed.items()])
                    ax.plot(plot_x, plot_mod, label=mod + '; ' + fc_str, linewidth=1.5)

        # Setting up the scaling aspects of the plot
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

        return ax

    def view(self, figsize: Tuple = (10, 7), lo_lim: Quantity = Quantity(0.3, "keV"),
             hi_lim: Quantity = Quantity(7.9, "keV"), back_sub: bool = True, energy: bool = True,
             src_colour: str = 'black', bck_colour: str = 'firebrick', grouped: bool = True, xscale: str = "log",
             yscale: str = "linear", fontsize: Union[int, float] = 14, show_model_fits: bool = True,
             save_path: str = None, model: str = None, fit_conf: Union[str, dict] = None):
        """
        A method for viewing the data associated with this Spectrum instance.

        A spectrum can be viewed prior to fitting, and this method will produce plots that should be the same as the
        XSPEC count/s/keV (or channel) spectrum views. If a model has been fit, and the user wishes to display it, then
        the 'normalised count/s/keV' that are plotted are extracted from the XSPEC data, rather than assembled in this
        method.

        :param tuple figsize: The desired size of the output figure, default is (10, 7).
        :param Quantity lo_lim: The lower limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 0.3 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param Quantity hi_lim: The upper limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 7.9 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param bool back_sub: Whether the plotted data should have their background subtracted, default is True.
        :param bool energy: Controls whether the x-axis is in units of energy, default is True. If False then
            channels are plotted instead.
        :param str src_colour: The colour in which to plot the source spectrum. Default is 'black'.
        :param str bck_colour: The colour in which to plot the background spectrum. Default is 'firebrick' red.
        :param bool grouped: Whether the grouped spectrum should be plotted, default is True. If the spectrum has not
            been grouped then this be automatically set to False.
        :param str xscale: The scaling to be applied to the x-axis, default is 'log'.
        :param str yscale: The scaling to be applied to the y-axis, default is 'linear'.
        :param int/float fontsize: The fontsize for axis labels. The legend fontsize will be fontsize - 1. The title
            fontsize will be fontsize + 1. Default is 14.
        :param bool show_model_fits: Whether any models fit to the spectrum by XSPEC should be shown. Default is
            True, but will be set to False if no fits have been performed.
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        :param str model: This parameter allows you to specify a particular model to plot (if show_model_fits is
            True). Default is None, in which case all models will be shown (if available).
        :param str/dict fit_conf: This parameter allows you to specify a particular fit configuration of a model to
            plot (if 'show_model_fits' is True and 'model' is set). Pass either a dictionary with keys being the names
            of parameters passed to the XGA XSPEC fit function that were changed from default, and values being the
            changed values, or a full string representation of the fit configuration that is being requested. Default
            is None, in which case all fit configurations of a model will be plotted.
        """

        # Create figure object
        plt.figure(figsize=figsize)
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()

        ax = self.get_view(ax, lo_lim, hi_lim, back_sub, energy, src_colour, bck_colour, grouped, xscale, yscale,
                           fontsize, show_model_fits, model, fit_conf)

        # Set the title with all relevant information about the spectrum object in it
        ax.set_title("{n} - {o}{i} Spectrum".format(n=self.src_name, o=self.obs_id, i=self.instrument.upper()),
                     fontsize=fontsize + 1)

        # Generate the legend for the data and model(s)
        plt.legend(loc="best", fontsize=fontsize - 1)

        # Removing extraneous whitespace around the plot
        plt.tight_layout()

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            plt.savefig(save_path)

        # Display the spectrum
        plt.show()

        # Wipe the figure
        plt.close("all")


class AnnularSpectra(BaseAggregateProduct):
    """
    A class designed to hold a set of XGA spectra generated in concentric, circular annuli.

    :param List[Spectrum] spectra: A list of XGA spectrum objects which make up this set.
    """
    def __init__(self, spectra: List[Spectrum]):
        """
        The init method for the AnnularSpectrum class, performs checks and organises the spectra which
        have been passed in, for easy retrieval.
        """
        super().__init__([s.path for s in spectra], 'spectrum', "combined", "combined")

        # There shouldn't be any way this can happen, but it doesn't hurt to check that all of the spectra
        #  have the same set ID
        set_idents = set([s.set_ident for s in spectra])
        if len(set_idents) != 1:
            raise XGASetIDError("You have passed spectra that have set IDs that do not match")

        # Just put the set ID into an attribute in case anyone ever wants to know it
        self._set_id = list(set_idents)[0]

        # Here I run through all the spectra and access their annulus_ident property, that way we can determine how
        #  many annuli there are and start storing spectra appropriately
        uniq_ann_ids = list(set([s.annulus_ident for s in spectra]))
        if min(uniq_ann_ids) != 0 or max(uniq_ann_ids) != (len(uniq_ann_ids) - 1):
            raise ValueError("Some expected annulus IDs are missing from the spectra passed to this AnnularSpectra. "
                             "Spectra with IDs {p} have been "
                             "passed.".format(p=', '.join([str(i) for i in uniq_ann_ids])))
        # Now we've made certain that the input annuli IDs are making sense, we can just use the length of the
        #  list of unique IDs to set the number of annuli
        self._num_ann = len(uniq_ann_ids)

        # While the official ObsID and Instrument of this product are 'combined', I do still
        #  want to know which ObsIDs and instruments the spectra belong to
        inst_dict = {o: [] for o in [s.obs_id for s in spectra]}
        for s in spectra:
            if s.instrument not in inst_dict[s.obs_id]:
                inst_dict[s.obs_id].append(s.instrument)

        # The same idea as the source.instruments dictionary
        self._instruments = inst_dict

        # All the radii will be in degrees, but I'll grab it dynamically anyway
        self._rad_unit = spectra[0].inner_rad.unit

        # I want to grab the radii out of the spectra, then put them in order, just so I have them
        radii = sorted(list(set([s.inner_rad for s in spectra] + [s.outer_rad for s in spectra])))
        self._radii = Quantity([r.value for r in radii], self._rad_unit)
        self._ann_centres = Quantity([(self._radii[r_ind] + self._radii[r_ind + 1]) / 2 for r_ind in
                                      range(len(self._radii) - 1)], self._rad_unit)

        # This can be set through a property, as products shouldn't have any knowledge of their source
        #  other than the name. And someone might define one of these source-lessly. It will contain radii
        #  which are proper, not in degrees
        self._proper_radii = None
        self._proper_ann_centres = None

        # self._component_products = {ai: {o: {i: None for i in self._instruments[o]} for o in self.obs_ids}
        #                             for ai in range(self._num_ann)}
        # self._component_products = {o: {i: {ai: None for ai in range(self._num_ann)}
        #                                 for i in self._instruments[o]} for o in self.obs_ids}

        # Finally storing the spectra inside the product, though with multiple layers of products
        # This sets up the component products dictionary, allowing for the separated storage of
        #  spectra from different ObsIDs. We don't require that every ObsID-inst combo has an entry for every annulus
        self._component_products = {o: {i: {} for i in self._instruments[o]} for o in self.obs_ids}
        # And putting the spectra in their place
        for s in spectra:
            self._component_products[s.obs_id][s.instrument][s.annulus_ident] = s

        # Run through all the spectra associated with this AnnularSpectra and see if they are usable
        self._all_usable = all(s.usable for s in self.all_spectra)

        # This set of properties describe the configuration of evselect/specgroup during generation. I take
        #  properties from the first spectra in the list because they're all part of the same set, so were
        #  generated with the same settings
        self._grouped = spectra[0].grouped
        self._min_counts = spectra[0].min_counts
        self._min_sn = spectra[0].min_sn
        if self._grouped and self._min_counts is not None:
            self._grouped_on = 'counts'
        elif self._grouped and self._min_sn is not None:
            self._grouped_on = 'signal to noise'
        else:
            self._grouped_on = None

        # The RA-Dec coordinates that this set of spectra are centred on
        self._central_coord = spectra[0].central_coord

        # Not to do with grouping, but this states the level of oversampling requested from evselect
        self._over_sample = spectra[0].over_sample

        # Here we generate the storage key for this object, its just convenient to do it in here
        # Sets up the extra part of the storage key name depending on if grouping is enabled
        if self._grouped and self._min_counts is not None:
            extra_name = "_mincnt{}".format(self._min_counts)
        elif self._grouped and self._min_sn is not None:
            extra_name = "_minsn{}".format(self._min_sn)
        else:
            extra_name = ''

        # And if it was oversampled during generation then we need to include that as well
        if self._over_sample is not None:
            extra_name += "_ovsamp{ov}".format(ov=self._over_sample)

        # Combines the annular radii into a string
        ann_rad_str = "_".join(self._radii.value.astype(str))

        spec_storage_name = "ra{ra}_dec{dec}_ar{ar}_grp{gr}"
        spec_storage_name = spec_storage_name.format(ra=self.central_coord[0].value,
                                                     dec=self.central_coord[1].value, ar=ann_rad_str, gr=self._grouped)

        spec_storage_name += extra_name
        # And we save the completed key to an attribute
        self._storage_key = spec_storage_name

        # Now for a very important step, all the constituent spectra need to know that their new background
        #  spectrum is the one from the outermost annulus. I have added a method to this class to find the correct
        #  file for an ObsID and instrument, and apparently past me added a property setter to the Spectrum class
        #  will automatically push the change to the file headers, so that is handy
        for s in self.all_spectra:
            s.background = self.background(s.obs_id, s.instrument)

        # Setting up attributes that allow for the storage of final fit results within this class, very similar
        #  to how they're stored in a source object. This makes sense here because an AnnularSpectra is an
        #  aggregate product of all the relevant spectra. All fit results are stored on annular basis, then most
        #  will have different entries for different models

        # The total exposure of the combined spectra, will be overwritten if multiple models are fit, but
        #  as its a property of the spectra and not the fit it should always be the same
        self._total_exp = {ai: None for ai in range(self._num_ann)}
        # These will be stored on a per model basis
        self._total_count_rate = {ai: {} for ai in range(self._num_ann)}
        self._test_stat = {ai: {} for ai in range(self._num_ann)}
        self._fit_stat = {ai: {} for ai in range(self._num_ann)}
        self._dof = {ai: {} for ai in range(self._num_ann)}

        # Finally the most important outputs, the fit results and luminosities. There obviously is some data
        #  duplication here with the source, but this will be so convenient I don't care
        self._fit_results = {ai: {} for ai in range(self._num_ann)}
        self._luminosities = {ai: {} for ai in range(self._num_ann)}

        # Observation order for an annulus describes, for results with multiple entries like normalisation can if
        #  it is not linked across multiple spectra during fitting, what order the fit results are in.
        self._obs_order = {ai: {} for ai in range(self._num_ann)}

        # This dictionary will store the paths to cross-arfs that might be generated for this annular spectrum. That
        #  is why there are two layers of annulus identifiers - each annulus gets one arf per every OTHER annulus
        #  (not itself)
        self._cross_arfs = {o: {i: {ai: {aii: None for aii in range(self._num_ann) if aii != ai}
                                    for ai in range(self._num_ann)}
                                for i in self._instruments[o]} for o in self.obs_ids}
        # If at any point the user decides that they want to actually access the data in the cross-arfs, which they
        #  might do to plot the response curves for instance, then the 'read on demand' method for the cross-arfs will
        #  store them in these attributes - the key structure is the same as the above _cross_arfs attribute that
        #  stores the paths to the files
        self._cross_arf_lo_ens = {o: {i: {ai: {aii: None for aii in range(self._num_ann) if aii != ai}
                                          for ai in range(self._num_ann)} for i in self._instruments[o]}
                                  for o in self.obs_ids}
        self._cross_arf_hi_ens = {o: {i: {ai: {aii: None for aii in range(self._num_ann) if aii != ai}
                                          for ai in range(self._num_ann)}
                                  for i in self._instruments[o]} for o in self.obs_ids}
        self._cross_arf_eff_areas = {o: {i: {ai: {aii: None for aii in range(self._num_ann) if aii != ai}
                                              for ai in range(self._num_ann)}
                                     for i in self._instruments[o]} for o in self.obs_ids}

        # Attributes to store ARF information
        # The actual effective area information will live in this attribute
        self._arf_eff_area = None
        # The corresponding energy bounds will be stored in these attributes
        self._arf_lo_en = None
        self._arf_hi_en = None

    # The src_name setter and getter have been overridden because there is an easier way of setting
    #  the source name for all spectra
    @property
    def src_name(self) -> str:
        """
        Method to return the name of the object a product is associated with. The product becomes
        aware of this once it is added to a source object.

        :return: The name of the source object this product is associated with.
        :rtype: str
        """
        return self._src_name

    @src_name.setter
    def src_name(self, name: str):
        """
        Property setter for the src_name attribute of a product, should only really be called by a source object,
        not by a user.

        :param str name: The name of the source object associated with this product.
        """
        self._src_name = name
        for p in self.all_spectra:
            p.src_name = name

    @property
    def central_coord(self) -> Quantity:
        """
        This property provides the central coordinates (RA-Dec) that this set of spectra was
        generated around.

        :return: Astropy quantity object containing the central coordinate in degrees.
        :rtype: Quantity
        """
        return self._central_coord

    @property
    def num_annuli(self) -> int:
        """
        A property getter for the number of annular spectra.

        :return: The number of annular spectra associated with this product.
        :rtype: int
        """
        return self._num_ann

    @property
    def annulus_ids(self) -> np.ndarray:
        """
        The set of annulus IDs for this AnnularSpectra; i.e. for an AnnularSpectra with 4 annuli, this will return
        an array of 0, 1, 2, and 3.

        :return: The array of annulus IDs.
        :rtype: np.ndarray
        """
        return np.array(range(0, self.num_annuli))

    def background(self, obs_id: str, inst: str) -> str:
        """
        This method returns the path to the background spectrum for a particular ObsID and
        instrument. It is the background associated with the outermost annulus of this object.

        :param str obs_id: The ObsID to get the background spectrum for.
        :param str inst: The instrument to get the background spectrum for.
        :return: Path of the background spectrum.
        :rtype: str
        """
        return self.get_spectra(self._num_ann-1, obs_id, inst).background

    def background_rmf(self, obs_id: str, inst: str) -> str:
        """
        This method returns the path to the background spectrum's RMF for a particular ObsID and
        instrument. It is the RMF of the background associated with the outermost annulus of this object.

        :param str obs_id: The ObsID to get the background spectrum's RMF for.
        :param str inst: The instrument to get the background spectrum' RMF for.
        :return: Path of the background spectrum RMF.
        :rtype: str
        """
        return self.get_spectra(self._num_ann-1, obs_id, inst).background_rmf

    def background_arf(self, obs_id: str, inst: str) -> str:
        """
        This method returns the path to the background spectrum's ARF for a particular ObsID and
        instrument. It is the ARF of the background associated with the outermost annulus of this object.

        :param str obs_id: The ObsID to get the background spectrum's ARF for.
        :param str inst: The instrument to get the background spectrum' ARF for.
        :return: Path of the background spectrum ARF.
        :rtype: str
        """
        return self.get_spectra(self._num_ann - 1, obs_id, inst).background_arf

    @property
    def obs_ids(self) -> list:
        """
        A property of this spectrum set that details which ObsIDs have contributed spectra to this object.

        :return: A list of ObsIDs.
        :rtype: list
        """
        return list(self._instruments.keys())

    @property
    def instruments(self) -> dict:
        """
        A property of this spectrum set that details which ObsIDs and instruments have contributed spectra
        to this object.

        :return: A dictionary of lists, with the top level keys being ObsIDs, and the lists
            containing instruments associated with those ObsIDs.
        :rtype: dict
        """
        return self._instruments

    def get_spectra(self, annulus_ident, obs_id: str = None, inst: str = None) -> Union[List[Spectrum], Spectrum]:
        """
        This is the getter for the spectra stored in the AnnularSpectra data storage structure. They can
        be retrieved based on annulus identifier, ObsID, and instrument.

        :param int annulus_ident: The annulus identifier to retrieve spectra for.
        :param str obs_id: Optionally, a specific obs_id to search for can be supplied.
        :param str inst: Optionally, a specific instrument to search for can be supplied.
        :return: List of matching spectra, or just a Spectrum object if one match is found.
        :rtype: Union[List[Spectrum], Spectrum]
        """
        def unpack_list(to_unpack: list):
            """
            A recursive function to go through every layer of a nested list and flatten it all out. It
            doesn't return anything because to make life easier the 'results' are appended to a variable
            in the namespace above this one.

            :param list to_unpack: The list that needs unpacking.
            """
            # Must iterate through the given list
            for entry in to_unpack:
                # If the current element is not a list then all is chill, this element is ready for appending
                # to the final list
                if not isinstance(entry, list):
                    out.append(entry)
                else:
                    # If the current element IS a list, then obviously we still have more unpacking to do,
                    # so we call this function recursively.
                    unpack_list(entry)

        if annulus_ident not in np.array(range(0, self._num_ann)):
            ann_str = ", ".join(np.array(range(0, self._num_ann)).astype(str))
            raise IndexError("{i} is not an annulus ID associated with this AnnularSpectra object. "
                             "Allowed annulus IDs are; {a}".format(i=annulus_ident, a=ann_str))
        elif obs_id not in self._component_products and obs_id is not None:
            raise NotAssociatedError("{0} is not associated with this AnnularSpectra.".format(obs_id))
        elif (obs_id is not None and obs_id in self._component_products) and \
                (inst is not None and inst not in self._component_products[obs_id]):
            raise NotAssociatedError("Instrument {1} is not associated with {0}".format(obs_id, inst))

        matches = []
        for match in dict_search(annulus_ident, self._component_products):
            out = []
            unpack_list(match)
            if (obs_id == out[0] or obs_id is None) and (inst == out[1] or inst is None):
                matches.append(out[-1])

        # Here I only return the object if one match was found
        if len(matches) == 1:
            matches = matches[0]
        return matches

    @property
    def all_spectra(self) -> List[Spectrum]:
        """
        Simple extra wrapper for get_spectra that allows the user to retrieve every single spectrum associated
        with this AnnularSpectra instance, for all annulus IDs.

        :return: A list of every single spectrum associated with this object.
        :rtype: List[Spectrum]
        """
        all_spec = []
        for ann_i in range(self._num_ann):
            # If there is only one spectrum per annulus then get_spectra will just return an object
            ann_spec = self.get_spectra(ann_i)
            if isinstance(ann_spec, Spectrum):
                ann_spec = [ann_spec]
            all_spec += ann_spec

        return all_spec

    @property
    def radii(self) -> Quantity:
        """
        A property to return all the boundary radii of the constituent annuli.

        :return: Astropy quantity of the radii.
        :rtype: Quantity
        """
        return self._radii

    @property
    def annulus_centres(self) -> Quantity:
        """
        Returns the centres of all the annuli, in the original units the radii were passed in with.

        :return: An astropy quantity containing radii.
        :rtype: Quantity
        """
        return self._ann_centres

    @property
    def proper_radii(self) -> Quantity:
        """
        A property to return the boundary radii of the constituent annuli in kpc. This has
        to be set using the setter first, otherwise the value is None.

        :return: Astropy quantity of the proper radii.
        :rtype: Quantity
        """
        if self._proper_radii is not None:
            to_return = self._proper_radii.to('kpc')
        else:
            to_return = self._proper_radii

        return to_return

    @proper_radii.setter
    def proper_radii(self, new_vals: Quantity):
        """
        A setter for the proper radii property.

        :param Quantity new_vals: The new values for proper radii, must be convertible to kpc.
        """
        if not new_vals.unit.is_equivalent('kpc'):
            raise UnitConversionError("Proper radii passed into this object must be convertable to kpc.")
        elif new_vals.isscalar:
            raise ValueError("A radii quantity for an AnnularSpectra object cannot be scalar")
        elif len(new_vals) != len(self._radii):
            raise ValueError("The proper radii quantity you have passed isn't the same length as the radii "
                             "attribute of this object, there should be {} entries.".format(len(self._radii)))

        self._proper_radii = new_vals
        pr = self.proper_radii.value
        # Minds the mid points of the annular boundaries - the centres of the bins
        mid_radii = [(pr[r_ind] + pr[r_ind + 1]) / 2 for r_ind in range(len(pr) - 1)]
        # Makes mid_radii a quantity
        self._proper_ann_centres = Quantity(mid_radii, new_vals.unit)

    @property
    def proper_annulus_centres(self) -> Quantity:
        """
        Returns the centres of all the annuli, in the units of proper radii which the user has to have
        set through the property setter.

        :return: An astropy quantity containing radii, or None if no proper radii exist
        :rtype: Quantity
        """
        return self._proper_ann_centres

    @property
    def set_ident(self) -> int:
        """
        This property returns the ID of this set of spectra.

        :return: The integer ID of this set.
        :rtype: int
        """
        return self._set_id

    @property
    def storage_key(self) -> str:
        """
        This property returns the storage key which this object assembles to place the AnnularSpectra in
        an XGA source's storage structure. The key is based on the properties of the AnnularSpectra, and
        some of the configuration options, and is basically human readable.

        :return: String storage key.
        :rtype: str
        """
        return self._storage_key

    @property
    def grouped(self) -> bool:
        """
        A property stating whether SAS was told to group the spectra in this set during generation or not.

        :return: Boolean variable describing whether the spectra are grouped or not
        :rtype: bool
        """
        return self._grouped

    @property
    def grouped_on(self) -> str:
        """
        A property stating what metric the spectra in this set were grouped on.

        :return: String representation of the metric the spectra were grouped on (None if not grouped).
        :rtype: str
        """
        return self._grouped_on

    @property
    def min_counts(self) -> int:
        """
        A property stating the minimum number of counts allowed in a grouped channel for the spectra in this set.

        :return: The integer minimum number of counts per grouped channel (if these spectra were grouped on
            minimum numbers of counts).
        :rtype: int
        """
        return self._min_counts

    @property
    def min_sn(self) -> Union[float, int]:
        """
        A property stating the minimum signal to noise allowed in a grouped channel for the spectra in this set.

        :return: The minimum signal to noise per grouped channel (if these spectra were grouped on
            minimum signal to noise).
        :rtype: Union[float, int]
        """
        return self._min_sn

    @property
    def over_sample(self) -> float:
        """
        A property string stating the amount of oversampling applied by evselect during the generation
        of the spectra in this set. e.g. if over_sample=3 then the minimum width of a group is
        1/3 of the resolution FWHM at that energy.

        :return: Oversampling applied during generation.
        :rtype: float
        """
        return self._over_sample

    @property
    def fitted_models(self) -> list:
        """
        A property that gets the list of spectral models that have been fit to this AnnularSpectra instance.

        :return: A list of models fit to this annular spectrum.
        :rtype: list
        """
        return list(self._fit_results[0].keys())

    @property
    def fitted_model_configurations(self) -> dict:
        """
        Property that returns a dictionary with model names as keys and lists of fit configuration identifiers as
        values, for the models that have been fit to this annular spectrum.

        :return: Dictionary with model names as keys, and lists of model configuration identifiers as values.
        :rtype: dict
        """
        return {m: list(self._fit_results[0][m].keys()) for m in self.fitted_models}

    def add_cross_arf(self, arf: Union[BaseProduct, str], obs_id: str, inst: str, src_ann_id: int, cross_ann_id: int,
                      set_ident: int):
        """
        This method allows you to add cross-arfs generated for this annular spectrum to a storage structure in
        the object. That means that other processes that make use of them, such as spectral fitting, will be able
        to retrieve them from this object easily.

        :param BaseProduct/str arf: Either an XGA BaseProduct instance (which is what is produced by the generation
            process) or a string path to the cross-arf file. This should represent the 'cross_ann_id' contribution
            to the 'src_ann_id' spectrum.
        :param str obs_id: The ObsID of the cross-arf - if 'arf' is a BaseProduct instance this argument will be
            compared the ObsID of the 'arf' object. It will also be checked to ensure that it is associated with
            this object.
        :param str inst: The instrument of the cross-arf - if 'arf' is a BaseProduct instance this argument will be
            compared the instrument of the 'arf' object. It will also be checked to ensure that it is associated with
            the ObsID in this object.
        :param int src_ann_id: The identifying number of the source annulus for this cross-arf.
        :param int cross_ann_id: The identifying number of the 'cross' annulus for this cross-arf.
        :param int set_ident: The set_ident that the cross-arf was generated for, it is checked against the
            set identifier of this object.
        """

        # Hopefully this is never triggered, and obviously you could just grab the set_ident from the object you're
        #  adding too if you want to get past this, but I mean this more as a check for the development of the
        #  cross_arfs function to ensure I'm adding the right cross-arfs to the right annular spectra.
        if set_ident != self.set_ident:
            raise ValueError("The passed 'set_ident' ({s}) does not match the identifier of this object "
                             "({ri}).".format(s=set_ident, ri=self.set_ident))

        # I check whether the specified ObsID is actually a part of this AnnularSpectra, and raise a hopefully
        #  informative error if it is not.
        if obs_id not in self.obs_ids:
            raise NotAssociatedError("The passed 'obs_id' ({o}) is not associated with this annular spectrum "
                                     "({ol})".format(o=obs_id, ol=', '.join(self.obs_ids)))
        # If we've passed the first check, and the passed arf is an XGA product instance, we perform a sanity check
        #  to ensure that the ObsID passed by the user matches the one in the product
        elif isinstance(arf, BaseProduct) and obs_id != arf.obs_id:
            raise ValueError("The 'obs_id' ({o}) argument does not match the ObsID set for the XGA product containing "
                             "the cross-arf ({po}).".format(o=obs_id, po=arf.obs_id))

        # We then repeat the exact same checks as above, but with the instrument
        if inst not in self.instruments[obs_id]:
            raise NotAssociatedError("The passed 'inst' ({i}) is not associated with the ObsID in this annular "
                                     "spectrum ({il}).".format(i=inst, il=', '.join(self.instruments[obs_id])))
        elif isinstance(arf, BaseProduct) and inst != arf.instrument:
            raise ValueError("The 'inst' ({i}) argument does not match the instrument set for the XGA product "
                             "containing the cross-arf ({pi}).".format(i=inst, pi=arf.inst))

        # I think I will allow either a string path, or a product (as I am declaring these arfs in BaseProducts in
        #  the execute_cmd function) - as such I need to account for both. In this case we know that everything is fine
        #  because the product has told us that it is usable
        if isinstance(arf, BaseProduct) and arf.usable:
            arf = arf.path
        # In this case though something has obviously gone wrong, so we'll tell the user about it
        elif isinstance(arf, BaseProduct) and not arf.usable:
            raise FailedProductError("The specified cross-arf is not usable for the following reasons; "
                                     "{}".format(', '.join(arf.not_usable_reasons)))
        # In this case we assume that the string was the path, and if it doesn't exist we say so.
        elif isinstance(arf, str) and not os.path.exists(arf):
            raise FileNotFoundError("The specified cross-arf file ({}) cannot be found.".format(arf))

        # We make absolutely sure that the input annulus identifiers are integer representations, not strings, just
        #  to be safe as it could screw things up downstream from here
        if isinstance(src_ann_id, str):
            src_ann_id = int(src_ann_id)
        if isinstance(cross_ann_id, str):
            cross_ann_id = int(cross_ann_id)

        # If we've got here we know that the 'arf' argument was alright, but now we have to try to check the
        #  validity of the src_ann_id and cross_ann_id values. Firstly they have to actually be identifying annuli
        #  present in this object
        if src_ann_id not in self.annulus_ids or cross_ann_id not in self.annulus_ids:
            raise NotAssociatedError("The 'src_ann_id' and 'cross_ann_id' arguments must be annulus identifiers "
                                     "associated with this AnnularSpectra, allowed values "
                                     "are; {}".format(', '.join([str(i) for i in self.annulus_ids])))
        # Having the same value for both isn't allowed, because that doesn't make sense. Cross arfs represent the
        #  contribution of one annulus to another, so what use would an arf be that represents the contribution of one
        #  annulus to itself?
        elif src_ann_id == cross_ann_id:
            raise ValueError("The 'src_ann_id' and 'cross_ann_id' arguments have the same value ({}). They must be "
                             "different as cross-arfs represent the contribution of one annulus to "
                             "another.".format(str(src_ann_id)))

        # Now we actually store the PATH to the file in the attribute that was set up in the init of this
        #  class - This one is four layers deep, as every source annulus will have a cross arf for every
        #  other annulus
        self._cross_arfs[obs_id][inst][src_ann_id][cross_ann_id] = arf

    def get_cross_arf_paths(self, obs_id: str, inst: str, src_ann_id: int, cross_ann_id: int = None) -> dict:
        """
        This method allows the user to retrieve cross-arf paths for a specific ObsID-instrument spectrum of an
        annulus. For instance, passing an ObsID and Instrument, along with a src_ann_id of 1, to an annular spectrum
        with four annuli, will return the paths to cross-arfs between annulus 1 and 0, annulus 1 and 2, and
        annulus 1 and 3 (labelling starts at zero). If a cross_ann_id is passed in addition to a src_ann_id, then
        that specific path will be retrieved (but still returned in a dictionary).

        :param str obs_id: The ObsID of the spectrum for which you wish to retrieve cross-arf paths.
        :param str inst: The instrument of the spectrum for which you wish to retrieve cross-arf paths.
        :param int src_ann_id: The annulus ID (e.g. 1) of the spectrum you wish to retrieve cross-arf paths for.
        :param int cross_ann_id: Optionally you can specify the cross-arf annulus ID. The default is None, in which
            case all cross-arf paths for the given source annulus will be returned.
        :return: A dictionary with annulus IDs as keys, and cross-arf paths as values.
        :rtype: dict
        """
        if obs_id not in self.obs_ids:
            raise NotAssociatedError("The passed 'obs_id' ({o}) is not associated with this annular spectrum "
                                     "({ol})".format(o=obs_id, ol=', '.join(self.obs_ids)))
        elif inst not in self.instruments[obs_id]:
            raise NotAssociatedError("The passed 'inst' ({i}) is not associated with the ObsID in this annular "
                                     "spectrum ({il}).".format(i=inst, il=', '.join(self.instruments[obs_id])))
        elif src_ann_id not in self.annulus_ids:
            raise NotAssociatedError("The passed 'src_ann_id' ({si}) is not an annulus ID associated with this annular"
                                     " spectrum ({ls}).".format(si=src_ann_id,
                                                                ls=', '.join([str(i) for i in self.annulus_ids])))
        elif cross_ann_id is not None and cross_ann_id not in self.annulus_ids:
            raise NotAssociatedError("The passed 'cross_ann_id' ({si}) is not an annulus ID associated with this "
                                     "annular spectrum "
                                     "({ls}).".format(si=cross_ann_id,
                                                      ls=', '.join([str(i) for i in self.annulus_ids])))

        if cross_ann_id is None:
            # If we pass those checks we can grab the requested cross-ARFs.
            rel_arfs = self._cross_arfs[obs_id][inst][src_ann_id]
        else:
            # In this case we just want to return a single, specific, path - we will still return it in the same
            #  form though - in a dictionary with the key being the cross-arf ann id
            rel_arfs = {cross_ann_id: self._cross_arfs[obs_id][inst][src_ann_id][cross_ann_id]}

        # We do a final check to make sure that no None values sneak through
        if None in rel_arfs.values():
            raise NotAssociatedError("One or more cross-arfs for your selection have not been assigned to this "
                                     "annular spectrum.")
        return rel_arfs

    def get_cross_arf_lo_ens(self, obs_id: str, inst: str, src_ann_id: int, cross_ann_id: int = None) -> dict:
        """
        A method that will retrieve the lower energy bound data from cross-arfs associated with this annular
        spectrum. A set of cross-arf lower energy bounds can be retrieved for a particular source annulus, or
        an individual cross-arf for a particular source annulus, depending on user input. It is extremely likely
        that all lower energy bounds will be the same for a given instrument, but this is not assumed.

        :param str obs_id: The ObsID of the spectrum for which you wish to retrieve cross-arf lower energy bounds.
        :param str inst: The instrument of the spectrum for which you wish to retrieve cross-arf lower energy bounds.
        :param int src_ann_id: The annulus ID (e.g. 1) of the spectrum you wish to retrieve cross-arf lower energy
            bounds for.
        :param int cross_ann_id: Optionally you can specify the cross-arf annulus ID. The default is None, in which
            case all lower energy bounds of cross-arfs for the given source annulus will be returned.
        :return: A dictionary with annulus IDs as keys, and astropy array quantities of lower energy bounds as values.
        :rtype: dict
        """
        # This is a bit cheesy, but by calling this get method for cross arf paths I can know that the correct
        #  checks are being made on the ObsID, instrument, src ann id, and cross ann id. The only other thing this
        #  method does is a dictionary call, which isn't going to be a huge burden
        self.get_cross_arf_paths(obs_id, inst, src_ann_id, cross_ann_id)

        # Now that we know that the input values are all valid, we first deal with the case where a set of cross-arf
        #  data are being retrieved. We check to see whether 'None' is in lo_ens attribute's values, and if it
        #  is then we decide that the data haven't been loaded in and we trigger the read on demand method
        if cross_ann_id is None and None in self._cross_arf_lo_ens[obs_id][inst][src_ann_id].values():
            self._read_cross_arf_on_demand(obs_id, inst, src_ann_id)
        # In this case a specific cross-arf's data are being requested, so we check that particular cross ann ID
        #  to see whether the entry in the attribute is currently None. If it is we trigger the read method
        elif cross_ann_id is not None and self._cross_arf_lo_ens[obs_id][inst][src_ann_id][cross_ann_id] is None:
            self._read_cross_arf_on_demand(obs_id, inst, src_ann_id, cross_ann_id)

        if cross_ann_id is None:
            rel_lo_ens = self._cross_arf_lo_ens[obs_id][inst][src_ann_id]
        else:
            # I want the return to be a dictionary regardless of whether the input was specific to one cross-arf or not
            rel_lo_ens = {cross_ann_id: self._cross_arf_lo_ens[obs_id][inst][src_ann_id][cross_ann_id]}

        return rel_lo_ens

    def get_cross_arf_hi_ens(self, obs_id: str, inst: str, src_ann_id: int, cross_ann_id: int = None) -> dict:
        """
        A method that will retrieve the upper energy bound data from cross-arfs associated with this annular
        spectrum. A set of cross-arf upper energy bounds can be retrieved for a particular source annulus, or
        an individual cross-arf for a particular source annulus, depending on user input. It is extremely likely
        that all upper energy bounds will be the same for a given instrument, but this is not assumed.

        :param str obs_id: The ObsID of the spectrum for which you wish to retrieve cross-arf upper energy bounds.
        :param str inst: The instrument of the spectrum for which you wish to retrieve cross-arf upper energy bounds.
        :param int src_ann_id: The annulus ID (e.g. 1) of the spectrum you wish to retrieve cross-arf upper energy
            bounds for.
        :param int cross_ann_id: Optionally you can specify the cross-arf annulus ID. The default is None, in which
            case all upper energy bounds of cross-arfs for the given source annulus will be returned.
        :return: A dictionary with annulus IDs as keys, and astropy array quantities of upper energy bounds as values.
        :rtype: dict
        """
        # This is a bit cheesy, but by calling this get method for cross arf paths I can know that the correct
        #  checks are being made on the ObsID, instrument, src ann id, and cross ann id. The only other thing this
        #  method does is a dictionary call, which isn't going to be a huge burden
        self.get_cross_arf_paths(obs_id, inst, src_ann_id, cross_ann_id)

        # Now that we know that the input values are all valid, we first deal with the case where a set of cross-arf
        #  data are being retrieved. We check to see whether 'None' is in hi_ens attribute's values, and if it
        #  is then we decide that the data haven't been loaded in, and we trigger the read on demand method
        if cross_ann_id is None and None in self._cross_arf_hi_ens[obs_id][inst][src_ann_id].values():
            self._read_cross_arf_on_demand(obs_id, inst, src_ann_id)
        # In this case a specific cross-arf's data are being requested, so we check that particular cross ann ID
        #  to see whether the entry in the attribute is currently None. If it is we trigger the read method
        elif cross_ann_id is not None and self._cross_arf_hi_ens[obs_id][inst][src_ann_id][cross_ann_id] is None:
            self._read_cross_arf_on_demand(obs_id, inst, src_ann_id, cross_ann_id)

        if cross_ann_id is None:
            rel_hi_ens = self._cross_arf_hi_ens[obs_id][inst][src_ann_id]
        else:
            # I want the return to be a dictionary regardless of whether the input was specific to one cross-arf or not
            rel_hi_ens = {cross_ann_id: self._cross_arf_hi_ens[obs_id][inst][src_ann_id][cross_ann_id]}

        return rel_hi_ens

    def get_cross_arf_eff_areas(self, obs_id: str, inst: str, src_ann_id: int, cross_ann_id: int = None) -> dict:
        """
        A method that will retrieve the effective area data from cross-arfs associated with this annular
        spectrum. A set of cross-arf effective areas can be retrieved for a particular source annulus, or
        an individual cross-arf for a particular source annulus, depending on user input.

        :param str obs_id: The ObsID of the spectrum for which you wish to retrieve cross-arf effective areas.
        :param str inst: The instrument of the spectrum for which you wish to retrieve cross-arf effective areas.
        :param int src_ann_id: The annulus ID (e.g. 1) of the spectrum you wish to retrieve cross-arf effective
            areas for.
        :param int cross_ann_id: Optionally you can specify the cross-arf annulus ID. The default is None, in which
            case all effective areas of cross-arfs for the given source annulus will be returned.
        :return: A dictionary with annulus IDs as keys, and astropy array quantities of effective areas as values.
        :rtype: dict
        """
        # This is a bit cheesy, but by calling this get method for cross arf paths I can know that the correct
        #  checks are being made on the ObsID, instrument, src ann id, and cross ann id. The only other thing this
        #  method does is a dictionary call, which isn't going to be a huge burden
        self.get_cross_arf_paths(obs_id, inst, src_ann_id, cross_ann_id)

        # Now that we know that the input values are all valid, we first deal with the case where a set of cross-arf
        #  data are being retrieved. We check to see whether 'None' is in the eff_area attribute's values, and if it
        #  is then we decide that the data haven't been loaded in, and we trigger the read on demand method
        if cross_ann_id is None and None in self._cross_arf_eff_areas[obs_id][inst][src_ann_id].values():
            self._read_cross_arf_on_demand(obs_id, inst, src_ann_id)
        # In this case a specific cross-arf's data are being requested, so we check that particular cross ann ID
        #  to see whether the entry in the attribute is currently None. If it is we trigger the read method
        elif cross_ann_id is not None and self._cross_arf_eff_areas[obs_id][inst][src_ann_id][cross_ann_id] is None:
            self._read_cross_arf_on_demand(obs_id, inst, src_ann_id, cross_ann_id)

        if cross_ann_id is None:
            rel_eff_areas = self._cross_arf_eff_areas[obs_id][inst][src_ann_id]
        else:
            # I want the return to be a dictionary regardless of whether the input was specific to one cross-arf or not
            rel_eff_areas = {cross_ann_id: self._cross_arf_eff_areas[obs_id][inst][src_ann_id][cross_ann_id]}

        return rel_eff_areas

    def _read_cross_arf_on_demand(self, obs_id: str, inst: str, src_ann_id: int, cross_ann_id: int = None):
        """
        An internal method to read cross-arf data into memory only when required by some other method or property
        of this function - that way we don't unnecessarily take up memory.

        :param str obs_id: The ObsID of the spectrum for which cross-arf data should be read into memory.
        :param str inst: The instrument of the spectrum for which cross-arf data should be read into memory.
        :param int src_ann_id: The annulus ID (e.g. 1) of the spectrum for which cross-arf data should be
            read into memory.
        :param int cross_ann_id: Optionally you can specify the cross-arf annulus ID. The default is None, in which
            case all cross-arf data for the given source annulus will be read into memory.
        """
        # Retrieves the paths to the cross-arfs specified by the input to this method
        cross_paths = self.get_cross_arf_paths(obs_id, inst, src_ann_id, cross_ann_id)

        # We cycle through the cross-arfs. Even if a cross_ann_id was specified (i.e. we've only retrieved a path for
        #  a single cross-arf), it will still be returned in a dictionary so this will work fun
        for c_a_id, rel_path in cross_paths.items():
            # Do the actual reading of the current cross-arf file
            arf_read = FITS(rel_path)
            # Now store the array of lower energy limits, upper energy limits, and effective areas in the attributes
            #  that were set up with an ObsID-Instrument-src_ann_id-cross_ann_id structure in the init of this class
            self._cross_arf_lo_ens[obs_id][inst][src_ann_id][c_a_id] = Quantity(arf_read[1]['ENERG_LO'].read(), 'keV')
            self._cross_arf_hi_ens[obs_id][inst][src_ann_id][c_a_id] = Quantity(arf_read[1]['ENERG_HI'].read(), 'keV')
            self._cross_arf_eff_areas[obs_id][inst][src_ann_id][c_a_id] = \
                Quantity(arf_read[1]['SPECRESP'].read(), 'cm^2')

            # And make sure to close the arf file after reading
            arf_read.close()

    def add_fit_data(self, model: str, tab_line: dict, lums: dict, obs_order: dict, fit_conf: str):
        """
        An equivalent to the add_fit_data method built into all source objects. The final fit results
        and luminosities are housed in a storage structure within the AnnularSpectra, which makes sense
        because this is an aggregate product of all the relevant spectra, storing them just as source objects
        store spectra that don't exist in a spectrum set.

        :param str model: The XSPEC definition of the model used to perform the fit. e.g. constant*tbabs*apec
        :param tab_line: A dictionary of table lines with fit data, the keys of the dictionary being
            the relevant annulus ID for the fit.
        :param dict lums: A dictionary of the luminosities measured during the fits, the keys of the
            outermost dictionary being annulus IDs, and the luminosity dictionaries being energy based.
        :param dict obs_order: A dictionary (with keys being annuli idents) of lists of lists describing the order
            the data is being passed, so that specific results can be related back to specific observations later
            (if applicable). The lists should be structured like [[obsid1, inst1], [obsid1, inst2], [obsid1, inst3]]
            for instance.
        :param str fit_conf: In order to be able to store results for different fit configurations (e.g. different
            starting pars, abundance tables, all that), we need to have a key that identifies the configuration. We
            do not expect the user to be adding fit data, so this will be a key generated by the fit function.
        """
        # Just headers that will always be present in tab_line that are not fit parameters
        not_par = ['MODEL', 'TOTAL_EXPOSURE', 'TOTAL_COUNT_RATE', 'TOTAL_COUNT_RATE_ERR',
                   'NUM_UNLINKED_THAWED_VARS', 'FIT_STATISTIC', 'TEST_STATISTIC', 'DOF']

        # Checking that we have the expected amount of data passed in
        if len(tab_line) != self._num_ann:
            raise ValueError("The dictionary passed in with the fit results in it does not have the same"
                             " number of entries as there are annuli.")
        elif len(lums) != self._num_ann:
            raise ValueError("The dictionary passed in with the luminosities in it does not have the same"
                             " number of entries as there are annuli.")

        for ai in range(0, self._num_ann):
            # Various global values of interest
            self._total_exp[ai] = float(tab_line[ai]["TOTAL_EXPOSURE"])
            # If the model isn't already a key in the nested dictionary, this will add a dictionary entry (neatest
            #  way I could find of doing this).
            self._total_count_rate[ai].setdefault(model, {})
            self._total_count_rate[ai][model][fit_conf] = [float(tab_line[ai]["TOTAL_COUNT_RATE"]),
                                                           float(tab_line[ai]["TOTAL_COUNT_RATE_ERR"])]
            self._test_stat[ai].setdefault(model, {})
            self._test_stat[ai][model][fit_conf] = float(tab_line[ai]["TEST_STATISTIC"])
            self._fit_stat[ai].setdefault(model, {})
            self._fit_stat[ai][model][fit_conf] = float(tab_line[ai]["FIT_STATISTIC"])
            self._dof[ai].setdefault(model, {})
            self._dof[ai][model][fit_conf] = float(tab_line[ai]["DOF"])
            self._obs_order[ai].setdefault(model, {})
            self._obs_order[ai][model][fit_conf] = obs_order[ai]

            # The parameters available will obviously be dynamic, so have to find out what they are and then
            #  then for each result find the +- errors.
            par_headers = [n for n in tab_line[ai].dtype.names if n not in not_par]
            mod_res = {}
            for par in par_headers:
                # The parameter name and the parameter index used by XSPEC are separated by |
                par_info = par.split("|")
                par_name = par_info[0]

                # The parameter index can also have an - or + after it if the entry in question is an uncertainty
                if par_info[1][-1] == "-":
                    ident = par_info[1][:-1]
                    pos = 1
                elif par_info[1][-1] == "+":
                    ident = par_info[1][:-1]
                    pos = 2
                else:
                    ident = par_info[1]
                    pos = 0

                # Sets up the dictionary structure for the results
                if par_name not in mod_res:
                    mod_res[par_name] = {ident: [0, 0, 0]}
                elif ident not in mod_res[par_name]:
                    mod_res[par_name][ident] = [0, 0, 0]

                mod_res[par_name][ident][pos] = float(tab_line[ai][par])

            # Storing the fit results
            self._fit_results[ai].setdefault(model, {})
            self._fit_results[ai][model][fit_conf] = mod_res
            # And now storing the luminosity results
            self._luminosities[ai].setdefault(model, {})
            self._luminosities[ai][model][fit_conf] = lums[ai]

    def _get_fit_checks(self, annulus_ident: int, model: str = None, par: str = None,
                        fit_conf: Union[str, dict] = None) -> Tuple[str, str]:
        """
        An internal function to perform input checks and pre-processing for get methods that access fit results, or
        other related information such as fit statistic.

        :param int annulus_ident: The annulus for which you wish to retrieve the fit results.
        :param str model: The name of the fitted model that you're requesting the results from
            (e.g. constant*tbabs*apec).
        :param str par: The name of the parameter you want a result for.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: The model name and fit configuration.
        :rtype: Tuple[str, str]
        """
        from ..xspec.fit import FIT_FUNC_MODEL_NAMES
        from ..xspec.fitconfgen import fit_conf_from_function

        # It is possible to pass a null value for the 'model' parameter, but we'll only accept that if a single model
        #  has been fit to this annular spectrum - otherwise how are we to know which model they want?
        if len(self.fitted_models) == 0:
            # Sort of duplicates an error further down, but this case will only trigger if 'model' is None.
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this AnnularSpectra object.")
        elif model is None and len(self.fitted_models) != 1:
            av_mods = ", ".join(self._fit_results[annulus_ident].keys())
            raise ValueError("Multiple models have been fit to this annular spectrum, so model=None is not "
                             "valid; available models are {a}".format(m=model, a=av_mods))
        else:
            # In this case there is ONE model fit, and the user didn't pass a model parameter value, so we'll just
            #  automatically select it for them
            model = self.fitted_models[0]

        # Checks the input fit configuration values - if they are completely illegal we throw an error
        if fit_conf is not None and not isinstance(fit_conf, (str, dict)):
            raise TypeError("'fit_conf' must be a string fit configuration key, or a dictionary with "
                            "changed-from-default fit function arguments as keys and changed values as items.")
        # If the input is a dictionary then we need to construct the key, as opposed to it being passed in whole
        #  as a string
        elif isinstance(fit_conf, dict):
            fit_conf = fit_conf_from_function(FIT_FUNC_MODEL_NAMES[model], fit_conf)
        elif isinstance(fit_conf, str) and fit_conf not in self._fit_results[0][model]:
            av_fconfs = ", ".join(self._fit_results[annulus_ident][model].keys())
            raise ModelNotAssociatedError("The {fc} fit configuration has not been used for any {m} fit to this "
                                          "annular spectrum; available fit configurations are "
                                          "{a}".format(fc=fit_conf, m=model, a=av_fconfs))
        # In this case the user passed no fit configuration key, but there are multiple fit configurations stored here
        elif fit_conf is None and len(self.fitted_model_configurations[model]) != 1:
            av_fconfs = ", ".join(self._fit_results[annulus_ident][model].keys())
            raise ValueError("The {m} model has been fit with multiple configuration, so fit_conf=None is not "
                             "valid; available fit configurations are {a}".format(m=model, a=av_fconfs))
        # However here they passed no fit configuration, and only one has been used for the model, so we're all good
        #  and will select it for them
        elif fit_conf is None and len(self.fitted_model_configurations[model]) == 1:
            fit_conf = self.fitted_model_configurations[model][0]

        # Have to check that the user has passed a legal annulus ID, otherwise we'll be getting key errors down the
        #  line from the dictionary accesses, and they are far less informative.
        if annulus_ident is not None and annulus_ident < 0:
            raise ValueError("Annulus IDs can only be positive.")
        elif annulus_ident is not None and annulus_ident >= self.num_annuli:
            raise ValueError("Annulus indexing starts at zero, and this AnnularSpectra only has {} "
                             "annuli.".format(self._num_ann))

        # Bunch of checks to make sure the requested results actually exist
        if annulus_ident is not None and len(self._fit_results[annulus_ident]) == 0:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this AnnularSpectra object.")
        elif annulus_ident is not None and model not in self._fit_results[annulus_ident]:
            av_mods = ", ".join(self._fit_results[annulus_ident].keys())
            raise ModelNotAssociatedError("{m} has not been fitted to this AnnularSpectra; available "
                                          "models are {a}".format(m=model, a=av_mods))
        elif (annulus_ident is not None and par is not None and
              par not in self._fit_results[annulus_ident][model][fit_conf]):
            av_pars = ", ".join(self._fit_results[annulus_ident][model][fit_conf].keys())
            raise ParameterNotAssociatedError("{p} was not a free parameter in the {m}-{fc} fit to this "
                                              "AnnularSpectra; available parameters are "
                                              "{a}".format(p=par, m=model, a=av_pars, fc=fit_conf))

        return model, fit_conf

    def get_results(self, annulus_ident: int, model: str = None, par: str = None, fit_conf: Union[str, dict] = None):
        """
        Important method that will retrieve fit results from the AnnularSpectra object. Either for a specific
        parameter of the supplied model combination, or for all of them. If a specific parameter is requested,
        all matching values from the fit will be returned in an N row, 3 column numpy array (column 0 is the value,
        column 1 is err-, and column 2 is err+). If no parameter is specified, the return will be a dictionary
        of such numpy arrays, with the keys corresponding to parameter names.

        If no model name is supplied, but only one model has been fit to this annular spectrum, then that model will
        be automatically selected - this behavior also applies to the fit configuration (fit_conf) parameter; if a
        model was only fit with one fit configuration then that will be automatically selected.

        :param int annulus_ident: The annulus for which you wish to retrieve the fit results.
        :param str model: The name of the fitted model that you're requesting the results from
            (e.g. constant*tbabs*apec).
        :param str par: The name of the parameter you want a result for.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: The requested result value, and uncertainties.
        """
        model, fit_conf = self._get_fit_checks(annulus_ident, model, par, fit_conf)

        # Read out into variable for readabilities sake
        fit_data = self._fit_results[annulus_ident][model][fit_conf]
        proc_data = {}  # Where the output will ive
        for p_key in fit_data:
            # Used to shape the numpy array the data is transferred into
            num_entries = len(fit_data[p_key])
            # 'Empty' new array to write out the results into, done like this because results are stored
            #  in nested dictionaries with their XSPEC parameter number as an extra key
            new_data = np.zeros((num_entries, 3))

            # If a parameter is unlinked in a fit with multiple spectra (like normalisation for instance),
            #  there can be N entries for the same parameter, writing them out in order to a numpy array
            for incr, par_index in enumerate(fit_data[p_key]):
                new_data[incr, :] = fit_data[p_key][par_index]

            # Just makes the output a little nicer if there is only one entry
            if new_data.shape[0] == 1:
                proc_data[p_key] = new_data[0]
            else:
                proc_data[p_key] = new_data

        # If no specific parameter was requested, the user gets all of them
        if par is None:
            return proc_data
        else:
            return proc_data[par]

    def get_fit_statistic(self, annulus_ident: int, model: str = None, fit_conf: Union[str, dict] = None):
        """
        Method that will retrieve fit statistic from the AnnularSpectra object. If no model name is supplied, but
        only one model has been fit to this annular spectrum, then that model will be automatically selected - this
        behavior also applies to the fit configuration (fit_conf) parameter; if a model was only fit with one fit
        configuration then that will be automatically selected.

        :param int annulus_ident: The annulus for which you wish to retrieve the fit statistic.
        :param str model: The name of the fitted model that you're requesting the fit statistic of
            (e.g. constant*tbabs*apec).
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: The requested fit statistic.
        """
        model, fit_conf = self._get_fit_checks(annulus_ident, model, None, fit_conf)

        return self._fit_stat[annulus_ident][model][fit_conf]

    def get_test_statistic(self, annulus_ident: int, model: str = None, fit_conf: Union[str, dict] = None):
        """
        Method that will retrieve test statistic from the AnnularSpectra object. If no model name is supplied, but
        only one model has been fit to this annular spectrum, then that model will be automatically selected - this
        behavior also applies to the fit configuration (fit_conf) parameter; if a model was only fit with one fit
        configuration then that will be automatically selected.

        :param int annulus_ident: The annulus for which you wish to retrieve the test statistic.
        :param str model: The name of the fitted model that you're requesting the test statistic of
            (e.g. constant*tbabs*apec).
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: The requested test statistic.
        """
        model, fit_conf = self._get_fit_checks(annulus_ident, model, None, fit_conf)

        return self._test_stat[annulus_ident][model][fit_conf]

    def get_luminosities(self, annulus_ident: int, model: str = None, lo_en: Quantity = None, hi_en: Quantity = None,
                         fit_conf: Union[str, dict] = None) -> Union[Quantity, Dict[str, Quantity]]:
        """
        This will retrieve luminosities of specific annuli from fits performed on this AnnularSpectra object.
        A model name must be supplied, and if a luminosity from a specific energy range is desired then lower
        and upper energy bounds may be passed.

        If no model name is supplied, but only one model has been fit to this annular spectrum, then that model
        will be automatically selected - this behavior also applies to the fit configuration (fit_conf) parameter; if
        a model was only fit with one fit configuration then that will be automatically selected.

        :param int annulus_ident: The annulus for which you wish to retrieve the luminosities.
        :param str model: The name of the fitted model that you're requesting the luminosity of
            (e.g. constant*tbabs*apec).
        :param Quantity lo_en: The lower energy limit for the desired luminosity measurement.
        :param Quantity hi_en: The upper energy limit for the desired luminosity measurement.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: The requested luminosity value, and uncertainties. If a specific energy range has been supplied
            then a quantity containing the value (col 1), -err (col 2), and +err (col 3), will be returned. If no
            energy range is supplied then a dictionary of all available luminosity quantities will be returned.
        :rtype: Union[Quantity, Dict[str, Quantity]]
        """
        # Checking the input energy limits are valid, and assembles the key to look for lums in those energy
        #  bounds. If the limits are none then so is the energy key
        if all([lo_en is not None, hi_en is not None]) and lo_en > hi_en:
            raise ValueError("The low energy limit cannot be greater than the high energy limit")
        elif all([lo_en is not None, hi_en is not None]):
            en_key = "bound_{l}-{u}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value)
        else:
            en_key = None

        # Checking the model fit retrieval arguments that were passed in
        model, fit_conf = self._get_fit_checks(annulus_ident, model, None, fit_conf)

        # Checks that the requested region, model and energy band actually exist
        if len(self._luminosities[annulus_ident]) == 0:
            raise ModelNotAssociatedError("There are no XSPEC fits associated with this AnnularSpectra")
        elif model not in self._luminosities[annulus_ident]:
            av_mods = ", ".join(self._luminosities[annulus_ident].keys())
            raise ModelNotAssociatedError("{m} has not been fitted to this AnnularSpectra; "
                                          "available models are {a}".format(m=model, a=av_mods))
        elif en_key is not None and en_key not in self._luminosities[annulus_ident][model]:
            av_bands = ", ".join([en.split("_")[-1] + "keV" for en in self._luminosities[annulus_ident][model].keys()])
            raise ParameterNotAssociatedError("A luminosity within {l}-{u}keV was not measured for the fit "
                                              "with {m}; available energy bands are "
                                              "{b}".format(l=lo_en.to("keV").value, u=hi_en.to("keV").value, m=model,
                                                           b=av_bands))

        # If no limits specified,the user gets all the luminosities, otherwise they get the one they asked for
        if en_key is None:
            parsed_lums = {}
            for lum_key in self._luminosities[annulus_ident][model][fit_conf]:
                lum_value = self._luminosities[annulus_ident][model][fit_conf][lum_key]
                parsed_lum = Quantity([lum.value for lum in lum_value], lum_value[0].unit)
                parsed_lums[lum_key] = parsed_lum
            return parsed_lums
        else:
            lum_value = self._luminosities[annulus_ident][model][fit_conf][en_key]
            parsed_lum = Quantity([lum.value for lum in lum_value], lum_value[0].unit)
            return parsed_lum

    def generate_profile(self, model: str, par: str, par_unit: Union[Unit, str], upper_limit: Quantity = None,
                         fit_conf: Union[str, dict] = None) \
            -> Union[BaseProfile1D, ProjectedGasTemperature1D, ProjectedGasMetallicity1D]:
        """
        This generates a radial profile of the requested fit parameter using the stored results from
        an XSPEC model fit run on this AnnularSpectra. The profile is added to AnnularSpectra internal
        storage, and also returned to the user.

        :param str model: The name of the fitted model you wish to generate a profile from.
        :param str par: The name of the free model parameter that you wish to generate a profile for.
        :param Unit/str par_unit: The unit of the free model parameter as an astropy unit object, or a string
            representation (e.g. keV).
        :param Quantity upper_limit: Allows an allowed upper limit for the y values in the profile to be passed.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit method
            and values being the changed values (only values changed-from-default need be included) or a full string
            representation of the fit configuration that is being requested.
        :return: The requested profile object.
        :rtype: Union[BaseProfile1D, ProjectedGasTemperature1D, ProjectedGasMetallicity1D]
        """
        # If a string representation was passed, we make it an astropy unit
        if isinstance(par_unit, str):
            par_unit = Unit(par_unit)

        if self.proper_radii is None:
            raise UnitConversionError("Currently proper radius units are required to generate "
                                      "profiles, please assign some using the proper_radii property.")

        # This is somewhat redundant, as we run get_results in the loop, but we want the checked fit_conf value
        model, fit_conf = self._get_fit_checks(None, model, par, fit_conf)

        par_data = {}
        for ai in range(self._num_ann):
            # We read it out into an interim parameter
            cur_data = self.get_results(ai, model, par, fit_conf)
            # In cases where the parameter in question wasn't linked across separate spectra there will be a
            #  measurement for each spectrum per annulus
            if cur_data.ndim != 1:
                # There are multiple values available here, and we want to sort them out into the ObsID-instrument
                #  combinations
                obs_order = self._obs_order[ai][model][fit_conf]
                for i in range(cur_data.shape[0]):
                    obs_inst = "-".join(obs_order[i])
                    # This was we create a profile for each ObsID-Instrument combination
                    if obs_inst not in par_data:
                        par_data[obs_inst] = [cur_data[i, :]]
                    else:
                        par_data[obs_inst].append(cur_data[i, :])
            # If the quantity of interest was linked across spectra then we just store it under a 'combined' key
            elif cur_data.ndim == 1 and 'combined' not in par_data:
                par_data['combined'] = [cur_data]
            elif cur_data.ndim == 1 and 'combined' in par_data:
                par_data['combined'].append(cur_data)

        # For storing the profiles generated here
        profs = []
        # If there is only one profile to be generated (which means the quantity of interest was linked across
        #  spectra during the fit), then this will just iterate once and obs_key will be combined
        for obs_key in par_data:
            # Just makes the read out values into an astropy quantity
            par_quant = Quantity(par_data[obs_key], par_unit)
            par_val = par_quant[:, 0]
            # Extract the parameter uncertainties, and average because profiles currently only accept 1D errors
            par_errs = par_quant[:, 1:]
            par_errs = np.average(par_errs, axis=1)

            mid_radii = self.proper_annulus_centres.to("kpc")
            mid_radii_deg = self.annulus_centres.to("deg")
            # calculates radii errors, basically the extent of the bins
            rad_errors = Quantity(np.diff(self.proper_radii.to('kpc').value, axis=0) / 2, 'kpc')

            # Here we set up the ObsID and instrument information
            if obs_key == 'combined':
                obs_id = 'combined'
                inst = 'combined'
            else:
                # The obs key was made up of the ObsID and instrument joined by a -
                obs_id, inst = obs_key.split('-')

            try:
                if par == 'kT' and upper_limit is None:
                    new_prof = ProjectedGasTemperature1D(mid_radii, par_val, self.central_coord, self.src_name, obs_id,
                                                         inst, rad_errors, par_errs, associated_set_id=self.set_ident,
                                                         set_storage_key=self.storage_key, deg_radii=mid_radii_deg,
                                                         auto_save=True, spec_model=model, fit_conf=fit_conf)
                elif par == 'kT' and upper_limit is not None:
                    new_prof = ProjectedGasTemperature1D(mid_radii, par_val, self.central_coord, self.src_name, obs_id,
                                                         inst, rad_errors, par_errs, upper_limit, self.set_ident,
                                                         self.storage_key, deg_radii=mid_radii_deg, auto_save=True,
                                                         spec_model=model, fit_conf=fit_conf)
                elif par == 'Abundanc':
                    new_prof = ProjectedGasMetallicity1D(mid_radii, par_val, self.central_coord, self.src_name, obs_id,
                                                         inst, rad_errors, par_errs, self.set_ident, self.storage_key,
                                                         mid_radii_deg, auto_save=True, spec_model=model,
                                                         fit_conf=fit_conf)
                elif par == 'norm':
                    new_prof = APECNormalisation1D(mid_radii, par_val, self.central_coord, self.src_name, obs_id, inst,
                                                   rad_errors, par_errs, self.set_ident, self.storage_key,
                                                   mid_radii_deg, auto_save=True, spec_model=model, fit_conf=fit_conf)
                else:
                    prof_type = "1d_proj_{}"
                    new_prof = Generic1D(mid_radii, par_val, self.central_coord, self.src_name, obs_id, inst, par,
                                         prof_type.format(par), rad_errors, par_errs, self.set_ident, self.storage_key,
                                         mid_radii_deg, auto_save=True, spec_model=model, fit_conf=fit_conf)

                profs.append(new_prof)

            # This gets triggered if any funny values are present in the quantities passed to the profile declaration.
            #  Infinite/NaN values, negative errors (which can happen in XSPEC fits) etc.
            except ValueError:
                profs.append(None)

        if len(profs) == 1:
            profs = profs[0]

        return profs

    def view_cross_arfs(self, src_ann_id: int, obs_id: str, inst: str, src_arf_norm: bool = False,
                        figsize: tuple = (8, 6), xscale: str = 'log', yscale: str = 'linear',
                        lo_en: Quantity = Quantity(0.01, 'keV'), hi_en: Quantity = Quantity(16.0, 'keV')):
        """
        A method that produces a plot of the cross-arf curves for a specified source annulus spectrum, for a specified
        ObsID and instrument. The original source spectrum will also be plotted as a reference. This method can also
        be used to create 'normalized' curves were the cross-arf effective area values are divided by the source
        arf effective areas.

        :param int src_ann_id: The annulus ID of the source annulus for which you wish to plot cross-arf curves.
        :param str obs_id: The ObsID of the spectrum for which you wish to plot cross-arf curves.
        :param str inst: The instrument of the spectrum for which you wish to plot cross-arf curves.
        :param bool src_arf_norm:
        :param tuple figsize: The size of the figure, the default is (8, 6).
        :param str xscale: The xscale to use for the plot.
        :param str yscale: The yscale to use for the plot.
        :param Quantity lo_en: The lower energy limit for the x-axis. The default is 0.01 keV. This will be altered
            to reflect the minimum value of the energy scale for this curve if lo_en is smaller than the lowest
            energy bin.
        :param Quantity hi_en: The upper energy limit for the x-axis. The default is 16.0 keV. This will be altered
            to reflect the maximum value of the energy scale for this curve if hi_en is greater than the highest
            energy bin.
        """
        plt.figure(figsize=figsize)
        # Set the plot up to look nice and professional.
        ax = plt.gca()
        ax.minorticks_on()
        ax.tick_params(axis='both', direction='in', which='both', top=True, right=True)

        # I retrieve the energies and effective areas for the source arf from the source spectrum specified by the
        #  user input of src_ann_id, ObsID, and instrument. This will be used to provide context to the cross-arf
        #  plots (or to normalize them if src_arf_norm is set to True.
        src_spec = self.get_spectra(src_ann_id, obs_id, inst)
        # Just take the midpoint of the energy bins
        src_ens = (src_spec.eff_area_hi_en + src_spec.eff_area_lo_en) / 2
        src_eff_areas = src_spec.eff_area

        # Have to check the input energy bounds to make sure that they are sensible
        if lo_en >= hi_en:
            raise ValueError("The 'hi_en' argument cannot be greater than or equal to the 'lo_en' argument.")
        else:
            lo_en = lo_en.to("keV").value
            hi_en = hi_en.to("keV").value

        # We dynamically alter the upper and lower energy bounds passed by the user to reflect the actual maxima and
        #  minima of the energy scale, so there isn't a ton of blank space in on either side if they made a bad
        #  choice, or my default value choices were bad. We'll be plotting a bunch of arfs here, but I use the
        #  original source arf as the comparison (though the energies should be identical for a given instrument).
        if lo_en < src_ens.value.min():
            lo_en = src_ens.value.min()
        if hi_en > src_ens.value.max():
            hi_en = src_ens.value.max()

        # As we know, log scales don't like zero values, so I make a change if the user has set the minimum
        #  energy to be zero and set an x-axis log scale
        if xscale == 'log' and lo_en == 0:
            warn("The x-axis scale has been set to log, and 'lo_en' cannot be zero - it has been set to 0.01 "
                 "keV.", stacklevel=2)
            lo_en = 0.01

        # Now I grab the dictionaries of lower/upper energy bounds, and effective areas, for the cross-arfs that we
        #  wish to plot in this method
        all_lo_ens = self.get_cross_arf_lo_ens(obs_id, inst, src_ann_id)
        all_hi_ens = self.get_cross_arf_hi_ens(obs_id, inst, src_ann_id)
        all_eff_areas = self.get_cross_arf_eff_areas(obs_id, inst, src_ann_id)

        if not src_arf_norm:
            # Before I get to plotting the cross-arf curves, I plot the original source arf as a dashed black line
            plt.plot(src_ens[(src_ens.value >= lo_en) & (src_ens.value <= hi_en)],
                     src_eff_areas[(src_ens.value >= lo_en) & (src_ens.value <= hi_en)],
                     color='black', linestyle='dashed', label='Source annulus')
        # In this case though we are going to normalize the cross-arfs by the source arf, so actually we just
        #  want to plot a straight line with 1 on the y-axis
        else:
            # As I'm normalising by the source ARF here, I just plot a line at 1
            plt.plot(src_ens[(src_ens.value >= lo_en) & (src_ens.value <= hi_en)].value,
                     [1]*len(src_eff_areas[(src_ens.value >= lo_en) & (src_ens.value <= hi_en)]),
                     color='black', linestyle='dashed', label='Source reference')

        min_norm = 1
        for cross_ann_id, eff_area in all_eff_areas.items():
            ens = (all_hi_ens[cross_ann_id] + all_lo_ens[cross_ann_id]) / 2

            # Get the data and plot it
            sel_ens = (ens.value >= lo_en) & (ens.value <= hi_en)
            if not src_arf_norm:
                plt.plot(ens[sel_ens], eff_area[sel_ens], label='Annulus {} contribution'.format(cross_ann_id))
            else:
                norm_area = Quantity(np.zeros(len(eff_area)))
                np.divide(eff_area, src_eff_areas, out=norm_area, where=src_eff_areas != 0)
                plt.plot(ens[sel_ens], norm_area[sel_ens], label='Annulus {} normalised'.format(cross_ann_id))
                # We compare to the global min norm area to see whether we have a smaller value here or not - this
                #  will be used to set the minimum y value
                min_norm = min(min(norm_area[norm_area > 0]), min_norm)

        if not src_arf_norm:
            # Set the lower y-lim to be 1, and then the user supplied x-lims (supplementing the fact that we've already
            #  used those limits to select the data to plot
            plt.ylim(1)
        else:
            # In the case where we're normalizing, setting the lower y limit to 1 doesn't make sense.
            plt.ylim(min_norm)
        plt.xlim(lo_en, hi_en)

        # Set the user defined x and y scales
        plt.xscale(xscale)
        plt.yscale(yscale)

        # Title and axis labels
        if not src_arf_norm:
            plt.ylabel("Effective Area [cm$^{2}$]", fontsize=12)
        else:
            plt.ylabel("Normalised Effective Area", fontsize=12)

        plt.xlabel("Energy [keV]", fontsize=12)
        plt.title("Annulus {ai} {o}-{i} Cross-ARFs".format(o=obs_id, i=inst.upper(), ai=src_ann_id), fontsize=14)

        # This makes sure that the tick labels are formatted as 0.1, 1, 10, etc. keV on the x-axis (if logged) and
        #  10, 100, 100, 1000, etc. cm^2 on the y-axis if logged
        ax.xaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda inp, _: '{:g}'.format(inp)))

        # Make sure we add the legend in, it has a lot of context regarding which curve is for which cross-annulus
        plt.legend(loc='best', fontsize=12)

        # Aaaand finally actually plot it
        plt.tight_layout()
        plt.show()

    def view_annulus(self, ann_ident: int, figsize: Tuple = (10, 7), lo_lim: Quantity = Quantity(0.3, "keV"),
                     hi_lim: Quantity = Quantity(7.9, "keV"), back_sub: bool = True, energy: bool = True,
                     src_colour: str = 'black', bck_colour: str = 'firebrick', grouped: bool = True,
                     xscale: str = "log", yscale: str = "linear", fontsize: Union[int, float] = 14,
                     show_model_fits: bool = True, save_path: str = None, model: str = None,
                     fit_conf: Union[str, dict] = None):
        """
        A view method that allows all spectra from a particular annulus to be displayed on the same axis.

        A spectrum can be viewed prior to fitting, and this method will produce plots that should be the same as the
        XSPEC count/s/keV (or channel) spectrum views. If a model has been fit, and the user wishes to display it, then
        the 'normalised count/s/keV' that are plotted are extracted from the XSPEC data, rather than assembled in this
        method.

        :param int ann_ident: The integer identifier of the annulus you wish to see spectra for.
        :param tuple figsize: The desired size of the output figure, default is (10, 7).
        :param Quantity lo_lim: The lower limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 0.3 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param Quantity hi_lim: The upper limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 7.9 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param bool back_sub: Whether the plotted data should have their background subtracted, default is True.
        :param bool energy: Controls whether the x-axis is in units of energy, default is True. If False then
            channels are plotted instead.
        :param str src_colour: The colour in which to plot the source spectrum. Default is 'black'.
        :param str bck_colour: The colour in which to plot the background spectrum. Default is 'firebrick' red.
        :param bool grouped: Whether the grouped spectrum should be plotted, default is True. If the spectrum has not
            been grouped then this be automatically set to False.
        :param str xscale: The scaling to be applied to the x-axis, default is 'log'.
        :param str yscale: The scaling to be applied to the y-axis, default is 'linear'.
        :param int/float fontsize: The fontsize for axis labels. The legend fontsize will be fontsize - 1. The title
            fontsize will be fontsize + 1. Default is 14.
        :param bool show_model_fits: Whether any models fit to the spectrum by XSPEC should be shown. Default is
            True, but will be set to False if no fits have been performed.
        :param str save_path: The path where the figure produced by this method should be saved. Default is None, in
            which case the figure will not be saved.
        :param str model: This parameter allows you to specify a particular model to plot (if show_model_fits is
            True). Default is None, in which case all models will be shown (if available).
        :param str/dict fit_conf: This parameter allows you to specify a particular fit configuration of a model to
            plot (if 'show_model_fits' is True and 'model' is set). Pass either a dictionary with keys being the names
            of parameters passed to the XGA XSPEC fit function that were changed from default, and values being the
            changed values, or a full string representation of the fit configuration that is being requested. Default
            is None, in which case all fit configurations of a model will be plotted.
        """
        # Grabs the relevant spectra using the annular ident
        rel_spec = self.get_spectra(ann_ident)

        # Create figure object
        plt.figure(figsize=figsize)
        fig = plt.figure(figsize=figsize)
        ax = plt.gca()
        for sp in rel_spec:
            ax = sp.get_view(ax, lo_lim, hi_lim, back_sub, energy, src_colour, bck_colour, grouped, xscale, yscale,
                             fontsize, show_model_fits, model, fit_conf)

        # Generate the legend for the data and model(s)
        # plt.legend(loc="best", fontsize=fontsize - 1)

        # Removing extraneous whitespace around the plot
        plt.tight_layout()

        # If the user passed a save_path value, then we assume they want to save the figure
        if save_path is not None:
            plt.savefig(save_path)

        # Display the spectrum
        plt.show()

        # Wipe the figure
        plt.close("all")

    def view_annuli(self, obs_id: str, inst: str, model: str = None, fit_conf: Union[str, dict] = None,
                    elevation_angle: int = 30, azimuthal_angle: int = -60, figsize: Tuple = (12, 8),
                    lo_lim: Quantity = Quantity(0.3, "keV"), hi_lim: Quantity = Quantity(7.9, "keV"),
                    back_sub: bool = True, energy: bool = True, src_colour: str = 'black',
                    bck_colour: str = 'firebrick', grouped: bool = True, fontsize: Union[int, float] = 14):
        """
        This view method is one of several in the AnnularSpectra class, and will display data and associated model
        fits for a single ObsID-Instrument combination for all annuli in this AnnularSpectra, in a 3D plot. The
        output of this can be quite visually confusing, so you may wish to use view_annulus to see the spectrum
        of a particular annulus for a particular ObsID-Instrument in a more traditional way, or just view to see all
        model fits at all annuli.

        :param str obs_id: The ObsID of the spectra to display.
        :param str inst: The instrument of the spectra to display.
        :param str model: This parameter allows you to specify a particular model to plot. Default is None, in which
            case a model will be automatically selected if only one has been fit, or no model will be shown if
            one has not.
        :param str/dict fit_conf: This parameter allows you to specify a particular fit configuration of a model to
            plot (if 'show_model_fits' is True and 'model' is set). Pass either a dictionary with keys being the names
            of parameters passed to the XGA XSPEC fit function that were changed from default, and values being the
            changed values, or a full string representation of the fit configuration that is being requested. The
            default is None - if only one fit configuration has been run for the model, then that will be automatically
            selected.
        :param tuple figsize: The size of the figure.
        :param int elevation_angle: The elevation angle in the z plane, in degrees.
        :param int azimuthal_angle: The azimuth angle in the x,y plane, in degrees.
        :param Quantity lo_lim: The lower limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 0.3 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param Quantity hi_lim: The upper limit applied to the plot, either a unitless Quantity (representing
            channels) or an energy Quantity. Limits will be automatically converted to the units of the x-axis.
            Default is 7.9 keV, matching the default lower limit of the XGA implementation of XSPEC fitting.
        :param bool back_sub: Whether the plotted data should have their background subtracted, default is True.
        :param bool energy: Controls whether the x-axis is in units of energy, default is True. If False then
            channels are plotted instead.
        :param str src_colour: The colour in which to plot the source spectrum. Default is 'black'.
        :param str bck_colour: The colour in which to plot the background spectrum. Default is 'firebrick' red.
        :param bool grouped: Whether the grouped spectrum should be plotted, default is True. If the spectrum has not
            been grouped then this be automatically set to False.
        :param str xscale: The scaling to be applied to the x-axis, default is 'log'.
        :param str yscale: The scaling to be applied to the y-axis, default is 'linear'.
        :param int/float fontsize: The fontsize for axis labels. The legend fontsize will be fontsize - 1. The title
            fontsize will be fontsize + 1. Default is 14.
        """
        from ..xspec.fit import FIT_FUNC_MODEL_NAMES
        from ..xspec.fitconfgen import fit_conf_from_function

        # Setup the figure as we normally would
        fig = plt.figure(figsize=figsize)
        # This subplot with a 3D projection is what allows us to make a 3-axis plot
        ax = fig.add_subplot(111, projection='3d')
        # We use the user's passed in angle values to set the perspective that we have on the plot.
        ax.view_init(elevation_angle, azimuthal_angle)
        # Set a relevant title
        plt.title("{sn} - {o}-{i} Annular Spectra".format(sn=self.src_name, o=obs_id, i=inst))

        # This just checks whether the grouped argument to this method is compatible with whether the spectrum
        #  associated with this Spectrum instance has actually been grouped - if not then we automatically
        #  set the method argument to False
        if not self.grouped:
            grouped = False

        # This just ensures that everything works if someone has passed an integer for the channel limits
        lo_lim = Quantity(lo_lim)
        hi_lim = Quantity(hi_lim)

        # Performing checks on the limits
        if lo_lim >= hi_lim:
            raise ValueError("The hi_lim argument cannot be less than or equal to the lo_lim argument")

        # Keep the original values of model and fit_conf - they can change in the loop
        og_model = deepcopy(model)
        og_fit_conf = deepcopy(fit_conf)
        # Same deal with the energy limits
        og_lo_lim = lo_lim.copy()
        og_hi_lim = hi_lim.copy()

        # We iterate through all the annuli
        for ann_ident in range(0, self._num_ann):
            # The value of model and fit_conf will likely be changed as part of this iteration, so we reset them to
            #  the og values
            model = deepcopy(og_model)
            fit_conf = deepcopy(og_fit_conf)
            lo_lim = og_lo_lim.copy()
            hi_lim = og_hi_lim.copy()

            spec = self.get_spectra(ann_ident, obs_id, inst)

            # These just make sure that limits in units of either channel or energy are converted appropriately to what
            #  we're plotting on the x-axis, channels or energies.
            if not energy and lo_lim.unit != '':
                lo_lim = spec.conv_channel_energy(lo_lim)
            if not energy and hi_lim.unit != '':
                hi_lim = spec.conv_channel_energy(hi_lim)
            if energy and not lo_lim.unit.is_equivalent('keV'):
                lo_lim = spec.conv_channel_energy(lo_lim)
            if energy and not hi_lim.unit.is_equivalent('keV'):
                hi_lim = spec.conv_channel_energy(hi_lim)

            # Reads out the values of the limits as matplotlib sometimes gets upset by astropy quantities
            if energy:
                lo_lim = lo_lim.to("keV").value
                hi_lim = hi_lim.to("keV").value
            else:
                lo_lim = lo_lim.value
                hi_lim = hi_lim.value

            if len(self.fitted_models) > 0:
                show_model_fits = True
            else:
                show_model_fits = False

            # Now we deal with the different models/model fit configurations that can and cannot be specified
            if show_model_fits and model is None and fit_conf is not None:
                raise ValueError("Specifying a fit configuration ('fit_conf') is not supported without setting the "
                                 "'model' argument; use the 'fitted_model_configurations' property of this Spectrum to "
                                 "see which models and configurations are available.")
            elif show_model_fits and model is None and fit_conf is None:
                model = self.fitted_models
                fit_conf = self.fitted_model_configurations
            elif show_model_fits and model is not None and fit_conf is None:
                # I indent this check because it is just a bit easier for me that way
                if model not in self.fitted_models:
                    av_mods = ", ".join(self.fitted_models)
                    raise ModelNotAssociatedError("{m} has not been fitted to this Spectrum; available "
                                                  "models are {a}".format(m=model, a=av_mods))

                # If we're here then the model is valid, and in this case no fit configuration has been specified, so
                #  we grab ALL OF THEM - making sure that the structure of the parameters is the same (model in a list,
                #  fit configs in a list in a dictionary with model name as key
                fit_conf = {model: self.fitted_model_configurations[model]}
                model = [model]
            elif show_model_fits and model is not None and fit_conf is not None:
                if model not in self.fitted_models:
                    av_mods = ", ".join(self.fitted_models)
                    raise ModelNotAssociatedError("{m} has not been fitted to this Spectrum; available "
                                                  "models are {a}".format(m=model, a=av_mods))

                # If the configuration is a dictionary we need to try to turn that into a proper fit configuration key
                if isinstance(fit_conf, dict):
                    fit_conf = fit_conf_from_function(FIT_FUNC_MODEL_NAMES[model], fit_conf)

                # And now we check if the fit configuration is available to this Spectrum instance
                if fit_conf not in self.fitted_model_configurations[model]:
                    av_fconfs = ", ".join(self.fitted_model_configurations[model])
                    raise ModelNotAssociatedError(
                        "The {fc} fit configuration has not been used for any {m} fit to this "
                        "spectrum; available fit configurations are "
                        "{a}".format(fc=fit_conf, m=model, a=av_fconfs))

                fit_conf = {model: [fit_conf]}
                model = [model]

            if not grouped:
                sct = spec.count_rates.copy()
                bct = spec.back_count_rates.copy()
                if energy:
                    x_dat = spec.conv_channel_energy(spec.channels.copy()).value
                    x_wid = (spec.rmf_channels_hi_en - spec.rmf_channels_lo_en).value
                else:
                    x_dat = spec.channels.copy()
                    x_wid = 1
            else:
                grp_info = spec.get_grouped_data()
                sct = grp_info[0]
                bct = grp_info[1]
                if energy:
                    # This entry is the middle energy of each bin
                    x_dat = grp_info[-1][:, 0].value
                    # This entry is the 'error' (but really just half the width) of each energy bin
                    x_wid = grp_info[-1][:, 1].value
                else:
                    # This entry is the middle channel of each bin
                    x_dat = grp_info[4][:, 0].value
                    # This entry is the 'error' (but really just half the width) of each channel bin
                    x_wid = grp_info[4][:, 1].value

            # We check that the x limits are actually sensible values, if they are higher (for the top limit) or lower (
            #  (for the lower limit) than the data that are actually available then we nudge them to those values
            if lo_lim < x_dat.min():
                lo_lim = x_dat.min()
            if hi_lim > x_dat.max():
                hi_lim = x_dat.max()

            # We pre-select the data based on the passed lower and upper limits - first making a selection mask array
            sel_x = (x_dat <= hi_lim) & (x_dat >= lo_lim)
            # Then selecting the relevant source count, background count, and x-data (energy or channel) entries
            sct = sct[sel_x]
            bct = bct[sel_x]
            x_dat = x_dat[sel_x]
            x_wid = x_wid[sel_x]
            # This is what the y-data are divided by to make it per keV or per channel, the width of the bin essentially
            per_x = x_wid * 2

            # This uses the AREASCAL keyword (the product of EXPOSURE times AREASCAL is the exposure duration for any
            #  fully exposed pixels in each channel - my experience is that this is normally 1 for XMM products) to
            #  effectively scale the exposure time by dividing the count rate by it
            src_rate = sct / spec.header['AREASCAL']

            # This scales the background count rates by the AREASCAL (as above), but also by the ratio of BACKSCAL
            #  values, which scales the background flux to the same area as the source
            bck_rate = (spec.header['BACKSCAL'] / spec.back_header['BACKSCAL']) * (bct / spec.back_header['AREASCAL'])

            # And finally subtracting one from the other - they both have error columns which are also subtracted
            #  here (which is completely meaningless of course), but don't worry we'll fix that on the next line!
            src_sub_bck_rate = src_rate - bck_rate
            # Simple error propagation to replace the nonsense uncertainty column in src_sub_bck_rate
            src_sub_bck_rate[:, 1] = np.sqrt(src_rate[:, 1] ** 2 + bck_rate[:, 1] ** 2)

            # Depending on what radius information is available to this AnnularSpectra, depends which we use
            # We will always prefer to use proper radii if they are available
            if self.proper_radii is not None:
                # Need to set up an array for the y axis (the radius axis) which is the same dimensions
                #  as the x and z arrays
                y_fill = self.proper_annulus_centres[ann_ident].value
                chosen_unit = self.proper_radii.unit
                # patch = Rectangle((lo_lim, self.proper_radii[ann_ident].value), hi_lim-lo_lim,
                #                   self.proper_radii[ann_ident+1].value-self.proper_radii[ann_ident].value, hatch="/")
                # ax.add_patch(patch)
                # art3d.pathpatch_2d_to_3d(patch, )
                # ax.axhspan(self.proper_radii[ann_ident].value, self.proper_radii[ann_ident+1].value)
            else:
                y_fill = self.annulus_centres[ann_ident].value
                chosen_unit = self.radii.unit
                # ax.axhspan(self.radii[ann_ident].value, self.radii[ann_ident+1].value)

            if not show_model_fits:
                ys = np.full(shape=(len(x_dat),), fill_value=y_fill)

                # Plotting the data, accounting for the different combinations of x-axis and y-axis
                if back_sub:
                    # If we're going for background subtracted data, then that is all we plot
                    ax.errorbar(x_dat, ys, src_sub_bck_rate.value[:, 0] / per_x, xerr=x_wid,
                                yerr=src_sub_bck_rate.value[:, 1] / per_x, fmt="+", color=src_colour,
                                label="Background subtracted source data", zorder=1)
                else:
                    # But if we're not wanting background subtracted, we need to plot the source and background spectra
                    ax.errorbar(x_dat, ys, src_rate.value[:, 0] / per_x, xerr=x_wid, yerr=src_rate.value[:, 1] / per_x,
                                fmt="+",
                                color=src_colour, label="Source data", zorder=1)
                    ax.errorbar(x_dat, ys, bck_rate.value[:, 0] / per_x, xerr=x_wid, yerr=bck_rate.value[:, 1] / per_x,
                                fmt="x",
                                color=bck_colour, label="Background data", zorder=1)

                # Energy vs channel has already been encoded in the x data, but we still need to plot different
                #  axis labels
                if energy:
                    ax.set_ylabel("Counts s$^{-1}$ keV$^{-1}$", fontsize=fontsize)
                    ax.set_xlabel("Energy [keV]", fontsize=fontsize)
                else:
                    ax.set_ylabel("Counts s$^{-1}$ Channel$^{-1}$", fontsize=fontsize)
                    ax.set_xlabel("Channel", fontsize=fontsize)

            # In this case the user wants the fitted spectra, and there ARE fits to plot, so rather than plot our own
            #  calculated values we plot the normalised counts/s/keV (or channel) that were extracted from XSPEC
            else:
                # Set the axis labels
                ax.set_ylabel("Normalised Counts s$^{-1}$ keV$^{-1}$", fontsize=fontsize)
                ax.set_xlabel("Energy [keV]", fontsize=fontsize)

                plot_cnt = 0
                for mod in model:
                    # We also iterate through the different fit configurations for the current model, and plot them
                    #  separately - currently with the only the model name in the legend
                    for fc in fit_conf[mod]:

                        cur_fit_data = spec.get_plot_data(mod, fc)

                        # Extract the x values which we gathered from XSPEC (they will be in keV)
                        x = cur_fit_data["x"]
                        # Cut the x dataset to just the energy range we want
                        sel_x = (x > lo_lim) & (x < hi_lim)
                        plot_x = x[sel_x]

                        ys = np.full(shape=(len(x),), fill_value=y_fill)

                        if plot_cnt == 0:
                            # Read out the data just for line length reasons
                            # Make the cuts based on energy values supplied to the view method
                            plot_y = cur_fit_data["y"][sel_x]
                            plot_xerr = cur_fit_data["x_err"][sel_x]
                            plot_yerr = cur_fit_data["y_err"][sel_x]
                            plot_mod = cur_fit_data["model"][sel_x]
                            if ann_ident == 0:
                                ax.errorbar(plot_x, ys, plot_y, xerr=plot_xerr, yerr=plot_yerr, fmt="k+",
                                            label="Background subtracted source data", zorder=1)
                            else:
                                ax.errorbar(plot_x, ys, plot_y, xerr=plot_xerr, yerr=plot_yerr, fmt="k+", zorder=1)
                            plot_cnt += 1
                        else:
                            # Don't want to re-plot data points as they should be identical, so if there is another model
                            #  only it will be plotted
                            plot_mod = cur_fit_data["model"][sel_x]

                        # The model line is put on
                        changed = spec.fitted_model_configuration_diffs[mod][fc]
                        fc_str = "; ".join([par + " - " + val for par, val in changed.items()])
                        ax.plot(plot_x, ys, plot_mod, label=mod + '; ' + fc_str, linewidth=1.5)

            # data_line = ax.plot(plot_x, ys, all_plot_data['y'], '+', alpha=0.5)
            # mod_line = ax.plot(plot_x, ys, plot_mod, alpha=0.5, linewidth=2, color=data_line[0].get_color())

        # Simply setting x-label and limits, don't currently scale this axis with log (though I would like to),
        #  because the 3D version of matplotlib doesn't easily support it
        ax.set_xlabel("Energy [keV]")
        # ax.set_xlim3d(plot_x.min(), plot_x.max())

        # Setting the lower limit of the z axis to zero, but leaving the top end open
        ax.set_zlim3d(0)
        ax.set_zlabel("Normalised Counts s$^{-1}$ keV$^{-1}$")

        # Setting up y label (with dynamic unit) and the correct radius limits
        ax.set_ylabel("Radius [{u}]".format(u=chosen_unit.to_string()))
        if self.proper_radii is not None:
            y_lims = [self.proper_annulus_centres.value[0], self.proper_radii.value[-1]]
        else:
            y_lims = [self.annulus_centres.value[0], self.proper_radii.value[-1]]
        ax.set_ylim3d(y_lims)

        # plt.tight_layout()
        plt.legend()
        ax.set_box_aspect(aspect=(5.5, 4, 3), zoom=0.86)
        plt.show()
        plt.close('all')

    def view(self, model: str = None, fit_conf: Union[str, dict] = None, figsize: tuple = (12, 8),
             elevation_angle: int = 30, azimuthal_angle: int = -60):
        """
        This view method is one of several in the AnnularSpectra class, and will display model fits to
        all spectra for each annuli in a 3D plot. No data is displayed in this viewing method, primarily
        because it's so visually confusing. If you wish to see model fits displayed over actual data in this style,
        please use view_annuli.

        :param str model: The model fit to display. The default is None, in which case if one model has been fit
            then it will be automatically selected. If multiple models have been fit then a model name must
            be supplied.
        :param str/dict fit_conf: Either a dictionary with keys being the names of parameters passed to the fit
            method and values being the changed values (only values changed-from-default need be included) or a
            full string representation of the fit configuration that is being requested. Default is None, and if
            only one fit configuration has been run for the model then it will be automatically selected, otherwise
            it will need to be specified.
        :param tuple figsize: The size of the figure.
        :param int elevation_angle: The elevation angle in the z plane, in degrees.
        :param int azimuthal_angle: The azimuth angle in the x,y plane, in degrees.
        """

        # This is a complete bodge, but just putting it here stops my IDE (PyCharm), from removing the import when it
        #  commits, because it's trying to be clever. It's a behaviour I normally appreciate, but not here.
        Axes3D

        # Setup the figure as we normally would
        fig = plt.figure(figsize=figsize)
        # This subplot with a 3D projection is what allows us to make a 3-axis plot
        ax = fig.add_subplot(111, projection='3d')
        # We use the user's passed in angle values to set the perspective that we have on the plot.
        ax.view_init(elevation_angle, azimuthal_angle)
        # Set a relevant title
        plt.title("{sn} - Annular Spectra Folded Models".format(sn=self.src_name))

        # Storing the original model and fit_conf values
        og_model = deepcopy(model)
        og_fit_conf = deepcopy(fit_conf)

        # The colour dictionary is to store a colour for a specific ObsID-instrument combo once its
        #  first been plotted - this is because we want the same ObsID-instrument combos to have the same colours
        #  for all annuli
        colour_dict = {}
        # Set up lists to hold line handlers and labels for the legend we add at the end
        handlers = []
        labels = []
        # We iterate through all the annuli
        for ann_ident in range(0, self._num_ann):
            # Restoring the original values of model and fit_conf
            model = og_model
            fit_conf = deepcopy(og_fit_conf)
            # Run the model checks
            model, fit_conf = self._get_fit_checks(ann_ident, model, None, fit_conf)

            # Remember the instruments property is a dictionary of ObsID: {instruments}, which is why we do a nested
            #  for loop like we do here
            for o in self.instruments:
                if o not in colour_dict:
                    colour_dict[o] = {}
                for i in self.instruments[o]:
                    # If we've not gone over this instrument for this ObsID before, we need to add it to
                    #  the colour dictionary for the current ObsID
                    if i not in colour_dict[o]:
                        colour_dict[o][i] = None

                    # Simply grabbing the single spectrum which is from the current ObsID and instrument, and is for
                    #  the current annulus
                    spec = self.get_spectra(ann_ident, o, i)

                    try:
                        all_plot_data = spec.get_plot_data(model, fit_conf)
                    except ModelNotAssociatedError:
                        continue

                    # Gets x data and model data
                    plot_x = all_plot_data["x"]
                    plot_mod = all_plot_data["model"]

                    # Depending on what radius information is available to this AnnularSpectra, depends which we use
                    # We will always prefer to use proper radii if they are available
                    if self.proper_radii is not None:
                        # Need to set up an array for the y axis (the radius axis) which is the same dimensions
                        #  as the x and z arrays
                        ys = np.full(shape=(len(plot_x), ), fill_value=self.proper_annulus_centres[ann_ident].value)
                        chosen_unit = self.proper_radii.unit
                    else:
                        ys = np.full(shape=(len(plot_x), ), fill_value=self.annulus_centres[ann_ident].value)
                        chosen_unit = self.radii.unit

                    # If we've not already plotted a line for this ObsID-Instrument combo, the behaviour is different
                    if colour_dict[o][i] is None:
                        mod_line = ax.plot(plot_x, ys, plot_mod, alpha=0.6, linewidth=1.5)
                        # We add the colour chosen by the colour cycle to our colour dict
                        colour_dict[o][i] = mod_line[0].get_color()
                        # Also add the line object to the handlers list and a label to the labels list - this
                        #  only needs to happen once because we only want one entry in the legend per
                        #  ObsID-Instrument combo
                        handlers.append(mod_line[0])
                        labels.append("{o}-{i}".format(o=spec.obs_id, i=spec.instrument))
                    else:
                        ax.plot(plot_x, ys, plot_mod, color=colour_dict[o][i], alpha=0.6, linewidth=1.5)

        # Simply setting x-label and limits, don't currently scale this axis with log (though I would like to),
        #  because the 3D version of matplotlib doesn't easily support it
        ax.set_xlabel("Energy [keV]")
        ax.set_xlim3d(plot_x.min(), plot_x.max())

        # Setting the lower limit of the z axis to zero, but leaving the top end open
        ax.set_zlim3d(0)
        ax.set_zlabel("Normalised Counts s$^{-1}$ keV$^{-1}$")

        # Setting up y label (with dynamic unit) and the correct radius limits
        ax.set_ylabel("Radius [{u}]".format(u=chosen_unit.to_string()))
        if self.proper_radii is not None:
            y_lims = [self.proper_annulus_centres.value[0], self.proper_radii.value[-1]]
        else:
            y_lims = [self.annulus_centres.value[0], self.proper_radii.value[-1]]
        ax.set_ylim3d(y_lims)

        # Sets up the legend so that matching data point and models are on the same line in the legend
        ax.legend(handles=handlers, labels=labels, handler_map={tuple: legend_handler.HandlerTuple(None)}, loc='best')

        # plt.tight_layout()
        ax.set_box_aspect(aspect=(5.5, 4, 3), zoom=0.86)
        plt.show()

        plt.close('all')

    def __len__(self) -> int:
        """
        The length of a AnnularSpectra is the number of annuli, so essentially a proxy for num_annuli.
        :return: The num_annuli property.
        :rtype: int
        """
        return self._num_ann

    def __getitem__(self, ind):
        return self.all_spectra[ind]



