#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 9/28/26, 4:19 PM. Copyright (c) The Contributors.
"""
This module implements XGA product classes that provide interfaces to high-energy astrophysics mission data
products, such as bad pixel and attitude files.
"""

from . import BaseProduct


class BadPixels(BaseProduct):
    """
    A product class for bad pixel files, mission-health data products (as opposed to science data), that describe
    which instrumental pixels are considered 'bad' for a particular observation.

    :param str path: The path to the bad pixel file, OR an S3-bucket (or S3-bucket-like) path/url to stream
            the event list data from.
    :param str obs_id: The ObsID related to the bad pixel file being declared.
    :param str instrument: The instrument related to the bad pixel file being declared.
    :param str stdout_str: The stdout from calling the terminal command.
    :param str stderr_str: The stderr from calling the terminal command.
    :param str gen_cmd: The command used to generate the bad pixel file.
    :param str telescope: The telescope that is the source of this bad pixel file. The default is None.
    :param bool force_remote: Used to force the product instantiation to treat the passed path string as a url to
            a remote dataset, and to use fsspec to read/stream the data.
    :param dict fsspec_kwargs: Optional arguments that can be passed fsspec when reading or streaming remote
        datasets - e.g. to pass credentials to access an S3 bucket. Default value is None, which sets the
        argument to {"anon": True}, making it instantly compatible with NASA archive S3 buckets.
    :param bool check_exists: Controls whether the product instantiation process checks for the file
        path's existence. Default is True, in which case a check will be performed. However, if declaring
        many products from the same directory/directory structure, it can be more performant to run listdir
        or scandir and confirm files exist externally, than one by one in each product declaration.
    """

    def __init__(
        self,
        path: str,
        obs_id: str,
        instrument: str,
        stdout_str: str,
        stderr_str: str,
        gen_cmd: str,
        extra_info: dict | None = None,
        telescope: str | None = None,
        force_remote: bool = False,
        fsspec_kwargs: dict | None = None,
        check_exists: bool = True,
    ) -> None:
        """
        The init for the BadPixels product class.

        :param str path: The path to the bad pixel file, OR an S3-bucket (or S3-bucket-like) path/url to stream
                the event list data from.
        :param str obs_id: The ObsID related to the bad pixel file being declared.
        :param str instrument: The instrument related to the bad pixel file being declared.
        :param str stdout_str: The stdout from calling the terminal command.
        :param str stderr_str: The stderr from calling the terminal command.
        :param str gen_cmd: The command used to generate the bad pixel file.
        :param str telescope: The telescope that is the source of this bad pixel file. The default is None.
        :param bool force_remote: Used to force the product instantiation to treat the passed path string as a url to
                a remote dataset, and to use fsspec to read/stream the data.
        :param dict fsspec_kwargs: Optional arguments that can be passed fsspec when reading or streaming remote
            datasets - e.g. to pass credentials to access an S3 bucket. Default value is None, which sets the
            argument to {"anon": True}, making it instantly compatible with NASA archive S3 buckets.
        :param bool check_exists: Controls whether the product instantiation process checks for the file
            path's existence. Default is True, in which case a check will be performed. However, if declaring
            many products from the same directory/directory structure, it can be more performant to run listdir
            or scandir and confirm files exist externally, than one by one in each product declaration.
        """
        # Call the BaseProduct init, which handles setting up main attributes
        super().__init__(
            path,
            obs_id,
            instrument,
            stdout_str,
            stderr_str,
            gen_cmd,
            extra_info,
            telescope,
            force_remote,
            fsspec_kwargs,
            check_exists,
        )
        self._prod_type = "badpix"
