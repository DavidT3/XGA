Installing and Configuring XGA
==============

This is a slightly more complex installation than many Python modules, but shouldn't be too difficult. If you're
having issues feel free to get in contact (see :doc:`Getting Help` section).

Data Required For Using This Module
-----------------------------------

**This is very important** - Currently, to make use of this module, you **must** have access to **either** cleaned XMM-Newton or eROSITA
event lists. For aquiring and cleaning X-ray data, the Python module `DAXA <https://github.com/DavidT3/DAXA>`_ is recommended. 

Region Files
------------

It will be beneficial if you have region files available, as it will allow XGA to remove interloper sources. If you
wish to use existing region files, then they must be in a DS9 compatible format, **point sources** must be **red** and
**extended sources** must be **green**.

The Module
----------

XGA is available on PyPi, so you can simply run:

.. code-block::

    pip install xga

Alternatively you can fetch the current working version from the git repository, which (as XGA is still in a fairly
early stage of development) may have more up-to-date features than the PyPi release:

.. code-block::

    git clone https://github.com/DavidT3/XGA
    cd XGA
    python setup.py install

Required Dependencies
---------------------

XGA depends on some non-Python pieces of software, dependent on the telescope from which your data originates:

For all telescopes:
    * HEASoft's XSPEC - Version 12.10.1 - It isn't guaranteed later versions will work.

For XMM data:
    * XMM's Science Analysis System (SAS) - Version 17.0.0, but other versions should be largely compatible with the software. SAS version 14.0.0 however, does not support features that PSF correction of images depends on.

For eROSITA data:
    * eROSITA Science Analysis Software System (eSASS) - EDR version. It must be installed on the system, we do not support interaction through a Docker container. 

All required Python modules can be found in requirements.txt, and should be added to your system during the installation of XGA.

Excellent installation guides for `SAS <https://www.cosmos.esa.int/web/xmm-newton/sas-installation>`_, 
`HEASoft <https://heasarc.gsfc.nasa.gov/lheasoft/install.html>`_, and `eSASS <https://erosita.mpe.mpg.de/edr/DataAnalysis/esassinstall.html>`_ already exist, please seek these for detailed instalation instructions.
XGA will not run without detecting at least one telescope-specific software installed on your system.

Optional Dependencies
---------------------

XGA can also make use of external software for some limited tasks, but they are not required to use
the module as a whole:

    * The R interpreter.
    * Rpy2 - A Python module that provides an interface with the R language in Python.
    * LIRA - An R fitting package.

    The R interpreter, Rpy2, and LIRA are all necessary only if you wish to use the LIRA scaling relation fitting function.

Configuring XGA
---------------

Before XGA can be used you must fill out a configuration file (a completed example can be found
`here <https://github.com/DavidT3/XGA/blob/master/docs/example_config/xga.cfg>`_).

Follow these steps to fill out the configuration file:

1. Import XGA to generate the initial, incomplete, configuration file.
2. Navigate to ~/.config/xga and open xga.cfg in a text editor. The .config directory is usually hidden, so it is probably easier to navigate via the terminal.
3. Take note of the entries that currently have /this/is/required at the beginning, without these entries the module will not function.
4. Set the directory in which XGA will save the products and files it generates. It is advised to just set it to xga_output, so wherever you run a script that imports XGA it will create a folder called xga_output there. You could choose to use an absolute path and have a global XGA folder however, it wouldn't make a lot of sense.
5. You may also set an optional parameter in the [XGA_SETUP] section, 'num_cores'. If you wish to manually limit the number of cores that XGA is allowed to use, then set this to an integer value, e.g. num_cores = 10. You can also set this at runtime, by importing NUM_CORES from xga and setting that to a value.
6. The root_<telescope>_dir entry is the path of the parent folder containing all of your observation data for <telescope>. It is not necessary to have data for all telescopes that XGA supports. 
7. Most of the other entries tell XGA how different files are named. clean_pn_evts, for instance, gives the naming convention for the cleaned PN events files that XGA generates products from.
8. Bear in mind when filling in the file fields that XGA uses the Python string formatting convention, so **anywhere you see {obs_id} will be filled formatted with the ObsID of interest when XGA is actually running**.
9. The lo_en and hi_en entries can be used to tell XGA what images and exposure maps you may already have. For instance, if you already had 0.50-2.00keV and 2.00-10.00keV images and exposure maps, you could set lo_en = ['0.50', '2.00'] and hi_en = ['2.00', '10.00'].
10. Finally, the region_file entry tells XGA where region files for each observation are stored (if they exist).

**Disclaimer: If region files are supplied for XMM data, XGA also expects at least one image per instrument per observation, for WCS information.**

This section aims to be as general as possible, but is biased by how our research group generates and
stores our data products. If you are an X-ray astronomer who wishes to use this module, but it seems to be incompatible
with your setup, please get in touch or raise an issue.

**Remote Data Access:** If your data lives on a remote server, and you want to use XGA on a local machine, it is recommended
to set up an SFTP connection and mounting the server as an external volume. Then you can fill out the configuration
file with paths going through the mount folder.
To mount a server, one can follow the steps

.. code-block::
    brew install sshfs

.. code-block::
    sshfs username@hostname:/remotepath /localpath -ovolname=sftp

Here the '-ovolname' argument controls the name of the directory on your local machine. 

To unmount use the command

.. code-block::
    umount -f /localpath 


XGA's First Run After Configuration
-----------------------------------

The first time you import any part of XGA, it will create an 'observation census', where it will search through
all the observations it can find (based on your entries in the configuration file), check that there are events
lists present, and record the pointing RA and DEC. *This can take a while*, but will only take that long on the first
run. The module will check the census against your observation directory and see if it needs to be updated on
every run.

Blacklisting ObsIDs
-------------------

If you don't wish your analyses to include certain ObsIDs, then you can 'blacklist' them and remove them from all
consideration, you simply need to add the ObsID to 'blacklist.csv', which is located in the same directory as the
configuration file. If you need to know where this configuration file is located, import CONFIG_FILE from xga.utils.

It is possible that you might want to do this so that ObsIDs with significant problems (flaring, for instance), don't
contribute to and spoil your current analysis.