Installing and Configuring XGA
==============

This is a slightly more complex installation than many Python modules, but shouldn't be too difficult. If you're
having issues feel free to contact me.

Data Required For Using This Module
-----------------------------------

**This is very important** - Currently, to make use of this module, you **must** have access to cleaned XMM-Newton
event lists, as XGA is not yet capable of producing them itself.

Region Files
------------

It will be beneficial if you have region files available, as it will allow XGA to remove interloper sources. If you
wish to use existing region files, then they must be in a DS9 compatible format, **point sources** must be **red** and
**extended sources** must be **green**.

The Module
----------
We **strongly recommend** that you make use of Python virtual environments, or (even better) Conda/Mamba virtual environments when installing XGA.

XGA is available on the popular Python Package Index (PyPI), and can be installed like this:

.. code-block::

    pip install xga

You can also fetch the current working version from the git repository, and install it (this method has replaced 'python setup.py install'):

.. code-block::

    git clone https://github.com/DavidT3/XGA
    cd XGA
    python -m pip install .

Alternatively you could use the 'editable' option (this has replaced running setup.py and passing 'develop') so that any changes you pull from the remote repository are reflected without having to reinstall XGA.

.. code-block::

    git clone https://github.com/DavidT3/XGA
    cd XGA
    python -m pip install --editable .

We also provide a Conda lock file in the conda_envs directory (see `conda-lock GitHub README <https://github.com/conda/conda-lock/README.md>`_ on how to install conda-lock), which can be used to create an Anaconda environment with the required dependencies (excepting PyAbel, which has to be installed through pip at this time):

.. code-block::
    conda-lock install -n <YOUR ENVIRONMENT NAME GOES HERE>
    conda activate <YOUR ENVIRONMENT NAME GOES HERE>
    pip install pyabel==0.9

Required Dependencies
---------------------

XGA depends on two non-Python pieces of software:

    * XMM's Science Analysis System (SAS) - Version 17.0.0, but other versions should be largely compatible with the software. SAS version 14.0.0 however, does not support features that PSF correction of images depends on.
    * HEASoft's XSPEC - Version 12.10.1 - I can't guarantee later versions will work.

All required Python modules can be found in requirements.txt, and should be added to your system during the installation of XGA.

Excellent installation guides for `SAS <https://www.cosmos.esa.int/web/xmm-newton/sas-installation>`_ and
`HEASoft <https://heasarc.gsfc.nasa.gov/lheasoft/install.html>`_ already exist, so I won't go into that here.
XGA will not generate XMM products without detecting SAS, and will not fit spectra without detecting XSPEC.

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
4. Set the directory in which XGA will save the products and files it generates. I just set it to xga_output, so wherever I run a script that imports XGA it will create a folder called xga_output there. You could choose to use an absolute path and have a global XGA folder however, it would make a lot of sense.
5. You may also set an optional parameter in the [XGA_SETUP] section, 'num_cores'. If you wish to manually limit the number of cores that XGA is allowed to use, then set this to an integer value, e.g. num_cores = 10. You can also set this at runtime, by importing NUM_CORES from xga and setting that to a value.
6. The root_xmm_dir entry is the path of the parent folder containing all of your observation data.
7. Most of the other entries tell XGA how different files are named. clean_pn_evts, for instance, gives the naming convention for the cleaned PN events files that XGA generates products from.
8. Bear in mind when filling in the file fields that XGA uses the Python string formatting convention, so **anywhere you see {obs_id} will be filled formatted with the ObsID of interest when XGA is actually running**.
9. The lo_en and hi_en entries can be used to tell XGA what images and exposure maps you may already have. For instance, if you already had 0.50-2.00keV and 2.00-10.00keV images and exposure maps, you could set lo_en = ['0.50', '2.00'] and hi_en = ['2.00', '10.00'].
10. Finally, the region_file entry tells XGA where region files for each observation are stored (if they exist).

**Disclaimer: If region files are supplied, XGA also expects at least one image per instrument per observation, for WCS information.**

I have tried to make this section as general as possible, but I am biased by how my research group generates and
stores our data products. If you are an X-ray astronomer who wishes to use this module, but it seems to be incompatible
with your setup, please get in touch or raise an issue.

**Remote Data Access:** If your data lives on a remote server, and you want to use XGA on a local machine, I recommend
setting up an SFTP connection and mounting the server as an external volume. Then you can fill out the configuration
file with paths going through the mount folder - its how I use it a lot of the time.

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