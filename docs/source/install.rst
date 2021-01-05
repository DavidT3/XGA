Installing and Configuring XGA
==============

This is a slightly more complex installation than many Python modules, but shouldn't be too difficult. If you're
having issues feel free to contact me.

.. toctree::
   :maxdepth: 2

    The Module
    =========

    XGA has been uploaded to PyPi, so you can simply run:
    .. code-block:: console
        pip install xga

    Alternatively, to get the current working version from the git repository run:
    .. code-block:: console
        git clone https://github.com/DavidT3/XGA
        cd XGA
        python setup.py install

    Required Dependencies
    =====================

    XGA depends on two non-Python pieces of software:
    * XMM's Science Analysis System (SAS) - Version 17.0.0, but other versions should be fine.
    * HEASoft's XSPEC - Version 12.10.1, **I can't guarantee later versions will work.**

    All required Python modules can be found in requirements.txt, and should be added to your system during the
    installation of XGA.

    Excellent installation guides for `SAS <https://www.cosmos.esa.int/web/xmm-newton/sas-installation>`__ and
    `HEASoft <https://heasarc.gsfc.nasa.gov/lheasoft/install.html>`__ already exist, so I won't go into that here.
    XGA will not run without detecting these pieces of software installed on your system.

    Optional Dependencies
    =====================

    XGA can also make use of external software for some limited tasks, but they are not required to use
    the module as a whole:
    * The R interpreter.
    * Rpy2 - A Python module that provides an interface with the R language in Python.
    * LIRA - An R fitting package.

    The R interpreter, Rpy2, and LIRA are all necessary only if you wish to use the LIRA scaling relation fitting function.


    Configuring XGA
    ===============

    Before XGA can be used you must fill out a configuration file (a completed example can be found
    `here <https://github.com/DavidT3/XGA/blob/master/docs/example_config/xga.cfg>`__).

    Follow these steps to fill out the configuration file:
    1. Import XGA to generate the initial, incomplete, configuration file.
    2. Navigate to ~/.config/xga and open xga.cfg in a text editor. The .config directory is usually hidden, so it is
    probably easier to navigate via the terminal.
    3. Take note of the entries that currently have /this/is/required at the beginning, without these entries the
    module will not function.
    4. Set the directory where you wish XGA to save the products and files it generates. I just set it to xga_output,
    so wherever I run a script that imports XGA it will create a folder called xga_output there. You could choose to use
    an absolute path and have a global XGA folder however, it would make a lot of sense.
    5. The root_xmm_dir entry is the path of the parent folder containing all of your observation data.
    6. Most of the other entries tell XGA how different files are named. clean_pn_evts, for instance, gives the naming
    convention for the cleaned PN events files that XGA generates products from.
    7. Bear in mind when filling in the file fields that XGA uses the Python string formatting convention, so anywhere
    you see {obs_id} will be filled formatted with the ObsID of interest when XGA is actually running.
    8. The lo_en and hi_en entries can be used to tell XGA what images and exposure maps you may already have. For instance,
     if you already had 0.50-2.00keV and 2.00-10.00keV images and exposure maps, you could set lo_en = ['0.50', '2.00'] and
     hi_en = ['2.00', '10.00'].
    9. Finally, the region_file entry tells XGA where region files for each observation are stored (if they exist).
    **Disclaimer: If region files are supplied, XGA also expects at least one image per instrument per observation.**

    I have tried to make this part as general as possible, but I am biased by how XCS generates and stores their data
    products. If you are an X-ray astronomer who wishes to use this module, but it seems to be incompatible with your setup,
     please get in touch or raise an issue.

    **Remote Data Access:** If your data lives on a remote server, and you want to use XGA on a local machine, I recommend
    setting up an SFTP connection and mounting the server as an external volume. Then you can fill out the configuration
    file with paths going through the mount folder - its how I use it a lot of the time.

    XGA's First Run After Configuration
    ===================================

    The first time you import any part of XGA, it will create an 'observation census', where it will search through
    all the observations it can find (based on your entries in the configuration file), check that there are events
    lists present, and record the pointing RA and DEC. *This can take a while*, but will only take that long on the first
    run. The module will check the census against your observation directory and see if it needs to be updated on
    every run.
