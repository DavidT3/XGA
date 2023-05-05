<p align="center">
    <img src="https://raw.githubusercontent.com/DavidT3/XGA/master/xga/files/long_xga_logo.png" width="500">
</p>

[![Documentation Status](https://readthedocs.org/projects/xga/badge/?version=latest)](https://xga.readthedocs.io/en/latest/?badge=latest)
[![Coverage Percentage](https://raw.githubusercontent.com/DavidT3/XGA/master/tests/coverage_badge.svg)](https://raw.githubusercontent.com/DavidT3/XGA/master/tests/coverage_badge.svg)

# What is X-ray: Generate and Analyse (XGA)?

XGA is a Python module designed to make it easy to analyse X-ray sources that have been observed by the XMM-Newton Space telescope. It is based around declaring different types of source and sample objects which correspond to real X-ray sources, finding all available data, and then insulating the user from the tedious generation and basic analysis of X-ray data products.

XGA will generate photometric products and spectra for individual sources, or whole samples, with just a few lines of code. It is not a pipeline itself, but pipelines for complex analysis can easily be built on top of it. XGA provides an easy to use (and parallelised) Python interface with XMM's Science Analysis System (SAS), as well as with XSPEC. A major goal of this module is that you shouldn't need to leave a Python environment at any point during your analysis, as all XMM products and fit results are read into an XGA source storage structure.

This module also supports more complex analyses for specific object types; the easy generation of scaling relations, the measurement of gas masses for galaxy clusters, and the PSF correction of images for instance.

# Installing XGA
This is a slightly more complex installation than many Python modules, but shouldn't be too difficult. If you're
having issues feel free to contact me.

## Data Required to use XGA
### Cleaned Event Lists
**This is very important** - Currently, to make use of this module, you **must** have access to cleaned XMM-Newton
event lists, as XGA is not yet capable of producing them itself.

### Region Files
It will be beneficial if you have region files available, as it will allow XGA to remove interloper sources. If you
wish to use existing region files, then they must be in a DS9 compatible format, **point sources** must be **red** and
**extended sources** must be **green**.

## The Module
XGA has been uploaded to PyPi, so you can simply run:
```shell script
pip install xga
```

Alternatively, to get the current working version from the git repository run:
```shell script
git clone https://github.com/DavidT3/XGA
cd XGA
python setup.py install
```

## Required Dependencies
XGA depends on two non-Python pieces of software:
* XMM's Science Analysis System (SAS) - Version 17.0.0, but other versions should be largely compatible with the
    software. SAS version 14.0.0 however, does not support features that PSF correction of images depends on.
* HEASoft's XSPEC - Version 12.10.1, but other versions should be largely compatible even if I have not tested them.

All required Python modules can be found in requirements.txt, and should be added to your system during the 
installation of XGA.

Excellent installation guides for [SAS](https://www.cosmos.esa.int/web/xmm-newton/sas-installation) and 
[HEASoft](https://heasarc.gsfc.nasa.gov/lheasoft/install.html) already exist, so I won't go into that in this readme. 
XGA will not run without detecting these pieces of software installed on your system.

## Optional Dependencies
XGA can also make use of external software for some limited tasks, but they are not required to use 
the module as a whole:
* The R interpreter.
* Rpy2 - A Python module that provides an interface with the R language in Python.  
* LIRA - An R fitting package.

The R interpreter, Rpy2, and LIRA are all necessary only if you wish to use the LIRA scaling relation fitting function.


## Configuring XGA - **THIS SECTION IS VERY IMPORTANT**
Before XGA can be used you must fill out a configuration file (a completed example can be found 
[here](https://github.com/DavidT3/XGA/blob/master/docs/example_config/xga.cfg)). 

Follow these steps to fill out the configuration file:
1. Import XGA to generate the initial, incomplete, configuration file.
2. Navigate to ~/.config/xga and open xga.cfg in a text editor. The .config directory is usually hidden, so it is 
probably easier to navigate via the terminal.
3. Take note of the entries that currently have /this/is/required at the beginning, without these entries the 
module will not function.
4. Set the directory where you wish XGA to save the products and files it generates. I just set it to xga_output,
so wherever I run a script that imports XGA it will create a folder called xga_output there. You could choose to use
an absolute path and have a global XGA folder however, it would make a lot of sense.
5. You may also set an optional parameter in the [XGA_SETUP] section, 'num_cores'. If you wish to manually limit the 
   number of cores that XGA is allowed to use, then set this to an integer value, e.g. num_cores = 10. You can also
   set this at runtime, by importing NUM_CORES from xga and setting that to a value.
6. The root_xmm_dir entry is the path of the parent folder containing all of your observation data.
7. Most of the other entries tell XGA how different files are named. clean_pn_evts, for instance, gives the naming
convention for the cleaned PN events files that XGA generates products from. 
8. Bear in mind when filling in the file fields that XGA uses the Python string formatting convention, so anywhere
you see {obs_id} will be filled formatted with the ObsID of interest when XGA is actually running.
9. The lo_en and hi_en entries can be used to tell XGA what images and exposure maps you may already have. For instance,
 if you already had 0.50-2.00keV and 2.00-10.00keV images and exposure maps, you could set lo_en = ['0.50', '2.00'] and 
 hi_en = ['2.00', '10.00'].
10. Finally, the region_file entry tells XGA where region files for each observation are stored (if they exist). 
**Disclaimer: If region files are supplied, XGA also expects at least one image per instrument per observation.**
    
I have tried to make this part as general as possible, but I am biased by how XCS generates and stores their data 
products. If you are an X-ray astronomer who wishes to use this module, but it seems to be incompatible with your setup,
 please get in touch or raise an issue.

**Remote Data Access:** If your data lives on a remote server, and you want to use XGA on a local machine, I recommend 
setting up an SFTP connection and mounting the server as an external volume. Then you can fill out the configuration 
file with paths going through the mount folder - its how I use it a lot of the time.

## XGA's First Run After Configuration
The first time you import any part of XGA, it will create an 'observation census', where it will search through
all the observations it can find (based on your entries in the configuration file), check that there are events
lists present, and record the pointing RA and DEC. *This can take a while*, but will only take that long on the first
run. The module will check the census against your observation directory and see if it needs to be updated on 
every run.


# How to use the module
Please refer to the tutorials in the documentation, which can be found [here](https://xga.readthedocs.io/)


# Problems and Questions
If you encounter a bug, or would like to make a feature request, please use the GitHub
[issues](https://github.com/DavidT3/XGA/issues) page, it really helps to keep track of everything.

However, if you have further questions, or just want to make doubly sure I notice the issue, feel free to send
me an email at turne540@msu.edu





