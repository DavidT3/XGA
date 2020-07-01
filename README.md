# What is XMM: Generate and Analyse (XGA)?

XGA is a Python package to explore and analyse X-ray sources detected in XMM-Newton observations. When a source is 
declared, XGA finds every observation it appears in, and all analysis is performed on all available data. 
It provides an easy to use Python interface with XMM's Science Analysis System (SAS), enabling independent products to be generated in parallel on 
local computers and HPCs. XGA also makes it easy to generate products for samples of objects.

XGA also provides a similar interface with the popular X-ray spectrum fitting tool XSPEC, and makes it extremely
simple to create an XSPEC script for whatever source you are analysing, run that script, and then read the results 
back into Python. The XSPEC interface will also run multiple fits in parallel.

# Installing XGA
This is a slightly more complex installation than many Python modules, but shouldn't be too difficult. If you're
having issues feel free to contact me.

## The Module
As XGA is still in a very early stage of development it hasn't been submitted to PyPi yet, as such you should clone 
this GitHub repository, and run the setup.py file.

```shell script
git clone https://github.com/DavidT3/XGA
python setup.py install
```

## External Dependencies
XGA depends on two non-Python pieces of software:
* XMM's Science Analysis System (SAS) - Version 17.0.0, but other versions should be fine.
* HEASoft's XSPEC - Version 12.10.1, **I can't guarantee later versions will work.**

Excellent installation guides for [SAS](https://www.cosmos.esa.int/web/xmm-newton/sas-installation) and 
[HEASoft](https://heasarc.gsfc.nasa.gov/lheasoft/install.html) already exist, so I won't go into that in this readme. 
XGA will not run without detecting these pieces of software installed on your system.


## Configuring XGA - **THIS SECTION IS VERY IMPORTANT**
Before XGA can be used you must fill out a configuration file (a completed example can be found 
[here](https://github.com/DavidT3/XGA/blob/docs/master/docs/example_config/xga.cfg)). 

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

## XGA's First Run After Configuration
The first time you import any part of XGA, it will create an 'observation census', where it will search through
all the observations it can find (based on your entries in the configuration file), check that there are events
lists present, and record the pointing RA and DEC. *This can take a while*, but will only take that long on the first
run. The module will check the census against your observation directory and see if it needs to be updated on 
every run.


# How to use the module
As it is in such an early stage of development, XGA doesn't have proper documentation yet. All the functions should
have at least vaguely helpful docstrings, and there is an example Jupyter Notebook 
[here](https://github.com/DavidT3/XGA/blob/docs/master/docs/example_notebooks/general_demo.ipynb) that 
demonstrates the basic functionality.


# Problems and Questions
If you encounter a bug, or would like to make a feature request, please use the GitHub
[issues](https://github.com/DavidT3/XGA/issues) page, it really helps to keep track of everything.

However, if you have further questions, or just want to make doubly sure I notice the issue, feel free to send
me an email at david.turner@sussex.ac.uk





