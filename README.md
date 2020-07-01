# What is XMM: Generate and Analyse (XGA)?

XGA is a Python package to explore and analyse X-ray sources detected in XMM-Newton observations. When a source is 
declared, XGA finds every observation it appears in, and all analysis is performed on all available data. 
It provides an easy to use Python interface with XMM's Science Analysis System (SAS), enabling independent products to be generated in parallel on 
local computers and HPCs. XGA also makes it easy to generate products for samples of objects.

XGA also provides a similar interface with the popular X-ray spectrum fitting tool XSPEC, and makes it extremely
simple to create an XSPEC script for whatever source you are analysing, run that script, and then read the results 
back into Python. The XSPEC interface will also run multiple fits in parallel.

# Installing XGA

## The Module
As XGA is still in a very early stage of development it hasn't been submitted to PyPi yet, as such you should clone 
this GitHub repository, and run the setup.py file.

```shell script
git clone https://github.com/DavidT3/XGA
python setup.py install
```

## External Dependencies
XGA depends on two non-Python pieces of software:
* XMM's Science Analysis System (SAS) - XGA has been developed using SAS 17.0.0, but other versions should be fine
* HEASOFT's XSPEC - XGA has been developed using XSPEC 12.10.1, **I can't guarantee later versions will work**

Excellent installation guides for [SAS](https://www.cosmos.esa.int/web/xmm-newton/sas-installation) and 
[HEASOFT](https://heasarc.gsfc.nasa.gov/lheasoft/install.html) already exist, so I won't go into that in this readme. 
XGA will not run without detecting these pieces of software installed on your system.


## Configuring XGA - **THIS SECTION IS VERY IMPORTANT**


## XGA's First Run
The first time you import any part of XGA, it will create an 'observation census', where it will search through
all the observations it can find (based on your entries in the configuration file), check that there are events
lists present, and record the pointing RA and DEC. *This can take a while*, but will only take that long on the first
run. The module will check the census against your observation directory and see if it needs to be updated on 
every run.


# How to use the module
As it is in such an early stage of development, XGA doesn't have proper documentation yet. All the functions should
have at least vaguely helpful docstrings, and there is an example Jupyter Notebook in the docs/example_notebooks 
folder that demonstrates the basic functionality.


# Problems and Questions
If you encounter a bug, or would like to make a feature request, please use the GitHub
[issues](https://github.com/DavidT3/XGA/issues) page, it really helps to keep track of everything.

However, if you have further questions, or just want to make doubly sure I notice the issue, feel free to send
me an email at david.turner@sussex.ac.uk





