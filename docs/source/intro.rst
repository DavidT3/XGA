Introduction to XGA
===================

XGA is a Python module designed to make it easy to analyse X-ray sources that have been observed by the
XMM-Newton Space telescope. It is based around declaring different types of source and sample objects which
correspond to real X-ray sources, finding all available data, and then insulating the user from the tedious
generation and basic analysis of X-ray data products.

XGA will generate photometric products and spectra for individual sources, or whole samples, with just a few lines
of code. It is not a pipeline itself, but pipelines for complex analysis can easily be built on top of it. XGA
provides an easy to use (and parallelised) Python interface with XMM's Science Analysis System (SAS), as well as
with XSPEC. A major goal of this module is that you shouldn't need to leave a Python environment at any point during
your analysis, as all XMM products and fit results are read into an XGA source storage structure.

This module also supports more complex analyses for specific object types; the easy generation of scaling relations,
the measurement of gas masses for galaxy clusters, and the PSF correction of images for instance.


