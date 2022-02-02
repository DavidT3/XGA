Introduction to XGA
===================

X-ray: Generate and Analyse (XGA) is a Python module designed to make it easy to analyse X-ray sources that have been
observed by the XMM-Newton Space telescope. It is based around declaring different types of source and sample
objects which correspond to real X-ray sources, finding all available data, and then insulating the user from the
tedious generation and basic analysis of X-ray data products (though with the option to get stuck into the data
directly if required).

XGA will generate photometric products and spectra for individual sources, or whole samples, with just a few lines
of code. It is not a pipeline in itself, as it can be used for interactive analyses in Jupyter Notebooks, however it is
quite possible to build pipeline's using XGA's features and methods. XGA provides an easy to use Python interface with
XMM's Science Analysis System (SAS) and XSPEC, where all generation and fitting procedures have been parallelised as
much as is possible. A major goal of this module is that you shouldn't need to leave a Python environment at any point
during your analysis, as all XMM products and fit results are read into an XGA source storage structure.

This module also supports more complex analyses for specific object types; the easy generation of scaling relations,
the measurement of gas masses for galaxy clusters, and the PSF correction of images for instance. It is also
possible, for extended sources (such as galaxy clusters), to generate and fit sets of annular spectra. This allows you
to investigate how properties change radially with distance from the centre, and enables the measurement of hydrostatic
masses of clusters.

While XGA is a piece of open source software, I would appreciate it if any work that makes use of it would cite the
paper accompanying this package, which can be found in the :ref:`_xga_pub` section.

If wish to contribute to XGA, have feature suggestions, or any comments at all, then please go to the
"Getting Support" section and submit an issue on GitHub/send me an email, I'll be happy to hear from you!