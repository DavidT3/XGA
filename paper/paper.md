---
title: '\texttt{XGA}: A module for the large-scale scientific exploitation of X-ray data'
tags:
  - Python
  - Astronomy
  - Astrophysics
  - X-ray astronomy
  - Galaxy clusters
  - XMM
authors:
  - name: David J. Turner
    orcid: 0000-0001-9658-1396
    affiliation: 1
  - name: Paul A. Giles
    orcid: 0000-0003-4937-8453
    affiliation: 1
  - name: Kathy Romer
    orcid: 0000-0002-9328-879X
    affiliation: 1
affiliations:
  - name: Department of Physics and Astronomy, University of Sussex, Brighton, BN1 9QH, UK
    index: 1
date: 02 November 2021
bibliography: paper.bib
---
# Summary
X-ray telescopes allow for the investigation of some of the most extreme objects and processes in the 
Universe; this includes galaxy clusters, active galactic nuclei (where a supermassive black hole at the centre of the 
galaxy is actively accreting matter), and supernovae remnants. This makes the analysis of X-ray observations very 
useful for a wide variety of fields in astrophysics and cosmology. Galaxy clusters, for instance, can act as 
laboratories for the exploration of many astrophysical processes, as well as providing insight into how the Universe 
has evolved during its lifetime, as they are excellent tracers of the formation of large scale structure.

We have developed a new Python module (X-ray: Generate and Analyse, hereafter referred to as \texttt{XGA}) to provide
interactive and automated analyses of X-ray emitting sources, in order to . `XGA` revolves 
around `source` objects, which are representative of X-ray sources in real life. These `source` classes all have 
different properties and methods, which either relate to relevant properties of or perform measurements which are only 
relevant to that type of astronomical source, with some properties/methods being common to all sources.

XGA also contains `product` classes, which provide interfaces to X-ray data products, with built in methods for 
analysis, manipulation, and visualisation. The `RateMap` (a count rate map of a particular observation) class for 
instance includes view methods (demonstrated in \autoref{fig:ratemap_mask}), 
methods for coordinate conversion, and for measuring the peak of the X-ray emission. 
We also provide classes for interacting with spectra (both global and annular, where \autoref{fig:a907_spec} and 
\autoref{fig:ann_spec} demonstrate the view methods), PSFs, and a base class for XGA profile
objects, which allow for the storage, fitting, and viewing of radial profiles generated through XGA processes.

![The output of the view method of a RateMap instance where a mask to remove interloper sources has been applied, with 
an added crosshair to indicate coordinates of 
interest. \label{fig:ratemap_mask}](figures/ratemap_crosshair_intmask.png){width=80%}

This approach means that the user can either remain removed from contact with the X-ray data if they choose, or
interact with it directly for lower level analyses that they are building around the `XGA` platform.
With the advent of new X-ray observatories such as eROSITA (@erosita), XRISM (@xrism), ATHENA (@athena), and 
Lynx (@lynx), it would seem to be a good time for a new, open-source, software package that is open for anyone to 
use and scrutinise.

# Statement of need
The initial goal for this new module was the measurement of hydrostatic masses of galaxy clusters for the XMM 
Cluster Survey (XCS, @xcsfoundation), but has become an attempt to provide the X-ray astronomy community with an 
open source, general purpose tool to build research projects upon. One of the chief advantages of this module is that 
it simplifies the process of generating the data products which are required for most work involving X-ray 
analysis; once the user has supplied cleaned event lists (and optionally region files), and a source object has decided 
which observations should be associated with it, an analysis region can be specified and spectra (along with any 
auxiliary files that are required) can be created. We can use XGA to investigate both average properties and, in the 
case of extended sources, how these properties vary spatially. Similar procedures for image based analysis are also 
available, where images (and merged images from all available data for a given source) can be easily generated en 
masse, then combined with masks automatically generated from supplied region files to perform photometric analyses.

Software to generate X-ray data products is supplied by the telescope teams, but in the case of XMM-Newton it can 
only be used on the command line, and most commands require significant setup and configuration. XGA wraps the most 
useful commands and provides the user with an easy way to generate these products for large samples of 
objects (which will scale across multiple cores), while taking into account complex factors (such as removing interloper sources) 
that vary from source to source. To extract useful information from the generated spectra, we implemented a method 
for fitting models, creating an interface with XSPEC (@xspec), the popular X-ray spectral fitting language. This interface again
provides simplified interaction with the underlying software that can be run simultaneously when multiple sources are
being analysed at the same time. 

![The output of the view method of a `Spectrum` instance associated with a GalaxyCluster source, which has been fitted 
with a plasma emission model. \label{fig:a907_spec}](figures/A907_spec.png){width=85%}

![The output of the view method of an `AnnularSpectrum` instance associated with a GalaxyCluster source. Here the 
plasma emission models which have been fitted to each annulus are 
displayed.\label{fig:ann_spec}](figures/ann_spec.png){width=90%}

Many more features are built into XGA, enabled by the source based structure, as well as the product generation 
and XSPEC interface. These features are largely motivated by a desire to measure hydrostatic galaxy cluster masses; this 
includes the measurement of 3D gas density profiles, 3D temperature profiles, gas mass, and total mass profiles. New 
methods for the measurement of central cluster coordinates and PSF correction of XMM images were also created to enable 
this, as well as Python classes for various data products (with many useful built in methods). This includes a radial 
profile class, with built in viewing methods, and a fitting method based around the `emcee` ensemble MCMC 
sampler (@emcee). The profile fitting capability also motivated the creation of model class, with methods for 
storing and interacting with fitted models; including integration and differentiation methods, inverse abel 
transforms, and predictions from the model.

# Existing software packages
To the knowledge of the authors, no software package exists that provides features completely equivalent to 
XGA, particularly in the open source domain. That is not to say that there are no software tools similar to 
the module that we have constructed; several research groups including XCS (@xcsmethod), XXL (@xxllt), 
LoCuSS (@locusshydro), and the cluster group at UC Santa Cruz (@matcha) have developed pipelines to measure 
the luminosity and temperature of X-ray emitting galaxy clusters, though these have not been made public. It is 
also important to note that these pipelines are normally designed to measure a particular aspect of a 
particular type of X-ray source (galaxy clusters in these cases), and as such they lack the generality and flexibility 
of `XGA`. Our new software is also designed to be used interactively, as well as a basis for building pipelines such
as these.

Some specific analyses built into `XGA` have comparable open source software packages available; for instance 
`pyprofit` (@erositagasmass) is a recently released Python module that was designed 
for the measurement of gas density from X-ray surface brightness profiles. We do not believe that any existing X-ray 
analysis module has an equivalent to the source and sample based structure which XGA is built around, or to the 
product classes that have been written to interact with X-ray data products.

The `XSPEC` (@xspec) interface we have developed for XGA is far less comprehensive than the full Python wrapping 
implemented in the `PyXspec` module, but scales with multiple cores for the analysis of multiple sources 
simultaneously much more easily. 

# Ongoing research projects
As \texttt{XGA} is a new piece of work, written over the last year, there are currently no published works that make use of 
it. There are, however, several projects that use XGA extensively nearing publication. The first of these is a hydrostatic 
and gas mass analysis of the SDSS redMaPPer (@redmappersdss)-XCS optically selected galaxy cluster sample (@sdssxcs) and 
well as the ACTDR5 (@actdr5)-XCS sample of Sunyaev-Zel'dovich (SZ) selected galaxy clusters. This work also compares commonly measured X-ray properties of clusters 
(the X-ray luminosity L$_{\rm{x}}$, and the temperature T$_{\rm{x}}$) both to results from the existing XCS pipeline and from literature, confirming 
that `XGA` measurements are consistent with previous work. Similar work is being done on a Dark Energy Survey (DES)Y3-XCS optically 
selected sample of clusters, though this will also include analysis from other XCS tools, and will not be focussed only
on mass measurements. `XGA`'s ability to stack and combine X-ray surface brightness profiles is currently being 
used, in combination with weak lensing information from DES, to look for signs of modified gravity in galaxy 
clusters. Finally an exploration of the X-ray properties of a new sample of Pea galaxies is being performed using
the point source class, the `XSPEC` interface, and the upper limit luminosity functionality.

# Acknowledgements
DT, KR, and PG acknowledge support from the UK Science and Technology Facilities Council via grants ST/P006760/1 (DT),  ST/P000525/1 and ST/T000473/1 (PG, KR).

David J. Turner would like to thank Aswin P. Vijayan, Lucas Porth, Tim Lingard, and Reese Wilkinson for useful 
discussions during the course of writing this module.

# References
