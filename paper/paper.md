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
  - name: Violetta Korbina
    orcid: 
    affiliation: 1
affiliations:
  - name: Department of Physics and Astronomy, University of Sussex, Brighton, BN1 9QH, UK
    index: 1
date: 31 January 2022
bibliography: paper.bib
---

# Summary
The _XMM_ Cluster Survey [XCS, @xcsfoundation] have developed a new Python module (X-ray: Generate and Analyse, hereafter 
referred to as \texttt{XGA}) to provide interactive and automated analyses of X-ray emitting sources observed by the 
_XMM_-Newton space telescope. \texttt{XGA} is centered around the concept of making all available data easily accessible 
and analysable; the user provides information on the source they want to investigate and \texttt{XGA} will then find 
all relevant observations and generate all the required date products. This approach means that the user can quickly 
and easily complete common analyses without manually searching through large amounts of archival data for relevant 
observations, thus being left free to focus on extracting the maximum scientific gain. With the advent of new X-ray 
observatories such as _eROSITA_ [@erosita], _XRISM_ [@xrism], _ATHENA_ [@athena], and _Lynx_ [@lynx], it is the perfect time 
for a new, open-source, software package that is open for anyone to use and scrutinise.

# Statement of need
X-ray telescopes allow for the investigation of some of the most extreme objects and processes in the 
Universe; this includes galaxy clusters, active galactic nuclei (AGN), and X-ray emitting stars. This makes the analysis 
of X-ray observations useful for a variety of fields in astrophysics and cosmology. Galaxy clusters, for instance, are 
useful as astrophysical laboratories, and provide insight into how the Universe has evolved during its lifetime.

Current generation X-ray telescopes have large archives of publicly available observations; _XMM_-Newton has been 
observing for over two decades, for instance. This allows for serendipitous analysis of large amounts of archival 
data, but also introduces issues with respect to accessing and analysing all the relevant data for a particular source. 
\texttt{XGA} solves this problem by automatically identifying the relevant _XMM_ observations then generating whatever
data products the user requires; from images to sets of annular spectra. Once the user has supplied cleaned event 
lists (and optionally region files) an analysis region can be specified and spectra (along with any 
auxiliary files that are required) can be created.

Software to generate X-ray data products is supplied by the telescope teams, and most commands require significant 
setup and configuration. The complexity only increases when analysing multiple observations of a single source, as is
often the case due to the large archive of data available. \texttt{XGA} provides the user with an easy way to generate 
_XMM_ data products for large samples of objects (which will scale across multiple cores), while taking into account 
complex factors (such as removing interloper sources) that vary from source to source.

# Features

We can use \texttt{XGA} to investigate both average properties and, in the 
case of extended sources, how these properties vary spatially. Similar procedures for image based analysis are also 
available, where images (and merged images from all available data for a given source) can be easily generated en 
masse, then combined with masks automatically generated from supplied region files to perform photometric analyses.

is centered around `source` and `sample` classes,  Different `source` classes, which represent different types of X-ray emitting astrophysical objects, all 
have different properties and methods. These either relate to relevant properties of or perform measurements which are only 
relevant to that type of astronomical source, with some properties/methods being common to all sources.

\texttt{XGA} also contains `product` classes, which provide interfaces to X-ray data products, with built-in methods for 
analysis, manipulation, and visualisation. The `RateMap` (a count rate map of a particular observation) class for 
instance includes view methods (a demonstration of masked views is shown in Figure \autoref{fig:ratemap_mask}), methods 
for coordinate conversion, and for measuring the peak of the X-ray emission. We also provide classes for interacting 
with and analysing spectra, both g

, PSFs, and a base class for \texttt{XGA} profile
objects, which allow for the storage, fitting, and viewing of radial profiles generated through \texttt{XGA} processes.


To extract useful information from the generated spectra, we implemented a method 
for fitting models, creating an interface with XSPEC [@xspec], the popular X-ray spectral fitting language. This interface again
provides simplified interaction with the underlying software that can be run simultaneously when multiple sources are
being analysed at the same time.

Many more features are built into \texttt{XGA}, enabled by the source based structure, as well as the product generation 
and XSPEC interface. This includes a set of profile classes, with built-in viewing methods, and a fitting method based 
around the `emcee` ensemble MCMC sampler [@emcee]. Profiles also support storing and interacting with fitted 
models; including integration and differentiation methods, inverse abel transforms, and predictions from the model. 
An example of the utility of these profiles is the galaxy cluster hydrostatic mass measurement feature; this 
requires the measurement of 3D gas density profiles, 3D temperature profiles, gas mass, and total mass profiles. 

![A flowchart giving a brief overview of \texttt{XGA} \label{fig:flowchart}](figures/xga_flowchart.png)

# Existing software packages
To the knowledge of the authors, no software package exists that provides features completely equivalent to 
\texttt{XGA}, particularly in the open source domain. That is not to say that there are no software tools similar to 
the module that we have constructed; several research groups including XCS [@xcsmethod], XXL [@xxllt], 
LoCuSS [@locusshydro], and the cluster group at UC Santa Cruz [@matcha] have developed pipelines to measure 
the luminosity and temperature of X-ray emitting galaxy clusters, though these have not been made public. It is 
also important to note that these pipelines are normally designed to measure a particular aspect of a 
particular type of X-ray source (galaxy clusters in these cases), and as such they lack the generality and flexibility 
of \texttt{XGA}. Our new software is also designed to be used interactively, as well as a basis for building pipelines such
as these.

Some specific analyses built into \texttt{XGA} have comparable open source software packages available; for instance 
`pyproffit` [@erositagasmass] is a recently released Python module that was designed 
for the measurement of gas density from X-ray surface brightness profiles. We do not believe that any existing X-ray 
analysis module has an equivalent to the source and sample based structure which \texttt{XGA} is built around, or to the 
product classes that have been written to interact with X-ray data products.

The `XSPEC` [@xspec] interface we have developed for \texttt{XGA} is far less comprehensive than the full Python wrapping 
implemented in the `PyXspec` module, but scales with multiple cores for the analysis of multiple sources 
simultaneously much more easily. 

# Research projects using \texttt{XGA}
\texttt{XGA} is stable and appropriate for scientific use, and as such it has been used in several recent pieces of 
work; this has included an XMM analysis of the eFEDS cluster 
candidate catalogue [@efedsxcs], where we produced the first temperature calibration between XMM and 
eROSITA, a multi-wavelength analysis of an ACT selected galaxy cluster [@denisha], and XMM
follow-up of Dark Energy Survey (DES) variability selected low-mass AGN candidates [@desagn].

There are also several projects that use \texttt{XGA} nearing publication. The first of these is a hydrostatic 
and gas mass analysis of the redMaPPeR [@redmappersdss] SDSS selected XCS galaxy cluster sample [@sdssxcs] and 
well as the ACTDR5 [@actdr5] Sunyaev-Zel'dovich (SZ) selected XCS sample of galaxy clusters. This work also compares commonly measured X-ray properties of clusters 
(the X-ray luminosity L$_{\rm{x}}$, and the temperature T$_{\rm{x}}$) both to results from the existing XCS pipeline and from literature, confirming 
that \texttt{XGA} measurements are consistent with previous work. Similar work is being done on a Dark Energy Survey Year 3 (DESY3) optically 
selected XCS sample of clusters, though this will also include analysis from other XCS tools, and will not be focussed only
on mass measurements. \texttt{XGA}'s ability to stack and combine X-ray surface brightness profiles is currently being 
used, in combination with weak lensing information from DES, to look for signs of modified gravity in galaxy 
clusters. Finally an exploration of the X-ray properties of a new sample of Pea galaxies is being performed using
the point source class, the `XSPEC` interface, and the upper limit luminosity functionality.

# Future Work

# Acknowledgements
DT, KR, and PG acknowledge support from the UK Science and Technology Facilities Council via grants ST/P006760/1 (DT), 
ST/P000525/1 and ST/T000473/1 (PG, KR).

David J. Turner would like to thank Aswin P. Vijayan, Lucas Porth, and Tim Lingard for useful 
discussions during the course of writing this module.

# References
