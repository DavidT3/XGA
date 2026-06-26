Planned XGA Features
========================

* **Support for other X-ray telescopes** - XGA's purpose is to facilitate the exploitation of the complete public X-ray archive, hence the aim is to support all X-ray telescopes eventually. Currently XMM and eROSITA are supported, and the software is setup so that integrating new telescope-specific softwares should not be an arduous task. If you are interested in integrating a telescope into XGA please reach out to the development team (:doc:`support`). 

* **Creating a Docker image for users to download** - Creating a Docker environment with all relevant software already installed.

* **More XSPEC Models** - Support for custom user defined models.

* **Upper Limit X-ray Luminosities** - These luminosities are measured from photometric products when there are not sufficient X-ray counts to generate a spectrum. This will support measurements from combined data, as well as single images. 

* **Implement emcee scaling relation fitter** - Implement an emcee fitter for the scaling relations so that there is an MCMC option that doesn't require the installation of optional dependencies.

* **Method for finding poorly removed point sources** - An attempt to mitigate occasional mistakes by source finders that produce regions that don't necessarily remove the entire point source.

