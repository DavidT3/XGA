Planned XGA Features
========================

* **Upper Limit X-ray Luminosities** - These luminosities are measured from photometric products when there are not sufficient X-ray counts to generate a spectrum. This will involve integrating a tool that I have already written for XCS into the structure of XGA, though rather than measuring upper limit luminosities from a single image, I intend to measure them from the combined data.

* **Source Finder** - It is likely that I will extend my Hierarchical Clustering Peak Finder into a full source finder, as well as completing the implementation of a convolutional peak finder/source finder, which was inspired by my friend and colleague Lucas Porth.

* **More XSPEC Models** - Including a two temperature APEC model, and support for custom user defined models.

* **Implement emcee scaling relation fitter** - What it says on the tin really. Implement an emcee fitter for the scaling relations so that there is an MCMC option that doesn't require the installation of a bunch of optional dependencies.

* **Ability to save ScalingRelation objects** - The ability to save ScalingRelation objects to disk in some way, so that code to generate them doesn't need to be run multiple times.

* **Support for other X-ray telescopes** - Support for generation and analysis of data products from other telescopes.

* **Creating a Docker image for users to download** - Creating a Docker environment with SAS and HEASoft already installed, for ease of use.

* **Method for finding poorly removed point sources** - An attempt to mitigate occasional mistakes by source finders that produce regions that don't necessarily remove the entire point source.

* **Overdensity radius measurement** - Using HydrostaticMass profiles to measure the overdensity radius of a cluster.
