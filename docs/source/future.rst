Planned XGA Features
========================

* **Upper Limit X-ray Luminosities** - These luminosities are measured from photometric products when there are not sufficient X-ray counts to generate a spectrum. This will involve integrating a tool that I have already written for XCS into the structure of XGA, though rather than measuring upper limit luminosities from a single image, I intend to measure them from the combined data.

* **Source Finder** - It is likely that I will extend my Hierarchical Clustering Peak Finder into a full source finder, as well as completing the implementation of a convolutional peak finder/source finder, which was inspired by my friend and colleague Lucas Porth.

* **More XSPEC Models** - Including a two temperature APEC model, and support for custom user defined models.

* **Implement emcee scaling relation fitter** - What it says on the tin really. Implement an emcee fitter for the scaling relations so that there is an MCMC option that doesn't require the installation of a bunch of optional dependencies.

* **Ability to save ScalingRelation objects** - The ability to save ScalingRelation objects to disk in some way, so that code to generate them doesn't need to be run multiple times.

