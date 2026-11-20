---
title: "X-ray: Generate and Analyse (XGA)"
subtitle: "Automated multi-mission X-ray astronomy data generation and analysis in Python"
site:
  hide_outline: true
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.6
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

**X-ray: Generate and Analyse (XGA)** {cite:p}`xgapaper;asclxga` is a Python package
designed to generate and analyse X-ray astronomy data products for individual sources
and large samples across multiple X-ray missions ({term}`XMM-Newton`, {term}`Chandra`,
and {term}`eROSITA`).

(explore-xga)=
:::::::{grid} 1 1 1 1
:label: xga-install-grid
:class: col-page-inset-right

%%%%%%% Card 1: Installation & Configuration (Full Width) %%%%%%%
::::::{card}
:header: **Install & Configure**

:::::{tab-set}
::::{tab-item} Environments
:::{div}
:class: placeholder-text
- Install XGA in pre-configured cloud environments like NASA Fornax, via the `fornax-hea` Docker/Apptainer containers, or locally with `pip` and Conda.
- Connect XGA to local installations of mission reduction software (XMM-SAS, eSASS, CIAO, and HEASoft/{term}`XSPEC`).
:::
::::
::::{tab-item} Configuration & Archives
:::{div}
:class: placeholder-text
- Configure mission directories, region files, and runtime defaults in `xga.cfg` (or via `XGA_CONFIG_DIR`).
- Point XGA at [DAXA](https://github.com/DavidT3/DAXA)-processed archives or custom directory structures on local filesystems or S3 storage.
:::
::::
:::::

:::{seealso}
:name: install-config-links
[Installation & Configuration Guide](#explore-xga)
:::
::::::

:::::::

:::::::{grid} 1 1 2 2
:label: xga-discovery-grid
:class: col-page-inset-right

%%%%%%% Card 2: Standalone Data Products %%%%%%%
::::::{card}
:header: **Use XGA Products with Your Own Data**

:::::{tab-set}
::::{tab-item} Use Cases
:::{div}
:class: placeholder-text
- Load and analyse your own reduced FITS files directly without needing to define XGA sources or samples.
- Take advantage of WCS coordinate utilities, spatial masking, publication-ready `.view()` plotting, and built-in MCMC profile fitting.
:::
::::
::::{tab-item} Product Classes
:::{div}
:class: placeholder-text
- Photometry & events: `Image`, `ExpMap`, `RateMap`, `PSF`, `PSFGrid`, and `EventList`.
- Spectra & timing: `Spectrum`, `AnnularSpectra`, `LightCurve`, and `AggregateLightCurve`.
- Profiles & relations: 1D/3D physical profiles (`xga.products.profile`) and `ScalingRelation`.
:::
::::
:::::

:::{seealso}
:name: standalone-products-links
[Standalone Data Products Guide](#explore-xga)
:::
::::::

%%%%%%% Card 3: Sources & Samples %%%%%%%
::::::{card}
:header: **Sources & Samples**

:::::{tab-set}
::::{tab-item} Individual Sources
:::{div}
:class: placeholder-text
- Define `PointSource`, `Star`, `ExtendedSource`, or `GalaxyCluster` objects from sky coordinates and redshifts.
- Automatically discover matching {term}`ObsID`s across telescopes, apply region masks, and manage generated products in one place.
:::
::::
::::{tab-item} Samples
:::{div}
:class: placeholder-text
- Group sources into `PointSample`, `ExtendedSample`, or `ClusterSample` containers to broadcast analyses across tens to thousands of objects.
- Generate multi-mission products in parallel and extract fitted parameters, luminosities, and temperatures as Astropy `Quantity` arrays.
:::
::::
:::::

:::{seealso}
:name: sources-samples-links
[Sources & Samples Overview](#explore-xga)
:::
::::::

%%%%%%% Card 4: Spectral Fitting, Models & Toolkits %%%%%%%
::::::{card}
:header: **Spectral Fitting, Models & Toolkits**

:::::{tab-set}
::::{tab-item} Spectral Fitting
:::{div}
:class: placeholder-text
- Run parallelized {term}`XSPEC` fits using pre-built models (`single_temp_apec`, `projct`) or `fakeit` count-rate simulations (`cluster_cr_conv`).
- Fit arbitrary {term}`XSPEC` model combinations or custom Tcl scripts with `general_model_fit` and `custom_xspec_fit`.
:::
::::
::::{tab-item} Models & Tools
:::{div}
:class: placeholder-text
- Fit analytical 1D surface brightness, density, temperature, and mass models (`xga.models`).
- Extract radial/sector brightness profiles, smooth or bin images (`xga.imagetools`), and measure deprojected densities and hydrostatic masses (`xga.sourcetools`).
:::
::::
:::::

:::{seealso}
:name: fitting-models-links
[Fitting, Models & Toolkits](#explore-xga)
:::
::::::

%%%%%%% Card 5: Pipelines & Science Case Studies %%%%%%%
::::::{card}
:header: **Pipelines & Science Case Studies**

:::::{tab-set}
::::{tab-item} Turnkey Pipelines
:::{div}
:class: placeholder-text
- Execute end-to-end workflows in `xga.tools`, such as `luminosity_temperature_pipeline` for iterative overdensity radii ($R_{500}$, $R_{2500}$, $R_{200}$), temperatures, luminosities, and scaling relations.
:::
::::
::::{tab-item} Case Studies
:::{div}
:class: placeholder-text
- Follow step-by-step science walkthroughs for measuring galaxy cluster thermodynamics, hydrostatic masses, and scaling relations (`xga.relations`).
- Characterize X-ray point-source populations across archival survey fields.
:::
::::
:::::

:::{seealso}
:name: pipelines-cases-links
[Pipelines & Case Studies](#explore-xga)
:::
::::::

:::::::
