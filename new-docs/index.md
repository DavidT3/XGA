---
title: "X-ray: Generate and Analyse (XGA)"
subtitle: "Automated multi-mission X-ray astronomy data generation and analysis in Python"
downloads: []
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
:::::{grid} 1 1 1 1
:label: xga-install-grid

%%%%% Card 1: Installation & Configuration (Full Width) %%%%%
::::{card}
:header: **Install & Configure**
:url: #explore-xga

:::{div}
:class: placeholder-text
Set up XGA locally or in cloud/container environments, link mission reduction software (XMM-SAS, eSASS, CIAO, and HEASoft/XSPEC), and configure local or DAXA-processed data archives.
:::
::::

:::::

:::::{grid} 1 1 2 2
:label: xga-discovery-grid

%%%%% Card 2: Standalone Data Products %%%%%
::::{card}
:header: **Use XGA Products with Your Own Data**
:url: #explore-xga

:::{div}
:class: placeholder-text
Load and analyse your own reduced FITS images, exposure maps, spectra, event lists, and 1D/3D profiles directly using XGA's product classes without defining XGA sources.
:::
::::

%%%%% Card 3: Sources & Samples %%%%%
::::{card}
:header: **Sources & Samples**
:url: #explore-xga

:::{div}
:class: placeholder-text
Define point and extended X-ray sources or large multi-object samples to automatically discover archival ObsIDs, apply spatial masks, and generate multi-mission products in parallel.
:::
::::

%%%%% Card 4: Spectral Fitting, Models & Toolkits %%%%%
::::{card}
:header: **Spectral Fitting, Models & Toolkits**
:url: #explore-xga

:::{div}
:class: placeholder-text
Run parallelized XSPEC spectral fits (including `general_model_fit` and `custom_xspec_fit`), fit analytical 1D models, and measure radial profiles, deprojected densities, and hydrostatic masses.
:::
::::

%%%%% Card 5: Pipelines & Science Case Studies %%%%%
::::{card}
:header: **Pipelines & Science Case Studies**
:url: #explore-xga

:::{div}
:class: placeholder-text
Execute end-to-end analysis pipelines like `luminosity_temperature_pipeline` and explore step-by-step science walkthroughs for galaxy clusters and X-ray point-source populations.
:::
::::

:::::
