---
title: Installation & Configuration
description: Installing XGA, setting up external high-energy astrophysics software, and configuring local or remote data archives
site:
  hide_authors: true
---

# Installation & Configuration

This is a slightly more complex installation than many Python modules, [if you wish to use XGA's full data-generation and spectral-fitting capabilities,]{.edited-text} but shouldn't be too difficult. If you're having issues feel free to get in contact [(see our GitHub repository or issue tracker)]{.edited-text}.

:::{div}
:class: placeholder-text
```{tip} Using XGA Without Observation Archives or Mission Software
You do **not** need to configure observation archives or install mission reduction software ({term}`SAS`, {term}`CIAO`, or {term}`eSASS`) if you only want to use XGA's standalone **Product** classes (such as `Image`, `ExpMap`, `RateMap`, `Spectrum`, `LightCurve`, or 1D/3D profiles), fit analytical 1D models, or fit scaling relations to your own already-reduced files. For those workflows, simply install the `xga` Python package and start using `xga.products` and `xga.models` directly!
```
:::

:::{div}
:class: placeholder-text
Choose a guide below to set up your environment and configure XGA for your workflow:
:::

:::::{grid} 1 1 2 2
:label: setup-overview-grid

::::{card}
:header: **1. Installing XGA & Dependencies**
:url: installation.md

:::{div}
:class: placeholder-text card-tint-steel
Install the `xga` Python package and set up external high-energy software (**HEASoft/XSPEC**, **CIAO**, **eSASS**, and **XMM-SAS**) primarily via Conda/Mamba.
:::
::::

::::{card}
:header: **2. Using XGA on NASA Fornax**
:url: fornax.md

:::{div}
:class: placeholder-text card-tint-teal
Run XGA directly on the NASA Fornax Science Console with pre-installed mission software, calibration databases, and cloud S3 data access.
:::
::::

::::{card}
:header: **3. Running XGA in Fornax Containers**
:url: fornax_containers.md

:::{div}
:class: placeholder-text card-tint-teal
Pull and run the pre-built `fornax-hea` Docker or Apptainer container locally on your laptop, desktop, or HPC cluster.
:::
::::

::::{card}
:header: **4. Prerequisites: Data & Region Files**
:url: data_and_regions.md

:::{div}
:class: placeholder-text card-tint-goldenrod
Learn what cleaned event lists, housekeeping files, and optional DS9 region files XGA expects for {term}`XMM-Newton`, {term}`Chandra`, and {term}`eROSITA`.
:::
::::

::::{card}
:header: **5. Configuring XGA (`xga.cfg`)**
:url: configuration.md

:::{div}
:class: placeholder-text card-tint-crimson
Configure `xga.cfg` for local archives or remote `s3://` / `https://` datasets, override paths via `XGA_CONFIG_DIR`, or switch configs at runtime with `reinitialise_xga()`.
:::
::::

::::{card}
:header: **6. Observation Census & Blacklisting**
:url: census_and_blacklist.md

:::{div}
:class: placeholder-text card-tint-violet
Understand how XGA builds its parallelized observation census on first run, refresh archives with `rebuild_census()`, and exclude problematic {term}`ObsID`s or instruments.
:::
::::

:::::
