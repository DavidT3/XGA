---
title: Installation & Configuration
description: Installing XGA, setting up external high-energy astrophysics software, and configuring local or remote data archives
site:
  hide_authors: true
---

# Installation & Configuration

This is a slightly more complex installation than many Python modules, <span class="edited-text">if you wish to use XGA's full data-generation and spectral-fitting capabilities,</span> but shouldn't be too difficult. If you're having issues feel free to get in contact <span class="edited-text">(see our GitHub repository or issue tracker)</span>.

:::::{div}
:class: placeholder-text
Expand any of the sections below to check whether you need that step for your workflow before diving in:

::::{dropdown} 1. Installing XGA — *Required for all users*
:open:

* **Who needs this**: Everyone.
* **What it covers**:
  * Installing the `xga` Python package into a Conda/Mamba or `venv` environment via `pip install xga` or from GitHub source.
  * Optional R / `rpy2` / `LIRA` dependencies for hierarchical Bayesian scaling-relation fits (`scaling_relation_lira`).
* [Go to Installing XGA](installation.md)
::::

::::{dropdown} 2. External Mission Software — *Required for product generation & spectral fitting*
* **Who needs this**: Users generating new data products from event lists ({term}`SAS`, {term}`CIAO`, {term}`eSASS`) or fitting spectra and running simulations ({term}`HEASoft` / {term}`XSPEC`).
* **What it covers**:
  * Installing **HEASoft & XSPEC**, **Chandra CIAO & CALDB**, **eROSITA eSASS**, and **XMM-Newton SAS**, with **Conda/Mamba as the primary installation method** alongside links to official manual builds.
  * Important multi-mission environment notes, including the current Conda dependency conflict between `ciao` and `heasoft`.
* [Go to External Mission Software](external_software.md)
::::

::::{dropdown} 3. Prerequisites: Data & Region Files — *Required for Source & Sample workflows*
* **Who needs this**: Users analysing observations through XGA's `Source` and `Sample` classes (`BaseSource`, `ExtendedSource`, `PointSource`, `GalaxyCluster`, `ClusterSample`, etc.).
* **What it covers**:
  * Required per-{term}`ObsID` files for **XMM-Newton** (`xmm`), **Chandra** (`chandra`), and **eROSITA / eRASS** (`erosita`, `erass`), and how {term}`DAXA` can prepare these archives automatically.
  * Optional pre-generated images, exposure maps, and DS9 `.reg` region files.
  * XGA's region colour classification scheme (`red` for point sources; `green`, `magenta`, `blue`, `cyan`, and `yellow` for extended sources) used for automated interloper masking.
* [Go to Prerequisites: Data & Region Files](data_and_regions.md)
::::

::::{dropdown} 4. Configuring XGA (`xga.cfg`) — *Required for Source & Sample workflows*
* **Who needs this**: Users connecting XGA to local observation directories or remote cloud datasets.
* **What it covers**:
  * How XGA's lazy initialisation creates `~/.config/xga/xga.cfg` on first use.
  * Setting `xga_save_path`, `num_cores`, `root_<tel>_dir`, and `{obs_id}` / `{lo_en}` / `{hi_en}` file naming templates.
  * Pointing `root_<tel>_dir` directly at **remote datasets** (`s3://` buckets and `https://` servers via `fsspec`).
  * Switching configurations via `XGA_CONFIG_DIR` or at runtime with `xga.reinitialise_xga()`.
  * Complete copy-pasteable `xga.cfg` examples for DAXA archives, remote S3/HTTPS datasets, and classic local directory layouts.
* [Go to Configuring XGA](configuration.md)
::::

::::{dropdown} 5. Observation Census & Blacklisting — *Required for Source & Sample workflows*
* **Who needs this**: Users managing local or remote observation archives over time or excluding bad exposures.
* **What it covers**:
  * How XGA's parallelized first-run census scans FITS headers and stores `<tel>/<tel>_census.csv`.
  * Updating, full-rebuilding, or pruning deleted observations with `xga.rebuild_census()`.
  * Excluding specific {term}`ObsID`s or individual instruments (`EXCLUDE_PN`, `EXCLUDE_MOS1`, `EXCLUDE_ACIS`, `EXCLUDE_TM1`–`TM7`) via `<tel>/<tel>_blacklist.csv`.
* [Go to Observation Census & Blacklisting](census_and_blacklist.md)
::::

::::{dropdown} 6 & 7. NASA Fornax Science Console & Docker Containers — *Optional pre-configured environments*
* **Who needs this**: Users who want to run XGA in a pre-built environment where HEASoft, SAS, CIAO, eSASS, and calibration databases are already installed, or who want to run analyses directly alongside HEASARC cloud data on AWS.
* **What they cover**:
  * **NASA Fornax Science Console**: Launching an interactive JupyterLab session on Fornax, selecting high-energy astrophysics kernels, and streaming S3 archives.
  * **Docker & Apptainer Containers**: Pulling pre-built images to run XGA locally on macOS/Linux/Windows (Docker) or on university/national supercomputing clusters (Apptainer/Singularity).
* [Go to Using XGA on NASA Fornax](fornax.md) | [Go to Running XGA in Docker Containers](docker_containers.md)
::::

```{tip} Using XGA Without Observation Archives or Mission Software
You do **not** need to configure observation archives or install [External Mission Software](external_software.md) ({term}`SAS`, {term}`CIAO`, or {term}`eSASS`) if you only want to use XGA's standalone **Product** classes (such as `Image`, `ExpMap`, `RateMap`, `Spectrum`, `LightCurve`, or 1D/3D profiles), fit analytical 1D models, or fit scaling relations to your own already-reduced files. For those workflows, simply [install the `xga` Python package](installation.md) and start using `xga.products` and `xga.models` directly!
```
:::::
