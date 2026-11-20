---
title: External Mission Software
description: Installing external X-ray astronomy software (HEASoft/XSPEC, Chandra CIAO & CALDB, eROSITA eSASS, and XMM-Newton SAS)
site:
  hide_authors: true
---

# External Mission Software

XGA depends on some non-Python pieces of software, dependent on the telescope from which your data originates <span class="edited-text">and whether you are generating new data products or fitting spectra</span>.

XGA will not generate XMM products without detecting {term}`SAS`, or generate eROSITA products without detecting {term}`eSASS`, <span class="edited-text">or generate Chandra products without detecting</span> {term}`CIAO` <span class="edited-text">and</span> {term}`CALDB`<span class="edited-text">,</span> or fit spectra without detecting {term}`XSPEC`.

:::{div}
:class: placeholder-text
```{note} When Do You Need External Mission Software?
You **only** need the external software on this page if you are using XGA to:
* **Generate new data products** (images, exposure maps, spectra, ARF/RMFs, light curves, or PSFs) from cleaned event lists using {term}`SAS`, {term}`CIAO`, or {term}`eSASS`.
* **Fit spectra or run `fakeit` simulations** using {term}`HEASoft`'s {term}`XSPEC` (or query XSPEC model parameters/units).

If you only want to install the Python module to work with existing products (`xga.products`), fit 1D analytical models (`xga.models`), or fit scaling relations (`xga.relations`), see [Installing XGA](installation.md).
```

```{tip} Installing Mission Software via Conda / Mamba
Most of the external high-energy software packages that XGA wraps (**HEASoft/XSPEC**, **Chandra CIAO**, and **eROSITA eSASS**) can now be installed directly via `conda` or `mamba`, avoiding manual source builds. We present the Conda/Mamba workflows as the primary method in the tabs below, alongside links to the official manual installation guides.

If you prefer not to manage local installations at all, see [Using XGA on NASA Fornax](fornax.md) or [Running XGA in Docker Containers](docker_containers.md).
```
:::

---

:::{div}
:class: placeholder-text
## Mission & Spectral Fitting Software
:::

::::::{div}
:class: placeholder-tabs

:::::{tab-set}

::::{tab-item} HEASoft & XSPEC (All Missions)
**For all telescopes <span class="edited-text">(spectral fitting & fakeit simulations)</span>:**
* {term}`HEASoft`'s {term}`XSPEC` — <span class="edited-text">Originally tested with</span> Version 12.10.1<span class="edited-text">, and compatible with modern HEASoft 6.3x / XSPEC 12.13+ releases.</span>

:::{div}
:class: placeholder-text
### Primary Method: Install via Conda / Mamba
**HEASoft** (including XSPEC and PyXspec) is distributed via the official HEASARC Conda channel and can be installed directly into a Conda/Mamba environment on Linux and macOS:

```bash
conda create -n xga-env -c https://hea-www.harvard.edu/conda/heasoft -c conda-forge python=3.12 heasoft
conda activate xga-env
pip install xga
```

When activating the environment, Conda automatically sets `$HEADAS` and initializes HEASoft/XSPEC so `xspec` is on your `$PATH` and `model.dat` can be parsed by XGA.

### Alternative: Manual / Source Installation
If you prefer a standalone system installation or only want to build the XSPEC subset of HEASoft from source, please follow the official [HEASoft Installation Guide](https://heasarc.gsfc.nasa.gov/lheasoft/install.html). Make sure `$HEADAS` is set and initialized (`source $HEADAS/headas-init.sh`) before running XGA.
:::
::::

::::{tab-item} Chandra (CIAO & CALDB)
:::{div}
:class: placeholder-text
**For Chandra data:**
* Chandra Interactive Analysis of Observations ({term}`CIAO`) and the Chandra Calibration Database ({term}`CALDB`).

```{warning} Conda Environment Conflict: CIAO vs. HEASoft
**Important (especially for developers and multi-mission environments):** `ciao` and `heasoft` **cannot currently be installed into the same Conda environment** via `conda`/`mamba` due to conflicting shared library pins (such as `cfitsio`, `wcslib`, and `ccfits`) between the CXC and HEASARC Conda channels.

If you need both CIAO and HEASoft available in the same shell session, install one via Conda (e.g. CIAO inside your Conda environment) and initialize a standalone installation of the other (e.g. a system-level HEASoft installation via `source $HEADAS/headas-init.sh`), or use a pre-built [Docker container environment](docker_containers.md).
```

### Primary Method: Install via Conda / Mamba
The Chandra X-ray Center distributes CIAO and the Chandra CALDB directly as Conda packages:

```bash
conda create -n xga-ciao -c https://cxc.cfa.harvard.edu/conda/ciao -c conda-forge python=3.12 ciao sherpa ciao-contrib caldb_main
conda activate xga-ciao
pip install xga
```

XGA verifies CIAO and CALDB availability on demand by running `ciaover -v`.

### Alternative: `ciao-install` Script
You can also install CIAO and CALDB as a standalone distribution using the official [CIAO Installation Guide](https://cxc.harvard.edu/ciao/download/).
:::
::::

::::{tab-item} eROSITA & eRASS (eSASS)
**For eROSITA data:**
* eROSITA Science Analysis Software System ({term}`eSASS`) — <span class="edited-text">EDR (`eSASS4EDR`) or DR1 (`eSASS4DR1`) version.</span> It must be installed on the system <span class="edited-text">(or inside the same container where XGA is running; XGA does not invoke an external Docker container from the host)</span>.

:::{div}
:class: placeholder-text
### Primary Method: Install via Conda / Mamba
**eSASS** (`eSASS4DR1`) can be installed via Conda/Mamba (for instance via the HEASARC / Fornax high-energy Conda channels or the MPE eROSITA Conda distribution):

```bash
conda create -n xga-esass -c conda-forge python=3.12
conda activate xga-esass
# Install eSASS from the configured channel or local eSASS conda package, then install XGA:
pip install xga
```

Ensure that the eSASS initialization script has been sourced so that tasks like `evtool` and `srctool` are available on your `$PATH`.

### Alternative: Manual / Tarball Installation
See the official [eSASS Installation Guide](https://erosita.mpe.mpg.de/dr1/eSASS4DR1/eSASS4DR1_installation/) (or the [EDR Installation Guide](https://erosita.mpe.mpg.de/edr/DataAnalysis/esassinstall.html)) for instructions on building or unpacking eSASS manually and downloading the eROSITA calibration database.
:::
::::

::::{tab-item} XMM-Newton (SAS)
**For XMM data:**
* XMM's Science Analysis System ({term}`SAS`) — <span class="edited-text">Version 17.0.0 or later (including SAS 20, 21, and 22),</span> should be largely compatible with the software. SAS version 14.0.0 however, does not support features that PSF correction of images depends on.

:::{div}
:class: placeholder-text
### Installation & Environment Variables
XMM-SAS requires **HEASoft** to be initialized in your environment and can be installed using the official [SAS Installation Guide](https://www.cosmos.esa.int/web/xmm-newton/sas-installation) (or used pre-installed inside [Fornax containers](docker_containers.md)).

Before using XGA to generate XMM-Newton products, ensure that:
1. The SAS initialization script (`setsas.sh` / `setsas.csh`) has been sourced so `SAS_DIR` is set.
2. You have downloaded the XMM Current Calibration Files (CCF) and exported `SAS_CCFPATH` pointing to your CCF directory:

```bash
export SAS_CCFPATH=/path/to/xmm/ccf
source /path/to/sas/setsas.sh
```
:::
::::

:::::
::::::

---

:::{div}
:class: placeholder-text
## Next Steps

* If you have not yet installed the `xga` Python package itself, see [Installing XGA](installation.md).
* Once your software environment is ready, proceed to [Prerequisites: Data & Region Files](data_and_regions.md) and [Configuring XGA (`xga.cfg`)](configuration.md) to connect XGA to your observation archives.
:::
