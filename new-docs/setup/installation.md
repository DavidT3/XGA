---
title: Installing XGA & Dependencies
description: Installing the XGA Python module and external X-ray astronomy software (HEASoft, CIAO, eSASS, and SAS)
site:
  hide_authors: true
---

# Installing XGA & Dependencies

:::{div}
:class: placeholder-text
XGA can be installed as a standard Python package on its own (which is sufficient for working with standalone XGA products, analytical 1D models, and scaling relations), or paired with external high-energy astrophysics software to generate new data products and fit spectra.
:::

## The XGA Python Module

We **strongly recommend** that you make use of Python virtual environments, or (even better) Conda/Mamba virtual environments when installing XGA.

::::{tab-set}

:::{tab-item} Conda / Mamba + pip (Recommended)
If using Conda or Mamba, you can create a fresh environment and then install XGA (and its dependencies) via `pip`:

```bash
conda create -n <YOUR ENVIRONMENT NAME GOES HERE> python=3.12
conda activate <YOUR ENVIRONMENT NAME GOES HERE>
pip install xga
```
:::

:::{tab-item} Standard pip (PyPI)
XGA is available on the popular Python Package Index (PyPI), and can be installed like this:

```bash
pip install xga
```
:::

:::{tab-item} From GitHub Source
You can also fetch the current working version from the git repository, and install it:

```bash
git clone https://github.com/DavidT3/XGA
cd XGA
python -m pip install .
```

Alternatively you could use the 'editable' option so that any changes you pull from the remote repository are reflected without having to reinstall XGA.

```bash
git clone https://github.com/DavidT3/XGA
cd XGA
python -m pip install --editable .
```
:::

::::

[All required Python modules are declared in `pyproject.toml` (and `requirements.txt`),]{.edited-text} and should be added to your system during the installation of XGA.

---

## External High-Energy Software Dependencies

XGA depends on some non-Python pieces of software, dependent on the telescope from which your data originates [and whether you are generating new data products or fitting spectra]{.edited-text}.

XGA will not generate XMM products without detecting {term}`SAS`, or generate eROSITA products without detecting {term}`eSASS`, [or generate Chandra products without detecting {term}`CIAO` and {term}`CALDB`,]{.edited-text} or fit spectra without detecting {term}`XSPEC`.

:::{div}
:class: placeholder-text
```{tip} Installing Mission Software via Conda / Mamba
Most of the external high-energy software packages that XGA wraps (**HEASoft/XSPEC**, **Chandra CIAO**, and **eROSITA eSASS**) can now be installed directly via `conda` or `mamba`, avoiding manual source builds. We present the Conda/Mamba workflows as the primary method in the tabs below, alongside links to the official manual installation guides.

If you prefer not to manage local installations at all, see [Using XGA on NASA Fornax](fornax.md) or [Running XGA in Fornax Containers](fornax_containers.md).
```
:::

:::::{tab-set}

::::{tab-item} HEASoft & XSPEC (All Missions)
**For all telescopes (spectral fitting & fakeit simulations):**
* {term}`HEASoft`'s {term}`XSPEC` — [Originally tested with Version 12.10.1, and compatible with modern HEASoft 6.3x / XSPEC 12.13+ releases.]{.edited-text}

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

If you need both CIAO and HEASoft available in the same shell session, install one via Conda (e.g. CIAO inside your Conda environment) and initialize a standalone installation of the other (e.g. a system-level HEASoft installation via `source $HEADAS/headas-init.sh`), or use the pre-built [Fornax container environment](fornax_containers.md).
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
* eROSITA Science Analysis Software System ({term}`eSASS`) — [EDR (`eSASS4EDR`) or DR1 (`eSASS4DR1`) version.]{.edited-text} It must be installed on the system [(or inside the same container where XGA is running; XGA does not invoke an external Docker container from the host)]{.edited-text}.

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
* XMM's Science Analysis System ({term}`SAS`) — [Version 17.0.0 or later (including SAS 20, 21, and 22),]{.edited-text} should be largely compatible with the software. SAS version 14.0.0 however, does not support features that PSF correction of images depends on.

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

---

## Optional Dependencies

XGA can also make use of external software for some limited tasks, but they are not required to use the module as a whole:

* The **R** interpreter.
* **rpy2** — A Python module that provides an interface with the R language in Python.
* **LIRA** — An R fitting package.

The R interpreter, `rpy2`, and `LIRA` are all necessary only if you wish to use the LIRA scaling relation fitting function (`xga.relations.fit.scaling_relation_lira`).
