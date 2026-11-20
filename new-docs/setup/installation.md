---
title: Installing XGA
description: Installing the XGA Python module and optional Python/R dependencies
site:
  hide_authors: true
---

# Installing XGA

:::{div}
:class: placeholder-text
XGA can be installed as a standard Python package on its own (which is all you need for working with standalone XGA products, analytical 1D models, and scaling relations), or paired with [External Mission Software](external_software.md) ({term}`HEASoft`/{term}`XSPEC`, {term}`CIAO`, {term}`eSASS`, and {term}`SAS`) to generate new data products and fit spectra.
:::

## <span class="edited-text">The XGA Python</span> Module

We **strongly recommend** that you make use of Python virtual environments, or (even better) Conda/Mamba virtual environments when installing XGA.

:::::{div}
:class: placeholder-tabs

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
:::::

<span class="edited-text">All required Python modules are declared in `pyproject.toml` (and `requirements.txt`),</span> and should be added to your system during the installation of XGA.

:::{div}
:class: placeholder-text
```{tip} Need Mission Reduction Software or XSPEC?
If your workflow involves generating new X-ray data products from cleaned event lists or fitting spectra with {term}`XSPEC`, see the next page on [External Mission Software](external_software.md) for instructions on installing **HEASoft/XSPEC**, **Chandra CIAO & CALDB**, **eROSITA eSASS**, and **XMM-Newton SAS** (including Conda/Mamba commands that install both the mission software and XGA in one environment).

Alternatively, you can use XGA with all mission software pre-installed on [NASA Fornax](fornax.md) or inside [Docker Containers](docker_containers.md).
```
:::

---

## Optional Dependencies

XGA can also make use of external software for some limited tasks, but they are not required to use the module as a whole:

* The **R** interpreter.
* **rpy2** — A Python module that provides an interface with the R language in Python.
* **LIRA** — An R fitting package.

The R interpreter, `rpy2`, and `LIRA` are all necessary only if you wish to use the LIRA scaling relation fitting function (`xga.relations.fit.scaling_relation_lira`).
