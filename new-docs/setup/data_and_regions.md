---
title: "Prerequisites: Data & Region Files"
description: Cleaned event lists, mission housekeeping files, and DS9 region file conventions expected by XGA
site:
  hide_authors: true
---

# Prerequisites: Data & Region Files

## Data Required for Using XGA Sources & Samples

**This is very important** — Currently, to make use of [XGA's `Source` and `Sample` classes]{.edited-text}, you **must** have access to **at least one** of: cleaned {term}`XMM-Newton`, [{term}`Chandra`,]{.edited-text} or {term}`eROSITA` [({term}`eRASS` or pointed/CalPV)]{.edited-text} event lists. For acquiring and cleaning X-ray data, the Python module [{term}`DAXA` (`daxa`)](https://github.com/DavidT3/DAXA) is recommended.

:::{div}
:class: placeholder-text
```{note} Standalone XGA Product Classes Do Not Require Archive Setup
If you only want to load and analyse specific FITS files (such as an existing spectrum, image, exposure map, or event list) using `xga.products`, you do **not** need a full observation archive directory structure. The requirements below apply when using XGA's automated `Source` and `Sample` classes.
```

Depending on which X-ray mission(s) you are analysing, XGA expects each {term}`ObsID` directory inside your configured `root_<telescope>_dir` to contain specific reduced files (and optionally pre-generated images and exposure maps):
:::

:::::{tab-set}

::::{tab-item} XMM-Newton (`xmm`)
:::{div}
:class: placeholder-text
**Required per ObsID:**
* **Cleaned event lists**: Separate cleaned event lists for each available EPIC instrument (`clean_pn_evts`, `clean_mos1_evts`, `clean_mos2_evts`).
* **Attitude file**: The spacecraft attitude history file (`attitude_file`, e.g., `P{obs_id}OBX000ATTTSR0000.FIT`).

**Optional per ObsID:**
* Pre-generated images (`pn_image`, `mos1_image`, `mos2_image`) and exposure maps (`pn_expmap`, `mos1_expmap`, `mos2_expmap`) in energy bands specified by `lo_en` and `hi_en`. (If omitted, XGA can generate them on demand using {term}`SAS`.)
* A DS9 region file (`region_file`).
:::
::::

::::{tab-item} Chandra (`chandra`)
:::{div}
:class: placeholder-text
**Required per ObsID:**
* **Cleaned event list**: Cleaned ACIS event list (`clean_acis_evts`).
* **Aspect solution file**: Spacecraft attitude/aspect solution (`attitude_file`).
* **Bad pixel & mask files**: Observation-specific ACIS bad pixel file (`acis_badpix_file`) and mask file (`acis_mask_file`).

**Optional per ObsID:**
* Pre-generated ACIS images (`acis_image`) and exposure maps (`acis_expmap`) in energy bands specified by `lo_en` and `hi_en`. (If omitted, XGA can generate them on demand using {term}`CIAO`.)
* A DS9 region file (`region_file`).
:::
::::

::::{tab-item} eROSITA & eRASS (`erosita` / `erass`)
:::{div}
:class: placeholder-text
XGA supports both pointed/CalPV eROSITA observations (`erosita`) and all-sky survey tiles (`erass`).

**Required per ObsID:**
* **Cleaned event list(s)**: Either a single combined event list across telescope modules (`clean_erosita_evts` / `clean_erass_evts`, which is standard for CalPV and eRASS releases) or separate `clean_tm1_evts`–`clean_tm7_evts` files.

**Optional per ObsID:**
* Pre-generated images (`erosita_image` / `erass_image`) and exposure maps (`erosita_expmap` / `erass_expmap`) in energy bands specified by `lo_en` and `hi_en`. (If omitted, XGA can generate them on demand using {term}`eSASS`.)
* A DS9 region file (`region_file`).
:::
::::

:::::

---

## Region Files

It will be beneficial if you have region files available, as it will allow XGA to remove interloper sources. If you wish to use existing region files, then they must be in a DS9 compatible format, **point sources** must be **red** and **extended sources** must be **green** [(or one of the additional XCS extended-source colour classifications listed below)]{.edited-text}.

```{warning} XMM-Newton Region File WCS Requirement
**Disclaimer:** If region files are supplied for XMM data, XGA also expects at least one image per instrument per observation, for WCS information.
```

:::{div}
:class: placeholder-text
### Region Colour Classifications

XGA uses the colour of each region in a DS9 `.reg` file (stored in `xga.utils.SRC_REGION_COLOURS`) to distinguish point-like contaminants from extended emission when automatically masking interlopers and background regions (originating from the XMM Cluster Survey / XAPA classification scheme):

| Region Colour | XGA Source Category | Meaning in XCS / XAPA |
| :--- | :--- | :--- |
| `red` | **Point Source (`pnt`)** | Point source (masked out during extended-source analysis unless it is the target of a `PointSource`) |
| `green` | **Extended Source (`ext`)** | Standard extended source |
| `magenta` | **Extended Source (`ext`)** | PSF-sized extended source |
| `blue` | **Extended Source (`ext`)** | Extended source with significant point-source contribution |
| `cyan` | **Extended Source (`ext`)** | Extended source with significant Run1 contribution |
| `yellow` | **Extended Source (`ext`)** | Extended source with fewer than 10 counts |

### Example DS9 Region File (`.reg`)

Region files can be specified in celestial (`fk5` / `icrs`) or image/physical coordinates (provided a reference image with valid WCS headers is available). Below is an example DS9 `.reg` file in `fk5` coordinates containing two extended sources (`green`) and three point sources (`red`):

```text
# Region file format: DS9 version 4.1
global color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
fk5
ellipse(36.369583,-4.456111,65.2",52.8",134.5) # color=green
circle(36.412083,-4.410278,12.5") # color=red
circle(36.321667,-4.489167,10.0") # color=red
ellipse(36.295000,-4.391111,42.0",38.1",12.0) # color=green
circle(36.380417,-4.448889,9.5") # color=red
```
:::
