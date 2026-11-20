---
title: Configuring XGA
description: Setting up xga.cfg, multi-mission archive paths, remote S3/HTTPS data directories, XGA_CONFIG_DIR, and reinitialise_xga()
site:
  hide_authors: true
---

# Configuring XGA

Before XGA [sources and samples]{.edited-text} can be used you must fill out a configuration file (completed examples can be found [at the bottom of this page](#example-configuration-files) and in the repository [here](https://github.com/DavidT3/XGA/blob/master/docs/example_config/xga.cfg)).

This section aims to be as general as possible, but is biased by how our research group generates and stores our data products [(and by the directory structure produced by {term}`DAXA`)]{.edited-text}. If you are an X-ray astronomer who wishes to use this module, but it seems to be incompatible with your setup, please get in touch or raise an issue.

---

## Step-by-Step Configuration Guide

Follow these steps to fill out the configuration file:

1. **Trigger initial configuration file creation**:
   [Import XGA and access any configuration-dependent attribute (such as `xga.CONFIG_FILE`), or instantiate a source,]{.edited-text} to generate the initial, incomplete, configuration file.
   * **Note**: XGA uses a **lazy loading** architecture. Importing the module does not perform any filesystem checks or software discovery immediately. These only occur when a configuration variable or data-dependent function is first accessed.

   :::{div}
   :class: placeholder-text
   ```python
   import xga

   # Accessing CONFIG_FILE triggers initial creation of ~/.config/xga/xga.cfg if it doesn't exist
   print(xga.CONFIG_FILE)
   ```
   :::

2. **Open `xga.cfg`**:
   Navigate to `~/.config/xga` [(or `$XDG_CONFIG_HOME/xga` if `XDG_CONFIG_HOME` is set)]{.edited-text} and open `xga.cfg` in a text editor. The `.config` directory is usually hidden, so it is probably easier to navigate via the terminal.
   * **Note**: The location of the configuration directory can be overridden by setting the `XGA_CONFIG_DIR` environment variable. If set, XGA will look for its configuration and store its census files in exactly that directory. Because of XGA's lazy loading, you can even set this environment variable **after** importing XGA in a session.
   * **Note**: If you need to switch configuration directories [or override the output directory]{.edited-text} mid-session, you can use the `reinitialise_xga()` function [(see [Custom Config Locations & Runtime Re-initialisation](#custom-config-locations-runtime-re-initialisation) below for details)]{.edited-text}.

3. **Identify required entries**:
   Take note of the entries that currently have `/this/is/required/` at the beginning; without these entries [the corresponding telescope section]{.edited-text} will not function. [Any telescope section whose required entries are left as `/this/is/required/...` is simply marked as `USABLE[tel] = False` and ignored, while optional entries can be left with `/this/is/optional/...` if you do not have pre-generated images, exposure maps, or region files.]{.edited-text}

4. **Set the XGA output directory (`xga_save_path`)**:
   Set the directory in which XGA will save the products and files it generates. It is advised to just set it to `xga_output`, so wherever you run a script that imports XGA it will create a folder called `xga_output` there. You could choose to use an absolute path and have a global XGA folder; however, it wouldn't make a lot of sense.

5. **Set the CPU core limit (`num_cores`)**:
   You may also set an optional parameter in the `[XGA_SETUP]` section, `'num_cores'`. [By default, new configuration files set `num_cores = auto`, which allows XGA to use up to 90% of the available CPU cores on your system.]{.edited-text} If you wish to manually limit the number of cores that XGA is allowed to use, then set this to an integer value, e.g. `num_cores = 10`. You can also set this at runtime, by importing `NUM_CORES` from `xga` and setting that to a value.

6. **Configure mission root directories (`root_<telescope>_dir`)**:
   The `root_<telescope>_dir` entry ( inside `[XMM_FILES]`, `[CHANDRA_FILES]`, `[EROSITA_FILES]`, or `[ERASS_FILES]`) is the path [(or remote `s3://` / `https://` URI)]{.edited-text} of the parent folder containing all of your observation data for `<telescope>`. It is not necessary to have data for all telescopes that XGA supports.

7. **Configure file naming patterns & relative paths**:
   Most of the other entries tell XGA how different files are named. `clean_pn_evts`, for instance, gives the naming convention for the cleaned PN events files that XGA generates products from.
   :::{div}
   :class: placeholder-text
   * **Relative vs. Absolute Paths**: Any file path in a `[<TELESCOPE>_FILES]` section that does **not** start with a leading `/` (and does not already start with `root_<telescope>_dir`) is treated as **relative to `root_<telescope>_dir`** and automatically has `root_<telescope>_dir` prepended during initialisation. If your files (such as `region_file`) live outside `root_<telescope>_dir`, provide an **absolute** path starting with `/`.
   :::

8. **String formatting templates (`{obs_id}`, `{lo_en}`, `{hi_en}`)**:
   Bear in mind when filling in the file fields that XGA uses the Python string formatting convention, so **anywhere you see `{obs_id}` will be formatted with the {term}`ObsID` of interest when XGA is actually running**.

9. **Pre-existing image & exposure map energy bounds (`lo_en` and `hi_en`)**:
   The `lo_en` and `hi_en` entries can be used to tell XGA what images and exposure maps you may already have. For instance, if you already had 0.50–2.00 keV and 2.00–10.00 keV images and exposure maps, you could set `lo_en = ['0.50', '2.00']` and `hi_en = ['2.00', '10.00']`.

10. **Region file paths (`region_file`)**:
    Finally, the `region_file` entry tells XGA where region files for each observation are stored (if they exist).

---

(custom-config-locations-runtime-re-initialisation)=
## Custom Config Locations & Runtime Re-initialisation

:::{div}
:class: placeholder-text
XGA provides flexible ways to manage multiple configuration files (for example, switching between different projects, surveys, or test archives) without editing `~/.config/xga/xga.cfg` by hand:

::::{tab-set}

:::{tab-item} `reinitialise_xga()` (At Runtime)
You can call `xga.reinitialise_xga()` at any point during a Python session or notebook to switch to a different configuration directory, reload changes made to `xga.cfg`, or override the output directory (`OUTPUT`) where XGA saves generated products:

```python
import xga

# Switch to a different configuration directory AND override the output save path
xga.reinitialise_xga(
    config_dir="/path/to/project_config/",
    override_output_dir="/path/to/custom_xga_output/",
)

# Or reload the current configuration file with a new output directory only
xga.reinitialise_xga(override_output_dir="new_analysis_output")
```
:::

:::{tab-item} `XGA_CONFIG_DIR` (Environment Variable)
You can set `XGA_CONFIG_DIR` in your shell before launching Python (or inside Python via `os.environ` **before** accessing any lazy-loaded XGA attributes):

```bash
export XGA_CONFIG_DIR=/path/to/project_config/
python run_analysis.py
```

Or in Python prior to triggering XGA's lazy initialisation:

```python
import os
import xga

os.environ["XGA_CONFIG_DIR"] = "/path/to/project_config/"
# Initialisation happens here when CENSUS is first accessed
print(xga.CENSUS.keys())
```
:::

::::
:::

---

## Remote Data Access (`s3://`, `https://`, and SFTP)

:::{div}
:class: placeholder-text
XGA natively supports pointing `root_<telescope>_dir` in `xga.cfg` directly at **remote datasets** hosted on **Amazon S3 buckets (`s3://`)** (such as the open-access HEASARC cloud archive on AWS / NASA Fornax) or **HTTP/HTTPS web servers (`https://` / `http://`)** using `fsspec` and `astropy`'s lazy header streaming.

When a remote `root_<telescope>_dir` URI is configured:
* XGA inspects the URI protocol (`s3`, `https`, `http`, or local filesystem) via `fsspec` and stores the remote filesystem object in `xga.ROOT_DIR_FS`.
* During the initial observation census, XGA streams only the required FITS headers (`lazy_load_hdus=True`, `use_fsspec=True`) in parallel without downloading full event tables into memory.
* Relative file paths in `[<TELESCOPE>_FILES]` (such as `clean_pn_evts = {obs_id}/pps/...`) are automatically resolved relative to the remote `s3://` or `https://` root URI.

```{note} Public S3 Buckets (`anon=True`)
Currently, `s3://` URIs in `xga.cfg` connect anonymously (`anon=True`), which is designed for open-access astronomy buckets such as the public HEASARC S3 archive.
```
:::

::::{dropdown} Alternative: Mounting a Remote Server via SFTP / SSHFS
If your data lives on a private remote institutional server (accessible via SSH rather than S3/HTTPS), and you want to use XGA on a local machine, it is recommended to set up an SFTP connection and mount the server as an external volume. Then you can fill out the configuration file with local paths going through the mount folder. To mount a server, one can follow the steps detailed in this [guide](https://sftptogo.com/blog/how-to-mount-sftp-as-a-drive-on-mac/).
::::

---

(example-configuration-files)=
## Example Configuration Files

:::{div}
:class: placeholder-text
Below are three complete examples of `xga.cfg` demonstrating a multi-mission **{term}`DAXA`-processed archive**, a **classic local directory layout**, and a **remote S3 / HTTPS archive configuration**:
:::

:::::{tab-set}

::::{tab-item} Multi-Mission DAXA Archive (`xga.cfg`)
:::{div}
:class: placeholder-text
This example shows `xga.cfg` configured for XMM-Newton, eRASS, and Chandra archives processed by {term}`DAXA`, using paths relative to `root_<tel>_dir`:

```ini
[XGA_SETUP]
xga_save_path = xga_output
num_cores = auto

[XMM_FILES]
root_xmm_dir = /data/daxa_archives/my_survey/processed_data/xmm_pointed/
clean_pn_evts = {obs_id}/events/obsid{obs_id}-instPN-subexpALL-en-finalevents.fits
clean_mos1_evts = {obs_id}/events/obsid{obs_id}-instM1-subexpALL-en-finalevents.fits
clean_mos2_evts = {obs_id}/events/obsid{obs_id}-instM2-subexpALL-en-finalevents.fits
attitude_file = {obs_id}/P{obs_id}OBX000ATTTSR0000.FIT
lo_en = ['0.50', '2.00']
hi_en = ['2.00', '10.00']
pn_image = {obs_id}/images/obsid{obs_id}-instPN-subexpALL-en{lo_en}_{hi_en}keV-image.fits
mos1_image = {obs_id}/images/obsid{obs_id}-instM1-subexpALL-en{lo_en}_{hi_en}keV-image.fits
mos2_image = {obs_id}/images/obsid{obs_id}-instM2-subexpALL-en{lo_en}_{hi_en}keV-image.fits
pn_expmap = {obs_id}/images/obsid{obs_id}-instPN-subexpALL-en{lo_en}_{hi_en}keV-expmap.fits
mos1_expmap = {obs_id}/images/obsid{obs_id}-instM1-subexpALL-en{lo_en}_{hi_en}keV-expmap.fits
mos2_expmap = {obs_id}/images/obsid{obs_id}-instM2-subexpALL-en{lo_en}_{hi_en}keV-expmap.fits
region_file = /data/regions/xmm/{obs_id}.reg

[CHANDRA_FILES]
root_chandra_dir = /data/daxa_archives/my_survey/processed_data/chandra/
clean_acis_evts = {obs_id}/events/obsid{obs_id}-instACIS-subexpE001-en-cleanevents.fits
attitude_file = {obs_id}/misc/obsid{obs_id}-instACIS-subexpE001-aspectsolution.fits
acis_badpix_file = {obs_id}/misc/obsid{obs_id}-instACIS-subexpE001-badpix.fits
acis_mask_file = {obs_id}/misc/obsid{obs_id}-instACIS-subexpE001-mask.fits
lo_en = ['0.5', '2.0']
hi_en = ['2.0', '7.0']
acis_image = {obs_id}/images/obsid{obs_id}-instACIS-subexpE001-en{lo_en}_{hi_en}keV-image.fits
acis_expmap = {obs_id}/images/obsid{obs_id}-instACIS-subexpE001-en{lo_en}_{hi_en}keV-expmap.fits
region_file = /data/regions/chandra/{obs_id}.reg

[ERASS_FILES]
root_erass_dir = /data/daxa_archives/my_survey/processed_data/erosita_all_sky_de_dr1/
clean_erass_evts = {obs_id}/events/obsid{obs_id}-instTM1_TM2_TM3_TM4_TM5_TM6_TM7-subexpALL-en0.2_10.0keV-finalevents.fits
lo_en = ['0.5']
hi_en = ['2.0']
erass_image = /this/is/optional/
erass_expmap = /this/is/optional/
region_file = /data/regions/erass/{obs_id}.reg

[EROSITA_FILES]
root_erosita_dir = /this/is/required/erosita_obs/data/
clean_erosita_evts = /this/is/required/{obs_id}/{obs_id}.fits
lo_en = ['0.50', '2.00']
hi_en = ['2.00', '10.00']
erosita_image = /this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV_img.fits
erosita_expmap = /this/is/optional/{obs_id}/{obs_id}-{lo_en}-{hi_en}keV_expmap.fits
region_file = /this/is/optional/erosita_obs/regions/{obs_id}/regions.reg
```
:::
::::

::::{tab-item} Remote S3 / HTTPS Dataset (`xga.cfg`)
:::{div}
:class: placeholder-text
This example demonstrates pointing `root_<tel>_dir` at an open-access `s3://` bucket or `https://` archive URL:

```ini
[XGA_SETUP]
xga_save_path = xga_output
num_cores = auto

[XMM_FILES]
root_xmm_dir = s3://nasa-heasarc/xmm/data/rev0/
clean_pn_evts = {obs_id}/PPS/P{obs_id}PNU002PIEVLI0000.FTZ
clean_mos1_evts = {obs_id}/PPS/P{obs_id}M1U002MIEVLI0000.FTZ
clean_mos2_evts = {obs_id}/PPS/P{obs_id}M2U002MIEVLI0000.FTZ
attitude_file = {obs_id}/PPS/P{obs_id}OBX000ATTTSR0000.FTZ
lo_en = ['0.50', '2.00']
hi_en = ['2.00', '10.00']
pn_image = /this/is/optional/
mos1_image = /this/is/optional/
mos2_image = /this/is/optional/
pn_expmap = /this/is/optional/
mos1_expmap = /this/is/optional/
mos2_expmap = /this/is/optional/
region_file = /this/is/optional/
```
:::
::::

::::{tab-item} Classic Local Layout (`docs/example_config/xga.cfg`)
```ini
[XGA_SETUP]
xga_save_path = xga_output
num_cores = 4

[XMM_FILES]
root_xmm_dir = /home/dt237/apollo_mnt/xmm_obs/data/
clean_pn_evts = {obs_id}/eclean/pn_exp1_clean_evts.fits
clean_mos1_evts = {obs_id}/eclean/mos1_exp1_clean_evts.fits
clean_mos2_evts = {obs_id}/eclean/mos2_exp1_clean_evts.fits
attitude_file = {obs_id}/epchain/P{obs_id}OBX000ATTTSR0000.FIT
lo_en = ['0.50', '2.00']
hi_en = ['2.00', '10.00']
pn_image = {obs_id}/images/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_img.fits
mos1_image = {obs_id}/images/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_img.fits
mos2_image = {obs_id}/images/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_img.fits
pn_expmap = {obs_id}/images/{obs_id}-{lo_en}-{hi_en}keV-pn_merged_expmap.fits
mos1_expmap = {obs_id}/images/{obs_id}-{lo_en}-{hi_en}keV-mos1_merged_expmap.fits
mos2_expmap = {obs_id}/images/{obs_id}-{lo_en}-{hi_en}keV-mos2_merged_expmap.fits
region_file = /home/dt237/apollo_mnt/xmm_obs/code/xapa/id_results/{obs_id}/final_class_regions_REDO.reg

[EROSITA_FILES]
root_erosita_dir = /home/dt237/apollo_mnt/erosita_obs/data/
clean_erosita_evts = {obs_id}/clean_evts.fits
lo_en = ['0.5']
hi_en = ['2.0']
erosita_image = {obs_id}/images/{obs_id}_combined_{lo_en}-{hi_en}keVimg.fits
erosita_expmap = {obs_id}/images/{obs_id}_combined_{lo_en}-{hi_en}keVexpmap.fits
region_file = /home/dt237/apollo_mnt/erosita_obs/regions/{obs_id}.reg
```
::::

:::::
