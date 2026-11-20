---
title: Observation Census & Blacklisting
description: How XGA builds and maintains per-mission observation censuses and blacklists
site:
  hide_authors: true
---

# Observation Census & Blacklisting

## XGA's First Run After Configuration

The first time you access any configuration-dependent part of XGA, it will create an 'observation census', where it will search through all the observations it can find (based on your entries in the configuration file), check that there are events lists present, and record the pointing RA and DEC [(along with which instruments are usable, excluding observations taken with banned filters like `CalClosed`/`Closed` or banned observation modes like XMM `SLEW`)]{.edited-text}. *This can take a while*, but will only take that long on the first run. This process is fully parallelized and will utilize your available CPU cores to speed up the filesystem [(or remote S3/HTTPS header)]{.edited-text} I/O.

The module will check the census against your observation directory and see if it needs to be updated on every run.

:::{div}
:class: placeholder-text
### Where Census Files Are Stored

For each configured telescope (`xmm`, `chandra`, `erosita`, `erass`), XGA stores its census CSV inside a mission subdirectory of your configuration directory (`xga.CONFIG_PATH`, which defaults to `~/.config/xga/`):

```text
~/.config/xga/
├── xga.cfg
├── xmm/
│   ├── xmm_census.csv
│   └── xmm_blacklist.csv
├── chandra/
│   ├── chandra_census.csv
│   └── chandra_blacklist.csv
└── erass/
    ├── erass_census.csv
    └── erass_blacklist.csv
```

You can inspect the active census DataFrames in Python via `xga.CENSUS` (and their file paths via `xga.CENSUS_FILES`):

```python
import xga

# View the XMM-Newton observation census DataFrame
print(xga.CENSUS["xmm"].head())
```
:::

---

## Census Management (`rebuild_census`)

XGA provides the `rebuild_census()` function for manual management of the observation census. This is useful if you have deleted data from your disk [or added new observations mid-session]{.edited-text} or want to force a fresh scan of your archives.

* **Standard Update**: Calling `rebuild_census()` without arguments is equivalent to the update check performed at startup.
* **Full Rebuild**: `rebuild_census(full_rebuild=True)` deletes the existing census and parses every FITS header from scratch.
* **Cleanup**: `rebuild_census(clean_dead=True)` checks the current census against the filesystem and removes entries for {term}`ObsID`s that no longer exist.
* [**Target Specific Telescopes**: Pass `telescope='xmm'` or `telescope=['xmm', 'chandra']` to only rebuild the census for specific missions.]{.edited-text}

```python
import xga

# Prune census entries for data that has been deleted from disk
xga.rebuild_census(clean_dead=True)
```

---

## Blacklisting ObsIDs & Instruments

If you don't wish your analyses to include certain {term}`ObsID`s [(or specific instruments of an ObsID)]{.edited-text}, then you can 'blacklist' them and remove them from all consideration; you simply need to add the ObsID to [`<telescope>/<telescope>_blacklist.csv` (for example `~/.config/xga/xmm/xmm_blacklist.csv`),]{.edited-text} which is located in a mission subfolder of the configuration directory. If you need to know where these files are located, import [`BLACKLIST_FILES` (or `CONFIG_PATH`) from `xga.utils`]{.edited-text}.

It is possible that you might want to do this so that ObsIDs with significant problems (flaring, for instance), don't contribute to and spoil your current analysis.

:::{div}
:class: placeholder-text
### Per-Instrument Blacklist Format

Each mission's `<telescope>_blacklist.csv` contains an `ObsID` column followed by an `EXCLUDE_<INST>` boolean column (`T` or `F`) for every instrument on that telescope. Setting `T` excludes that instrument for that ObsID, while setting `F` allows the remaining instruments to be used:

::::{tab-set}

:::{tab-item} XMM-Newton (`xmm/xmm_blacklist.csv`)
```csv
ObsID,EXCLUDE_PN,EXCLUDE_MOS1,EXCLUDE_MOS2
0112230301,T,T,T
0201903501,T,F,F
```
In the example above, all EPIC cameras are excluded for `0112230301`, whereas only `pn` is excluded for `0201903501` (allowing `mos1` and `mos2` to still be used).
:::

:::{tab-item} Chandra (`chandra/chandra_blacklist.csv`)
```csv
ObsID,EXCLUDE_ACIS
1234,T
```
:::

:::{tab-item} eROSITA / eRASS (`erass/erass_blacklist.csv`)
```csv
ObsID,EXCLUDE_TM1,EXCLUDE_TM2,EXCLUDE_TM3,EXCLUDE_TM4,EXCLUDE_TM5,EXCLUDE_TM6,EXCLUDE_TM7
111099,T,T,T,T,T,T,T
```
:::

::::
:::
