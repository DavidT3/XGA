---
title: Using XGA on NASA Fornax
description: Running XGA directly on the NASA Fornax Science Console with pre-installed high-energy software and cloud archives
site:
  hide_authors: true
---

# Using XGA on NASA Fornax

:::::{div}
:class: placeholder-text

```{note} Placeholder — Work in Progress (Issue #1630)
This page is a placeholder for the upcoming guide on using XGA directly on the **NASA Fornax Science Console** and will be expanded as part of [Issue #1630](https://github.com/DavidT3/XGA/issues/1630).
```

The **NASA Fornax** initiative provides a cloud-based Science Console with pre-configured high-energy astrophysics software environments (**HEASoft/XSPEC**, **XMM-SAS**, **Chandra CIAO**, and **eSASS** — see [External Mission Software](external_software.md)), pre-mounted calibration databases (`CALDB` and `SAS_CCFPATH`), and direct low-latency access to public HEASARC cloud data buckets.

## Planned Topics for This Page

::::{grid} 1 1 2 2

:::{card}
:header: **1. Selecting the High-Energy Environments**
Choosing the appropriate Fornax High-Energy Astrophysics environment.
:::

:::{card}
:header: **2. Configuring XGA on Fornax**
Setting up `~/.config/xga/xga.cfg` (or `XGA_CONFIG_DIR` on persistent user storage) to point either at locally reduced DAXA archives on your Fornax persistent volume or directly at open-access `s3://` buckets.
:::

::::

### Quick Preview: ..


```bash
conda
```
:::::
