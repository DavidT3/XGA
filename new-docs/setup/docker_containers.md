---
title: Running XGA in a Docker Container
description: Running XGA locally or on HPC clusters inside the pre-built NASA Fornax high-energy Docker images
site:
  hide_authors: true
---

# Running XGA in Fornax Containers

:::::{div}
:class: placeholder-text

```{note} Placeholder — Work in Progress (Issue #1630)
This page is a placeholder for the upcoming guide on running XGA inside local **Fornax Docker (`fornax-hea`) containers** and will be expanded as part of [Issue #1630](https://github.com/DavidT3/XGA/issues/1630).
```

If you want all of XGA's [external mission dependencies](external_software.md) (**HEASoft/XSPEC**, **XMM-SAS**, **Chandra CIAO**, and **eSASS**) working together without having to compile them locally or work around Conda environment conflicts between CIAO and HEASoft, you can run XGA inside the official **Fornax High-Energy (`fornax-hea`)** container image.

## Planned Topics for This Page

::::{tab-set}

:::{tab-item} Docker (Laptop / Desktop)
How to pull the `fornax-hea` image with Docker and bind-mount:
* Your local observation archives (`root_<tel>_dir`)
* Local calibration directories (`CALDB` and `SAS_CCFPATH`)
* Your XGA configuration directory (`XGA_CONFIG_DIR`) and output directory (`xga_save_path`)

```bash
# Example placeholder command (to be finalized in #1630)
docker run -it --rm \
  -v /path/to/local/archives:/data:ro \
  -v /path/to/local/caldb:/caldb:ro \
  -v /path/to/xga_config:/config \
  -v $(pwd):/workspace \
  -e XGA_CONFIG_DIR=/config \
  ghcr.io/nasa-fornax/fornax-images/fornax-hea:latest
```
:::

:::{tab-item} Apptainer / Singularity (HPC Clusters)
How to pull `fornax-hea` as an Apptainer `.sif` image on shared institutional HPC systems where Docker root daemons are unavailable, and bind-mount scratch/archive filesystems for batch XGA jobs:

```bash
# Example placeholder command (to be finalized in #1630)
apptainer pull fornax-hea.sif docker://ghcr.io/nasa-fornax/fornax-images/fornax-hea:latest
apptainer exec \
  --bind /scratch/archives:/data,/scratch/caldb:/caldb \
  --env XGA_CONFIG_DIR=/scratch/xga_config \
  fornax-hea.sif python my_xga_pipeline.py
```
:::

::::

:::::
