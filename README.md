## ALL VERY OUTLINE-SKETCHY ATM

Installation

You're going to have to fill out the config file after installation, to point at all your XMM files

Certain entries are absolutely required, some are optional, some are dynamic
Required (anywhere where an obsid would go, put {obs_id} instead):
* Clean PN events file
* Clean MOS1 events file
* Clean MOS2 events file

Optional:
* CCF

Dynamic and optional (fill out the lo_en and hi_en headers to define the energy ranges available):
* Energy ranged images (in file names replace the lower energy number with {lo_en} and the higher with {hi_en})
* Energy ranged expmaps (same as above)

## XGA assumes that all your XMM obsid folders live in the same directory

## XGA assumes that region files are in a DS9 format


## Made a design decision that XGA will expect region files in the XCS format
* DS9 standard
* With XCS colours - green = extended, red = point etc.