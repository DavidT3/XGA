A note on XGA's parallelism
===========================

In every many functions built into XGA, you will find a
**num_cores** keyword argument which tells the function how many cores on your local machine it is allowed to use. The
default value is 90% of the available cores on your system, though you are of course free to set your own value when
you call the functions.

To see the number of cores which have automatically allocated to XGA, you can import the NUM_CORES constant from the
base xga module. You can also manually set this value globally, before running anything. Either set the
*num_cores* option in the [XGA_SETUP] section of the configuration file, or simply set the NUM_CORES constant
imported from xga.