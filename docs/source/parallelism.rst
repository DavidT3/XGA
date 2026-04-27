A note on XGA's parallelism
===========================

In every many functions built into XGA, you will find a
**num_cores** keyword argument which tells the function how many cores on your local machine it is allowed to use. The
default value is 90% of the available cores on your system, though you are of course free to set your own value when
you call the functions.

To see the number of cores which have automatically allocated to XGA, you can import the NUM_CORES constant from the
base xga module. You can also manually set this value globally, before running anything. Either set the
*num_cores* option in the [XGA_SETUP] section of the configuration file (setting it to **auto** will
instruct XGA to use 90% of available cores), or simply set the NUM_CORES constant
imported from xga.

.. code-block:: python

    from xga import NUM_CORES
    # You can override the global default like this
    import xga
    xga.NUM_CORES = 4

Parallel Observation Census
---------------------------
Building the observation census (reading headers from thousands of FITS files) is one of the most I/O
intensive parts of XGA's setup. This process is now fully parallelized and will utilize the global
**NUM_CORES** value to speed up both the initial setup and any subsequent updates or manual rebuilds.