#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 25/02/2021, 10:51. Copyright (c) David J Turner

# This script is designed to perform a spectra cleaning step where every spectrum is fit separately, and only those
#  which produce results that meet certain criteria will be used in the simultaneous final fit.
# This whole lot of XSPEC and TCL codes aren't particularly well programmed, as I'm not very familiar with TCL,
#  but they should at least work

# Things like the cosmology and fit statistic that XSPEC has already had set in the fit xcm will carry over into
#  this function, so I don't need to reset them, but variables need to be passed in

# Set up the function to be called by the fitting script
# The expected arguments are:
# 1. List of spectrum file paths
# 2. List of parameter names to perform checks on
# 3. A list of lists, with each element corresponding to a parameter that is being checked. For a given
#  parameter the first element should be a lower limit, the second an upper limit, and the third the maximum allowed
#  uncertainty.
# 4. Model name
# 5. Start values for model parameters
# 6. Model parameter names
# 7. A freeze list for the parameters, which should not be allowed to vary in the fits
# 8. Delta fit statistic for error command
proc spec_check {args} {
    set prompt "XSPEC12>"
    # Here we split out the arguments into individual variables, largely to make it easier for me to read

    puts $args


}