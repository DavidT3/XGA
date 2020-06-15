#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 04/05/2020, 12:18. Copyright (c) David J Turner

# This script writes a LOT of information about loaded spectra and current fit results to a fits file
# Inspired by the writefits script packaged with XSPEC, it will also calculate and write the current UNABSORBED
#  luminosities to the file for input energy ranges.

proc xga_extract { args } {
# writes a bunch of information to a single line of a FITS file
    set prompt "XSPEC12>"

# parse the arguments and find the name of the out file
# The first argument should be the name of the output file, the second is a list of energy limit
#  pairs to calculate luminosity in, the third is the relevant redshift, and the last is
#  the confidence level for Lx errors
   set FITSfile [lindex $args 0]

# Parse the arguments related to luminosity calculations
   set energy_lims [lindex $args 1]
   set redshift [lindex $args 2]
   set conf [lindex $args 3]

# find how many spectra are loaded in - need this for some loops
    set num_spec [tcloutr datasets]

# Get the exposure times for all the different spectra, calculate a total
# Define lists for storing times and rates read out for all the different spectra
    set total_time 0
    set times {}
    set rates {}
    set rates_errs {}
# Cycle through all the spectra, XSPEC spec indexing starts at 1, whereas TCL indexing starts at 0.
    for {set i 1} {$i < $num_spec+1} {incr i} {
        set time [tcloutr expos $i s]
        set rate [tcloutr rate $i]
        lappend times $time
        lappend rates [lindex $rate 0]
        lappend rates_errs [lindex $rate 1]
        set total_time [expr {$total_time + $time}]
    }

# Reading out the total rate and total rate error
    set total_rate [lindex [tcloutr rate all] 0]
    set total_rate_err [lindex [tcloutr rate all] 1]


# we will need to know the number of parameters - I think?
    set numpar [tcloutr modpar]

# get the parameter information
    for {set ipar 1} {$ipar <= $numpar} {incr ipar} {
	if {[scan [tcloutr param $ipar] "%f %f" tmp1 tmp2] == 2 && [tcloutr plink $ipar] == "F"} {
            set sparval($ipar) $tmp1
            set spardel($ipar) $tmp2
	    scan [tcloutr error $ipar] "%f %f" sparerrlow($ipar) sparerrhi($ipar)
        } elseif {[scan [tcloutr param $ipar] "%f %f" tmp1 tmp2] != 2 && [tcloutr plink $ipar] == "F"} {
            set sparval($ipar) $tmp1
            set spardel($ipar) -1
        } else {
            set sparval($ipar) 0
            set spardel($ipar) -1
        }
    }

# This next chunk writes the global values; fit results, total exposure, DoF etc.
# Get the fit and test statistic values, as well as numbers of DOF
    set fit_stat [tcloutr stat]
    set test_stat [tcloutr stat test]
    set dof [lindex [tcloutr dof] 0]

# create a temporary file for the column descriptors of the fit results table
    set cdfile $FITSfile
    append cdfile "-fitcd"
    rm -f $cdfile
    set fileid [open $cdfile w]

# Name all the columns I am adding manually to the fit result table
    puts $fileid "TOTAL_EXPOSURE E"
    puts $fileid "TOTAL_COUNT_RATE E"
    puts $fileid "TOTAL_COUNT_RATE_ERR E"
    puts $fileid "NUM_UNLINKED_THAWED_VARS K"
    puts $fileid "FIT_STATISTIC E"
    puts $fileid "TEST_STATISTIC E"
    puts $fileid "DOF K"

# Now all the relevant parameter columns get named (those that are allowed to vary and are unlinked)
# Also record where-ever there is a parameter called nH, so we know which parameters to 0 later for unabsorbed
#  luminosity calculations
    set nh_pars {}
    set count 0
    for {set ipar 1} {$ipar <= [array size spardel]} {incr ipar} {
	    set punit " "
	    scan [tcloutr pinfo $ipar] "%s %s" pname punit
	    if {$pname == "nH"} {
	        lappend nh_pars $ipar
	        }
	    lappend idents $ipar
        if { $spardel($ipar) > 0 } {
# Each parameter gets three columns; the value, the -error, and the +error
            puts $fileid [concat $pname$ipar " E " $punit]
            puts $fileid [concat E$pname$ipar- " E " $punit]
            puts $fileid [concat E$pname$ipar+ " E " $punit]
            incr count
	    }
    }
    close $fileid

# open a text version of the output file - for the actual values to be written to.
    set txtfile $FITSfile
    append txtfile "-fittxt"
    rm -f $txtfile
    set fileid [open $txtfile w]

# Write the text output
# These are the values that will always go in this table, the next chunk is more dynamic and depends how many pars
#  there are.
    set outstr "$total_time $total_rate $total_rate_err $count $fit_stat $test_stat $dof "
    for {set ipar 1} {$ipar <= $numpar} {incr ipar} {
	if { $spardel($ipar) > 0 } {
# Write the parameter value and errors to the outstring here, the errors ARE NOT written as confidence levels, but
#  have been converted to +- values already
	    append outstr "$sparval($ipar) [expr {$sparval($ipar)-$sparerrlow($ipar)}] [expr {$sparerrhi($ipar)-$sparval($ipar)}] "
	    }
    }
    puts $fileid $outstr

# close the output text file
    close $fileid

# Turn those text files into the first fits file, with global values
    ftcreate extname="results" cdfile=$cdfile datafile=$txtfile outfile=$FITSfile clobber="yes"
# Clean up temporary files
    rm -f $cdfile $txtfile




# This chunk will write plotting data for the spectra to separate fits files and add them to the main
#  IT HAS TO GO BEFORE LUM CALC OTHERWISE THE MODEL DATA READ OUT WILL BE THE MODEL WITH NH=0
    for {set spec_i 1} {$spec_i < $num_spec+1} {incr spec_i} {
        set plotcd $FITSfile
        append plotcd "-plotcd" $spec_i
        rm -f $plotcd
        set fileid [open $plotcd w]

# Column names for the plot tables, Y will be from data, YMODEL will be from the fitted model
        puts $fileid "X E"
        puts $fileid "Y E"
        puts $fileid "XERR E"
        puts $fileid "YERR E"
        puts $fileid "YMODEL E"
        close $fileid

        set plottxt $FITSfile
        append plottxt "-plottxt" $spec_i
        rm -f $plottxt
        set fileid [open $plottxt w]

# Reading out the plotting data I'm going to store for a particular loaded spectrum
        set xarr [tcloutr plot data x $spec_i]
        set yarr [tcloutr plot data y $spec_i]
        set xerrarr [tcloutr plot data xerr $spec_i]
        set yerrarr [tcloutr plot data yerr $spec_i]
        set modelarr [tcloutr plot data model $spec_i]

# write the text output
        set outstr ""
        for {set k 0} {$k < [llength $xarr]} {incr k} {
            append outstr "[lindex $xarr $k] [lindex $yarr $k] [lindex $xerrarr $k] [lindex $yerrarr $k] [lindex $modelarr $k] \n "
            }
# Write to the actual file
        puts $fileid $outstr
# close the output text file
        close $fileid

# Turn those text files into the first fits file, with global values
        set specout $FITSfile
        append specout "-plot" $spec_i
        set tab_name "plot"
        append tab_name $spec_i
        ftcreate extname=$tab_name cdfile=$plotcd datafile=$plottxt outfile=$specout clobber="yes"
# Append spec fits file to results
        ftappend infile=$specout outfile=$FITSfile
# Clean up temporary files
        rm -f $plotcd $plottxt $specout
    }




# Here I shall calculate luminosities for later writing to the file - first the errors
    set lum_min_err {}
    set lum_max_err {}
    foreach p $energy_lims {
        set lv [lindex $p 0]
        set uv [lindex $p 1]
        # Initial luminosity calculation with uncertainties turned on
        lumin $lv $uv $redshift err,,$conf
        set interim_lum_min {}
        set interim_lum_max {}

        for {set t 1} {$t < $num_spec+1} {incr t} {
            set lmine [tcloutr lumin $t]
            lappend interim_lum_min [expr {([lindex $lmine 0] - [lindex $lmine 1])}]
            #  * pow(10, 44)
            lappend interim_lum_max [expr {([lindex $lmine 2] - [lindex $lmine 0])}]
            #  * pow(10, 44)
            }

        lappend lum_min_err $interim_lum_min
        lappend lum_max_err $interim_lum_max
        }

    set lums {}
    # Then we do the trick to measure unabsorbed luminosities, zero the nh and measure luminosity again
    foreach nhi $nh_pars {
        newpar $nhi 0.0
        }
    # Then run through and calculate luminosities again
    foreach p $energy_lims {
        set lv [lindex $p 0]
        set uv [lindex $p 1]
        # Initial luminosity calculation with uncertainties turned on
        lumin $lv $uv $redshift err,,$conf
        set interim_lum {}
        for {set t 1} {$t < $num_spec+1} {incr t} {
            set lmine [tcloutr lumin $t]
            lappend interim_lum [expr {[lindex $lmine 0]}]
            #  * pow(10, 44)
            }
        lappend lums $interim_lum
        }

# This chunk will write some information about the separate spectra to another fits file
# create a temporary file for the column descriptors of the spectrum information table
    set speccdfile $FITSfile
    append speccdfile "-speccd"
    rm -f $speccdfile
    set fileid [open $speccdfile w]

    puts $fileid "EXPOSURE E"
    puts $fileid "COUNT_RATE E"
    puts $fileid "COUNT_RATE_ERR E"
    foreach p $energy_lims {
        set lum_name "Lx"
        set lum_min "Lx"
        set lum_max "Lx"
        append lum_name "_" [lindex $p 0] "_" [lindex $p 1] " E 10^44erg/s"
        append lum_min "_" [lindex $p 0] "_" [lindex $p 1] "- E 10^44erg/s"
        append lum_max "_" [lindex $p 0] "_" [lindex $p 1] "+ E 10^44erg/s"
        puts $fileid $lum_name
        puts $fileid $lum_min
        puts $fileid $lum_max
    }
    close $fileid

# open a text version of the output file
    set spectxtfile $FITSfile
    append spectxtfile "-spectxt"
    rm -f $spectxtfile
    set fileid [open $spectxtfile w]

# write the text output
    set outstr ""
    for {set spec_i 1} {$spec_i < $num_spec+1} {incr spec_i} {
        append outstr "[lindex $times $spec_i-1] [lindex $rates $spec_i-1] [lindex $rates_errs $spec_i-1] "
        for {set li 0} {$li < [llength $lums]} {incr li} {
            append outstr "[lindex [lindex $lums $li] $spec_i-1] [lindex [lindex $lum_min_err $li] $spec_i-1] [lindex [lindex $lum_max_err $li] $spec_i-1] "
            }
        append outstr "\n"
        }
# Write to the actual file
    puts $fileid $outstr
# close the output text file
    close $fileid

# Turn those text files into the first fits file, with global values
    set specout $FITSfile
    append specout "-spec"
    ftcreate extname="spec_info" cdfile=$speccdfile datafile=$spectxtfile outfile=$specout clobber="yes"
# Append spec fits file to results
    ftappend infile=$specout outfile=$FITSfile
# Clean up temporary files
    rm -f $speccdfile $spectxtfile $specout

}
