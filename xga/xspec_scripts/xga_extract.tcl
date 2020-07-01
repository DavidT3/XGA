#  This code is a part of XMM: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/06/2020, 15:37. Copyright (c) David J Turner

# This script writes a LOT of information about loaded spectra and current fit results to a fits file
# Inspired by the writefits script packaged with XSPEC, it will also calculate and write the current UNABSORBED
#  luminosities to the file for input energy ranges.

proc xga_extract { args } {
# writes a bunch of information to a single line of a FITS file
    set prompt "XSPEC12>"

# parse the arguments and find the name of the out file
# The first argument should be the prefix for the names of the output files, the second is a list of energy limit
#  pairs to calculate luminosity in, the third is the relevant redshift, the fourth is
#  the confidence level for Lx errors, and the last is the name of the model used to fit
   set fileprefix [lindex $args 0]

# Parse the arguments related to luminosity calculations
   set energy_lims [lindex $args 1]
   set redshift [lindex $args 2]
   set conf [lindex $args 3]
   set mod_name [lindex $args 4]

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

# I used to do this like writefits and make a fits file with command line tools from text files,
#  but thanks to an absolutely bizarre error when running from Jupyter Notebooks I can't do that
# Name all the columns I am adding manually to the fit result table
set col_list "MODEL,TOTAL_EXPOSURE,TOTAL_COUNT_RATE,TOTAL_COUNT_RATE_ERR,NUM_UNLINKED_THAWED_VARS,FIT_STATISTIC,TEST_STATISTIC,DOF"

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
            set divid "|"
            append col_list "," $pname$divid$ipar
            append col_list "," $pname$divid$ipar-
            append col_list "," $pname$divid$ipar+
            incr count
	    }
    }

# open a text version of the output file - for the actual values to be written to.
    set txtfile $fileprefix
    append txtfile "_results.csv"
    rm -f $txtfile
    set fileid [open $txtfile w]

# Write the text output
# These are the values that will always go in this table, the next chunk is more dynamic and depends how many pars
#  there are.
    set comma ,
    set outstr "$mod_name$comma$total_time$comma$total_rate$comma$total_rate_err$comma$count$comma$fit_stat$comma$test_stat$comma$dof"
    for {set ipar 1} {$ipar <= $numpar} {incr ipar} {
	if { $spardel($ipar) > 0 } {
# Write the parameter value and errors to the outstring here, the errors ARE NOT written as confidence levels, but
#  have been converted to +- values already
	    append outstr "$comma$sparval($ipar)$comma[expr {$sparval($ipar)-$sparerrlow($ipar)}]$comma[expr {$sparerrhi($ipar)-$sparval($ipar)}]"
	    }
    }
    puts $fileid $col_list
    puts $fileid $outstr

# close the output text file
    close $fileid





# This chunk will write plotting data for the spectra to separate csv files
#  IT HAS TO GO BEFORE LUM CALC OTHERWISE THE MODEL DATA READ OUT WILL BE THE MODEL WITH NH=0

# Column names for the plot tables, Y will be from data, YMODEL will be from the fitted model
        set col_list "X,Y,XERR,YERR,YMODEL"

    for {set spec_i 1} {$spec_i < $num_spec+1} {incr spec_i} {
        set plot_data $fileprefix
        append plot_data "_spec" $spec_i ".csv"
        rm -f $plot_data
        set fileid [open $plot_data w]

# Reading out the plotting data I'm going to store for a particular loaded spectrum
        set xarr [tcloutr plot data x $spec_i]
        set yarr [tcloutr plot data y $spec_i]
        set xerrarr [tcloutr plot data xerr $spec_i]
        set yerrarr [tcloutr plot data yerr $spec_i]
        set modelarr [tcloutr plot data model $spec_i]

# write the text output
        set outstr ""
        for {set k 0} {$k < [llength $xarr]} {incr k} {
            append outstr "[lindex $xarr $k]$comma[lindex $yarr $k]$comma[lindex $xerrarr $k]$comma[lindex $yerrarr $k]$comma[lindex $modelarr $k]\n"
            }
# Write to the actual file
        puts $fileid $col_list
        puts $fileid $outstr
# close the output text file
        close $fileid
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

# This chunk will write some information about the separate spectra to another csv
    set col_list "SPEC_PATH,EXPOSURE,COUNT_RATE,COUNT_RATE_ERR"

    foreach p $energy_lims {
        set lum_name "Lx"
        set lum_min "Lx"
        set lum_max "Lx"
        append lum_name "_" [lindex $p 0] "_" [lindex $p 1]
        append lum_min "_" [lindex $p 0] "_" [lindex $p 1] "-"
        append lum_max "_" [lindex $p 0] "_" [lindex $p 1] "+"
        append col_list "," $lum_name
        append col_list "," $lum_min
        append col_list "," $lum_max
    }

# open a text version of the output file
    set spec_info_file $fileprefix
    append spec_info_file "_info.csv"
    rm -f $spec_info_file
    set fileid [open $spec_info_file w]

# write the text output
    set outstr ""
    for {set spec_i 1} {$spec_i < $num_spec+1} {incr spec_i} {
        append outstr "[tcloutr filename $spec_i]$comma[lindex $times $spec_i-1]$comma[lindex $rates $spec_i-1]$comma[lindex $rates_errs $spec_i-1]"
        for {set li 0} {$li < [llength $lums]} {incr li} {
            append outstr "$comma[lindex [lindex $lums $li] $spec_i-1]$comma[lindex [lindex $lum_min_err $li] $spec_i-1]$comma[lindex [lindex $lum_max_err $li] $spec_i-1]"
            }
        append outstr "\n"
        }
# Write to the actual file
    puts $fileid $col_list
    puts $fileid $outstr
# close the output text file
    close $fileid

}
