#  This code is a part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (david.turner@sussex.ac.uk) 15/06/2020, 15:37. Copyright (c) David J Turner

# This is a convenience script that shouldn't really be needed by anyone but me, it takes a list of XSPEC models,
#  fetches their parameters, and saves them to a file.

# Open the list of models
set fp [open "../files/add_model_list.txt" r]
set file_data [read $fp]
close $fp
# Splits it into lines
set add_data [split $file_data "\n"]

# Open the list of models
set fp [open "../files/mult_model_list.txt" r]
set file_data [read $fp]
close $fp
# Splits it into lines
set mult_data [split $file_data "\n"]

set par_name_file [open "../files/xspec_model_pars.json5" w]
puts $par_name_file "{"
set par_unit_file [open "../files/xspec_model_units.json5" w]
puts $par_unit_file "{"


foreach m $add_data {
    set par_names \[
    set par_units \[

    model $m
    /*
    set num_pars [tcloutr modpar]

    for {set i 1} {$i < $num_pars+1} {incr i} {
        if {$i == 1} {
            append par_names \"[lindex [tcloutr pinfo $i] 0]\"
            append par_units \"[lindex [tcloutr pinfo $i] 1]\"
        } else {
            append par_names , " " \"[lindex [tcloutr pinfo $i] 0]\"
            append par_units , " " \"[lindex [tcloutr pinfo $i] 1]\"
        }
    }

append par_names \]
append par_units \]
set name \"$m\":
set name_line [concat $name " " $par_names,\n]
set unit_line [concat $name " " $par_units,\n]
puts $par_name_file $name_line
puts $par_unit_file $unit_line
}

# foreach m $mult_data {
for {set mi 0} {$mi < [llength $mult_data]-1} {incr mi} {
    set m [lindex $mult_data $mi]
    set par_names \[
    set par_units \[

    model $m*bbody
    /*
    set num_pars [tcloutr modpar]

    for {set i 1} {$i < $num_pars-1} {incr i} {
        if {$i == 1} {
            append par_names \"[lindex [tcloutr pinfo $i] 0]\"
            append par_units \"[lindex [tcloutr pinfo $i] 1]\"
        } else {
            append par_names , " " \"[lindex [tcloutr pinfo $i] 0]\"
            append par_units , " " \"[lindex [tcloutr pinfo $i] 1]\"
        }
    }

append par_names \]
append par_units \]
set name \"$m\":
set name_line [concat $name " " $par_names,\n]
set unit_line [concat $name " " $par_units,\n]
puts $par_name_file $name_line
puts $par_unit_file $unit_line
}

set m [lindex $mult_data [expr {[llength $mult_data]-1}]]
set par_names \[
set par_units \[

model $m*bbody
/*
set num_pars [tcloutr modpar]

for {set i 1} {$i < $num_pars-1} {incr i} {
    if {$i == 1} {
        append par_names \"[lindex [tcloutr pinfo $i] 0]\"
        append par_units \"[lindex [tcloutr pinfo $i] 1]\"
    } else {
        append par_names , " " \"[lindex [tcloutr pinfo $i] 0]\"
        append par_units , " " \"[lindex [tcloutr pinfo $i] 1]\"
    }
}

append par_names \]
append par_units \]
set name \"$m\":
set name_line [concat $name " " $par_names\n]
set unit_line [concat $name " " $par_units\n]
puts $par_name_file $name_line
puts $par_unit_file $unit_line

puts $par_name_file "}"
close $par_name_file
puts $par_unit_file "}"
close $par_unit_file
exit