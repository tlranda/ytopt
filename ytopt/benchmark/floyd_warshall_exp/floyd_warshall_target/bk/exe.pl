#!/usr/bin/env perl

#Author: Xingfu Wu
#MCS, ANL
# exe.pl: average the execution time in 5 runs
#
use Time::HiRes qw(gettimeofday); 

foreach $filename (@ARGV) {
 #  print "Start to preprocess ", $filename, "...\n";
   $ssum = 0.0;
   @nn = (1..5);
   for(@nn) {
    $retval = gettimeofday( ); 
    system("$filename >/dev/null 2>&1");
    $tt = gettimeofday( );
    $ttotal = $tt - $retval;
    $ssum = $ssum + $ttotal;
   }
   $avg = $ssum / 5;
 #  print "End to preprocess ", $avg, "...\n";
   printf("%.3f", $avg);
}
