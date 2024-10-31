#! /usr/bin/bash

busybox killall soc_to_mcu
busybox killall sensor_trans
busybox killall isomeipd_ics


export LD_LIBRARY_PATH=lib:$LD_LIBRARY_PATH
export NLOG=STDOUT
export IAUTOSAR_TMP=/data/tmp comdd_iautosar
chmod 775 bin/isomeipd_ics
bin/isomeipd_ics &
test/sensor_trans/bin/sensor_trans &
bin/soc_to_mcu 2 &