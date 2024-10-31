#!/bin/bash

cur_dir=$(pwd)

if [[ ${BASH_SOURCE} == /* ]]; then
    shell_root="${BASH_SOURCE}"
else
    shell_root="${cur_dir}/${BASH_SOURCE}"
fi

shell_root="${shell_root%/scripts/start_someip.sh}"

source ${shell_root}/scripts/env_setup.sh

# busybox killall sensor_trans
# busybox killall soc_to_mcu

# ${shell_root}/bin/sensor_trans &
# ${shell_root}/bin/soc_to_mcu &
