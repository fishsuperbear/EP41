#!/bin/sh

create_mdc_path()
{
local_path=$1
username=$2
usergroup=$3
mod=$4
mkdir -p ${local_path}
chown ${username}:${usergroup} ${local_path}
chmod ${mod} ${local_path}
}
cp_cfg_path()
{
filename=$1
if [ ! -e "/cfg/configserver/$filename" ]; then
   if [  -e "/cfg/$filename" ]; then
      mv "/cfg/$filename"  "/cfg/configserver/"
      chmod 750 /cfg/configserver/$filename
      chown nvidia:nvidia /cfg/configserver/$filename
   fi
fi
}

echo "start create nos directory and file"

create_mdc_path "/opt/usr/sbin" nvidia nvidia 755
create_mdc_path "/opt/usr/storage" nvidia nvidia 755
create_mdc_path "/opt/usr/storage/rmdiag" nvidia nvidia 755
create_mdc_path "/opt/usr/storage/rmdiag/plugin" nvidia nvidia 755
create_mdc_path "/opt/usr/col/fm"    nvidia nvidia 755
create_mdc_path "/opt/usr/col/fm/gdb" nvidia nvidia 755
create_mdc_path "/opt/usr/col/rt" nvidia nvidia 755
create_mdc_path "/opt/usr/col/runinfo" nvidia nvidia 755
create_mdc_path "/opt/usr/col/runinfo/flowchart/data" nvidia nvidia 755
create_mdc_path "/opt/usr/col/runinfo/stat" nvidia nvidia 755
create_mdc_path "/opt/usr/col/runinfo/limit" nvidia nvidia 755
create_mdc_path "/opt/usr/col/runinfo/advc" nvidia nvidia 755
create_mdc_path "/opt/usr/col/log" nvidia nvidia 755
create_mdc_path "/opt/usr/col/log/all" nvidia nvidia 755
create_mdc_path "/opt/usr/col/log/fm" nvidia nvidia 755
create_mdc_path "/opt/usr/col/log/svp" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/commonrec" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/videorec" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/masked" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/dssad" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/original" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/original/desense" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/original/videorec" nvidia nvidia 755
create_mdc_path "/opt/usr/col/bag/original/commonrec" nvidia nvidia 755
create_mdc_path "/opt/usr/col/can" nvidia nvidia 755
create_mdc_path "/opt/usr/col/eth" nvidia nvidia 755
create_mdc_path "/opt/usr/col/mcu" nvidia nvidia 755
create_mdc_path "/opt/usr/col/planning" nvidia nvidia 755
create_mdc_path "/opt/usr/col/planning/conf" nvidia nvidia 755
create_mdc_path "/opt/usr/col/planning/stat" nvidia nvidia 755
create_mdc_path "/opt/usr/col/planning/old" nvidia nvidia 755
create_mdc_path "/opt/usr/col/perception" nvidia nvidia 755
create_mdc_path "/opt/usr/col/calibration" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/MCUADAS" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/CAN" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/TRIGGER" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/FAULT" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/TRIGGERDESC" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/ETH" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/MCULOG" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/PLANNING" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/CALIBRATION" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/ALLLOG" nvidia nvidia 755
create_mdc_path "/opt/usr/col/toupload/mcu" nvidia nvidia 755
create_mdc_path "/opt/usr/log/soc_log"    nvidia nvidia 755
create_mdc_path "/opt/usr/log/ota_log"    nvidia nvidia 755
create_mdc_path "/opt/usr/log/mcu_log"    nvidia nvidia 755
create_mdc_path "/opt/usr/log/system_monitor_log"    nvidia nvidia 755
create_mdc_path "/opt/usr/upgrade/download"    nvidia nvidia 755
create_mdc_path "/opt/usr/upgrade/install"    nvidia nvidia 755
create_mdc_path "/opt/usr/log_bak/soc_log"    nvidia nvidia 755
create_mdc_path "/opt/usr/log_bak/mcu_log"    nvidia nvidia 755
create_mdc_path "/opt/usr/log_bak/res_collect"    nvidia nvidia 755
create_mdc_path "/opt/usr/sensor_calib"    nvidia nvidia 755
create_mdc_path "/opt/usr/mcu_adas"    nvidia nvidia 755
create_mdc_path "/hz_map/ntp"    nvidia nvidia 755
create_mdc_path "/hd_map/maps"    nvidia nvidia 755

create_mdc_path "/cfg/pki/certs"    nvidia nvidia 755
create_mdc_path "/cfg/sec/keys"    nvidia nvidia 755
create_mdc_path "/cfg/rt"    nvidia nvidia 755
create_mdc_path "/cfg/hd_uuid"    nvidia nvidia 755
create_mdc_path "/cfg/conf_app"    nvidia nvidia 755
create_mdc_path "/cfg/conf_app/file_monitor"    nvidia nvidia 755
create_mdc_path "/cfg/conf_ota"    nvidia nvidia 755
create_mdc_path "/cfg/configserver"    nvidia nvidia 755
create_mdc_path "/cfg/oemcertstorage"    nvidia nvidia 755

create_mdc_path "/cfg/conf_calib_7v"    nvidia nvidia 755
create_mdc_path "/cfg/conf_cam_7v"    nvidia nvidia 755
create_mdc_path "/cfg/conf_calib_apa"    nvidia nvidia 755
create_mdc_path "/cfg/conf_cam_apa"    nvidia nvidia 755
create_mdc_path "/cfg/conf_calib_lidar"    nvidia nvidia 755
create_mdc_path "/cfg/conf_pcl_lidar"    nvidia nvidia 755
create_mdc_path "/cfg/conf_car_XX"    nvidia nvidia 755
create_mdc_path "/cfg/conf_em"    nvidia nvidia 755
create_mdc_path "/cfg/dids"    nvidia nvidia 755
create_mdc_path "/cfg/version"    nvidia nvidia 755
create_mdc_path "/cfg/ota"    nvidia nvidia 755
create_mdc_path "/cfg/pki"    nvidia nvidia 755
create_mdc_path "/cfg/system"    nvidia nvidia 755
create_mdc_path "/cfg/alg_mem"    nvidia nvidia 755
create_mdc_path "/cfg/lidar_intrinsic_param"    nvidia nvidia 755

create_mdc_path "/cfg_bak/configserver"    nvidia nvidia 755
create_mdc_path "/cfg_bak/dids"    nvidia nvidia 755
create_mdc_path "/cfg_bak/version"    nvidia nvidia 755
create_mdc_path "/cfg_bak/ota"    nvidia nvidia 755
create_mdc_path "/cfg_bak/pki"    nvidia nvidia 755
create_mdc_path "/cfg_bak/system"    nvidia nvidia 755
create_mdc_path "/cfg_bak/alg_mem"    nvidia nvidia 755
create_mdc_path "/cfg_bak/conf_calib_7v"    nvidia nvidia 755
create_mdc_path "/cfg_bak/conf_calib_apa"    nvidia nvidia 755
create_mdc_path "/cfg_bak/conf_calib_lidar"    nvidia nvidia 755


create_mdc_path "/ota/recovery"    nvidia nvidia 755

mount -t tmpfs -o size=200M  tmpfs /opt/usr/col/bag/commonrec
mount -t tmpfs -o size=500M  tmpfs /opt/usr/col/bag/videorec
mount -t tmpfs -o size=4M tmpfs /opt/usr/col/runinfo/flowchart/data
mount -t tmpfs -o size=1500M  tmpfs /opt/usr/sensor_calib

if [ ! -e "/cfg/lidar_intrinsic_param/get_correct_file.py" ]; then
    cp -rf  /app/runtime_service/neta_lidar/conf/get_correct_file.py /cfg/lidar_intrinsic_param/get_correct_file.py
fi

if [ ! -e "/cfg/conf_em/startup_manifest.json" ]; then
    cp -rf  /app/conf/startup_manifest.json /cfg/conf_em/startup_manifest.json
fi

if [ ! -e "/opt/usr/col/runinfo/flowchart/index.html" ]; then
    cp -rf  /app/conf/dc_flow/index.html /opt/usr/col/runinfo/flowchart/index.html
fi

# link
if [ ! -e "/log/LogShare/soc_log" ]; then
   ln -sf "/opt/usr/log/soc_log"  "/log/LogShare/soc_log"
fi
if [ ! -e "/log/LogShare/mcu_log" ]; then
   ln -sf "/opt/usr/log/mcu_log"  "/log/LogShare/mcu_log"
fi
if [ ! -e "/log/LogShare/ota_log" ]; then
   ln -sf "/opt/usr/log/ota_log"  "/log/LogShare/ota_log"
fi
if [ ! -e "/log/LogShare/system_monitor_log" ]; then
   ln -sf "/opt/usr/log/system_monitor_log"  "/log/LogShare/system_monitor_log"
fi
if [ ! -e "/log/LogShare/svp_log" ]; then
   ln -sf "/svp_log"  "/log/LogShare/svp_log"
fi

if [! -e "/opt/usr/log/soc_log/history.log"]; then
   touch /opt/usr/log/soc_log/history.log
   chmod 666 /opt/usr/log/soc_log/history.log
   chown nvidia:nvidia /opt/usr/log/soc_log/history.log
else
   chmod 666 /opt/usr/log/soc_log/history.log
   chown nvidia:nvidia /opt/usr/log/soc_log/history.log
fi

NODE_LINK="/opt/usr/sbin/nos"
NODE="/app/scripts/nos_tool.sh"
if [ ! -e $NODE_LINK ]; then
    ln -sf $NODE $NODE_LINK
    if [ $? -ne 0 ]; then
        echo "create link: $NODE failed"
    fi
fi

cp_cfg_path "config_param.json"
cp_cfg_path "config_param.json_crc"
cp_cfg_path "config_param.json_bak_1"
cp_cfg_path "config_param.json_bak_1_crc"

# Init env
/app/scripts/coredump_setup.sh
/app/scripts/orin_cmd_history.sh
/app/scripts/set_irq_affiniy.sh
/app/scripts/pki_config.sh

# Start svr
echo "start nos app service"

/app/scripts/start_core_app.sh
/app/scripts/core_file_monitor.sh &

/app/scripts/start_someip.sh >> /dev/null &

#Get Lidar correct file
/app/scripts/get_lidar_correct_file.sh > /cfg/lidar_intrinsic_param/lidar.txt  &

#/opt/usr/col dir manager
/app/scripts/dc_col_mgr.sh 1 300 >> /dev/null &

#gptp
# ptp4l -i mgbe3_0 -f /etc/automotive-slave.cfg &
# phc2sys -s mgbe3_0 -O 0 -S 1 &
