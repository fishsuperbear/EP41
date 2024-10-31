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

create_mdc_path "/opt/usr/cfg/conf_calib_9yq" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_calib_9tt" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_calib_9zs" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_calib_ldp" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_app" mdc mdc 755
create_mdc_path "/opt/usr/cfg/pki" mdc mdc 755
create_mdc_path "/opt/usr/cfg/configserver" mdc mdc 755
create_mdc_path "/opt/usr/log/hz_log" mdc mdc 755
create_mdc_path "/opt/usr/log/ota_log" mdc mdc 755
create_mdc_path "/opt/usr/log/pki_log" mdc mdc 755
create_mdc_path "/opt/usr/log/hd_log" mdc mdc 755
create_mdc_path "/opt/usr/col/fm" mdc mdc 755
create_mdc_path "/opt/usr/col/runinfo" mdc mdc 755
create_mdc_path "/opt/usr/col/log" mdc mdc 755
create_mdc_path "/opt/usr/col/bag" mdc mdc 755
create_mdc_path "/opt/usr/hz_log" mdc mdc 755
create_mdc_path "/opt/usr/hw_log" mdc mdc 755
create_mdc_path "/opt/usr/cfg/log" mdc mdc 755
create_mdc_path "/opt/usr/cfg/log/gea" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_cam_9tt" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_cam_9yq" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_calib_prk" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_calib_temp" mdc mdc 755
create_mdc_path "/opt/usr/cfg/lidar_to_camera" mdc mdc 755
create_mdc_path "/opt/usr/cfg/conf_lidar_ldp" mdc mdc 755

cp -f /opt/app/1/conf/log_daemon_storage.conf /opt/usr/cfg/log/gea/
chmod 755 /opt/usr/cfg/log/gea/log_daemon_storage.conf

# doip iptables nat set for car 0s#37
iptables -t nat -A PREROUTING -d 192.168.33.42/32 -i ethg0.33 -p tcp -m tcp --dport 13402 -j DNAT --to-destination 192.168.10.6:13402
iptables -t nat -A PREROUTING -d 192.168.33.42/32 -i ethg0.33 -p udp -m udp --dport 13402 -j DNAT --to-destination 192.168.10.6:13402

# doip iptables nat set for bench 188
iptables -t nat -A PREROUTING -d 10.4.51.188/32 -i ethg1 -p tcp -m tcp --dport 13402 -j DNAT --to-destination 192.168.10.6:13402
iptables -t nat -A PREROUTING -d 10.4.51.188/32 -i ethg1 -p udp -m udp --dport 13402 -j DNAT --to-destination 192.168.10.6:13402

sleep 10

cd /opt/usr/
#killall diag_server
sh /opt/usr/diag_update/mdc-llvm/scripts/start_um.sh > /dev/null
sleep 2
sh /opt/usr/diag_update/mdc-llvm/scripts/start_dm.sh > /dev/null
cd -

sh /opt/app/1/script/load_config.sh
