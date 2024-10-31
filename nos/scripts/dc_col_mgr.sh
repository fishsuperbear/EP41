#!/bin/bash
configFileName="config.txt"
updateconfigflag=$1 #1: updates needed  other:  No update required
sleep_time=$2 # Periodic execution Unit is seconds
mk_cfg_path_folder()
{
    dirpath=$1
    maxFolderNum=$2
    maxFolderSize=$3
    type="folder"
    if [  -e "$dirpath/$configFileName" ]; then
        if [[ $updateconfigflag != 1 ]]; then
            echo "No update required  $dirpath/$configFileName"
            return 1
        fi
       
    fi
    touch $dirpath/$configFileName
    echo -e "type=$type\nmaxFolderNum=$maxFolderNum\nmaxFolderSize=$maxFolderSize" > $dirpath/$configFileName
    chmod 750 $dirpath/$configFileName
    chown nvidia:nvidia $dirpath/$configFileName
    echo "dirpath= $dirpath type= $type maxFolderNum= $maxFolderNum maxFolderSize= $maxFolderSize"
}
mk_cfg_path_file()
{
    dirpath=$1
    maxFilesNum=$2
    maxFileSize=$3
    maxDays=$4
    type="file"
    if [  -e "$dirpath/$configFileName" ]; then
        if [[ $updateconfigflag != 1 ]]; then
            echo "No update required  $dirpath/$configFileName"
            return 1
        fi        
    fi
    touch $dirpath/$configFileName
    echo -e "type=$type\nmaxFilesNum=$maxFilesNum\nmaxFileSize=$maxFileSize\nmaxDays=$maxDays" > $dirpath/$configFileName
    chmod 750 $dirpath/$configFileName
    chown nvidia:nvidia $dirpath/$configFileName
    echo "dirpath= $dirpath type= $type maxFilesNum= $maxFilesNum maxFileSize= $maxFileSize maxDays= $maxDays"
}

mk_cfg()
{
    mk_cfg_path_folder "/opt/usr/col/bag/original/desense" 7 $(expr 700 \* 1024 \* 1024)
    mk_cfg_path_folder "/opt/usr/col/bag/original/videorec" 7 $(expr 700 \* 1024 \* 1024)
    mk_cfg_path_folder "/opt/usr/col/bag/original/commonrec" 10 $(expr 400 \* 1024 \* 1024)
    mk_cfg_path_folder "/opt/usr/col/bag/masked" 10 $(expr 1500 \* 1024 \* 1024)
    mk_cfg_path_folder "/opt/usr/col/bag/dssad" 50 $(expr 1500 \* 1024 \* 1024)
    mk_cfg_path_file "/opt/usr/col/can" 210 $(expr 300 \* 1024 \* 1024) 180
    mk_cfg_path_file "/opt/usr/col/eth" 210 $(expr 300 \* 1024 \* 1024) 180
    mk_cfg_path_file "/opt/usr/col/mcu" 210 $(expr 300 \* 1024 \* 1024) 180
    mk_cfg_path_file "/opt/usr/col/planning/old" 210 $(expr 20 \* 1024 \* 1024) 180
    mk_cfg_path_file "/opt/usr/col/log/all" 2 $(expr 4000 \* 1024 \* 1024) 180
    mk_cfg_path_file "/opt/usr/col/log/fm" 20 $(expr 200 \* 1024 \* 1024) 180
    mk_cfg_path_file "/opt/usr/col/calibration" 5 $(expr 1000 \* 1024 \* 1024) 180
    mk_cfg_path_file "/opt/usr/mcu_adas" 210 $(expr 300 \* 1024 \* 1024) 180
}


col_mgr()
{
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/bag/original/desense"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/bag/original/videorec"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/bag/original/commonrec"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/bag/masked"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/bag/dssad"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/can"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/eth"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/mcu"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/planning/old"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/log/all"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/log/fm"
    /app/scripts/dc_file_mgr.sh "/opt/usr/col/calibration"
    /app/scripts/dc_file_mgr.sh "/opt/usr/mcu_adas"
}

mk_cfg
while true
do
  echo "col_mgr+ $(date "+%Y-%m-%d %H:%M:%S")"
  col_mgr
  echo "col_mgr- $(date "+%Y-%m-%d %H:%M:%S")"
  sleep ${sleep_time}
done