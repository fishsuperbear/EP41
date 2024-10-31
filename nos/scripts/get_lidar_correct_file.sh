#!/bin/bash

# 指定路径
target_path="/cfg/lidar_intrinsic_param"
file_path="/cfg/lidar_intrinsic_param/ATD128P.dat"
python_script_path="/cfg/lidar_intrinsic_param/get_correct_file.py"

echo "enter get correct file scripts..." >> /cfg/lidar_intrinsic_param/lidar.txt 

source /app/scripts/env_setup.sh
cd "$target_path" || exit 1

# 判断文件是否存在
if [ ! -e "$file_path" ]; then
    echo "File does not exist. Executing the Python script..." >> /cfg/lidar_intrinsic_param/lidar.txt 
    
    # 检查系统中是否存在 python 命令
    if command -v python &> /dev/null; then
        echo "Python is installed. Executing the Python script..." >> /cfg/lidar_intrinsic_param/lidar.txt
        # 执行 Python 脚本
        python "$python_script_path"
        wait $!
    else
        echo "Python is not installed. Please install Python and then run this script again.">> /cfg/lidar_intrinsic_param/lidar.txt
        # 在这里可以添加等待 Python 环境安装的逻辑
        while ! command -v python &> /dev/null; do
            echo "Waiting for Python 3 installation..." >> /cfg/lidar_intrinsic_param/lidar.txt
            sleep 1
        done
        
        echo "Python is now installed. Executing the Python script..." >> /cfg/lidar_intrinsic_param/lidar.txt
        python "$python_script_path"
        wait $!
    fi
    sleep 1
# else
    echo "File exists. Exiting the script." >> /cfg/lidar_intrinsic_param/lidar.txt
    exit 1
fi
exit 0