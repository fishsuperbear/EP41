#bin/bash
PROJECT_PATH=$(pwd)
echo $PROJECT_PATH
HOST=10.4.50.140
LOGIN=orin
PASSWD=orin
#BOARD_DIR=/home/orin/gjq/

read -p "请选择输出的consumer_file (1.enc 2.cuda 3.vic): " consumer_file
if [ "$consumer_file" == "1" ];then
  export consumer_file=h26*
elif [ "$consumer_file" == "2" ];then
  export consumer_file=jpg
elif [ "$consumer_file" == "3" ];then
  export consumer_file=yuv*
else
    echo "无效的consumer_file选择！"
    exit 1
fi
read -p "请选择输出到${HOST}_${LOGIN}board的路径 " BOARD_DIR

function build_exe()
{
  mkdir build
  cd build
  cmake ..
  make -j
  for file in *; do
    if file "$file" | grep -q 'ELF 64-bit LSB executable'; then
        name="$file"
        echo "已生成 $name 文件"
      if [ -n "$name" ]; then
      echo "正在将 $name 文件复制到远程服务器"
      sshpass -p "orin" scp "$name" ${LOGIN}@${HOST}:${BOARD_DIR} 
        if [ $? -eq 0 ]; then
           echo "文件复制成功"
        else
            echo "文件复制失败"
        exit 1
        fi
        # 4. 在远程服务器上执行$name文件
        echo "正在登录远程服务器并执行 $name 文件"
        sshpass -p "orin" ssh ${LOGIN}@${HOST} "cd ${BOARD_DIR} && rm -f *.${consumer_file} && ./$name && if ls *.${consumer_file} >/dev/null 2>&1; then echo '生成了${consumer_file}格式图片'; else echo '没有生成${consumer_file}格式图片'; fi"
    else
      echo "本文件夹中没有.out类型的文件"
      cd "${PROJECT_PATH}"
      exit 1
    fi
    fi
  done
  cd ${PROJECT_PATH}
  exit 1
}
# 判断文件夹是否存在
if [ -d "build" ]; then
  rm build -r
  build_exe
else
  build_exe
fi

cd ${PROJECT_PATH}




