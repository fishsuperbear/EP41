
PROJECT_PATH=`pwd`

mkdir -p ${PROJECT_PATH}/build

### build new gpuutils lib
GPUUTILS_PATH="${PROJECT_PATH}/../../code/thirdParty/gpuutils/"
mkdir -p ${GPUUTILS_PATH}/build
cd ${GPUUTILS_PATH}/build
cmake ${GPUUTILS_PATH}
make -j8
make install

cd ${PROJECT_PATH}

### build new yolov5 lib
# YOLOV5_PATH="${PROJECT_PATH}/../../code/thirdParty/yolov5/"
# mkdir ${YOLOV5_PATH}/build
# cd ${YOLOV5_PATH}/build
# cmake ${YOLOV5_PATH}
# make -j8
# make install

# cd ${PROJECT_PATH}


########### build Project #####################
cd ${PROJECT_PATH}/build
cmake ${PROJECT_PATH}

make -j8
