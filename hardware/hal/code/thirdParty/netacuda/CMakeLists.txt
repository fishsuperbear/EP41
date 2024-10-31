cmake_minimum_required(VERSION 3.5)

 project(netacuda)

 set(CMAKE_CXX_STANDARD 11)

 find_package(CUDA REQUIRED)
 # set(CUDA_ARCH "35" CACHE STRING "CUDA Architecture (default is '35')" )

#  Set the CUDA architecture to compile for
 set(CUDA_ARCH "-arch=sm_70; -arch=sm_72; -arch=sm_75; -arch=sm_80; -arch=sm_86" CACHE STRING "CUDA architecture")
 # Add the CUDA compiler flags
 set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH})
 FIND_INCLUDE_DIR(INCLUDE_DIR_PLAT_INTERFACE_LIST ${HARDWARE_ROOT_PATH}/hal/code/interface/platform)
 FIND_INCLUDE_DIR(INCLUDE_DIR_PORTING_LIST ${HARDWARE_ROOT_PATH}/hal/code/porting)

 include_directories(
     include
     ${INCLUDE_DIR_PLAT_INTERFACE_LIST}
     ${INCLUDE_DIR_PORTING_LIST}
 )

 file(GLOB_RECURSE SOURCES "src/zmq_u.cpp" "src/neta_mps.cpp" )
 
 file(GLOB_RECURSE TRT_BENCH_SRC "src/commandLine.cpp" 
 "src/tools.cpp" 
 "src/delayKernel.cu" 
 "src/customtasks/cudaKernelTask/cudaKernelTask.cu" 
 "src/Task.cpp"
 "src/customtasks/customTask.cpp"
 )
 cuda_add_library(${PROJECT_NAME} SHARED ${SOURCES} ${TRT_BENCH_SRC})

 link_directories(
      /usr/local/cuda/targets/aarch64-linux/lib
      /drive/drive-linux/lib-target
      /drive/drive-linux/filesystem/targetfs/usr/local/cuda-11.4/targets/aarch64-linux/lib
  )

  target_link_libraries(${PROJECT_NAME}
    hw_platform
    nvinfer
    nvinfer_plugin
    nvparsers
    nvonnxparser
    nvsciipc
    nvos
    nvsocsys
    nvtegrahv
    nvrm_sync
    nvrm_chip
    nvrm_host1x
    nvrm_gpu
    nvrm_mem
    cuda
    cudart 
  )
 
 include_directories(
     ${HARDWARE_ROOT_PATH}/driveos/v6.0.6/include/cuda
     )

 add_executable(mps_server "src/mps_server.cpp")

 target_link_libraries(mps_server
    nvinfer
    nvinfer_plugin
    nvdla_compiler
    nvdla_runtime
    cudla
    cublas
    cublasLt
    cudnn
    cuda
    cudart
    hw_platform
    netacuda
)

add_executable(mps_client "src/mps_client.cpp" "src/customtasks/cudaKernelTask/cudaKernelTask.cpp"  )
target_link_libraries(mps_client
nvinfer
nvinfer_plugin
nvdla_compiler
nvdla_runtime
cudla
cublas
cublasLt
cudnn
cuda
cudart
hw_platform
netacuda
)

set(SOURCE_DIR ${HARDWARE_ROOT_PATH}/hal/code/thirdParty/netacuda/include/netacuda)
file(GLOB HEADER_FILES "${SOURCE_DIR}/*.h" "${SOURCE_DIR}/*.hpp")

 install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/camera)
 install(TARGETS mps_server DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
#  install(TARGETS mps_client DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
 install(FILES  ${HEADER_FILES} DESTINATION ${CMAKE_INSTALL_PREFIX}/include/netacuda/)
