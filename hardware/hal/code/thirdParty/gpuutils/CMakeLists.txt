cmake_minimum_required(VERSION 3.5)

project(gpuutils)

set(CMAKE_CXX_STANDARD 11)

# set(CUDA_TOOLKIT_ROOT_DIR /drive/drive-linux/filesystem/targetfs/usr/local/cuda-11.4/bin/)
find_package(CUDA REQUIRED)
# enable_language(CUDA)
# set(CUDA_ARCH "35" CACHE STRING "CUDA Architecture (default is '35')" )

include_directories(
    include
    ${CUDA_INCLUDE_DIRS}
    ${HARDWARE_ROOT_PATH}/hal/code/modules/devices/video/modules/nvmedia/common
)

file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

cuda_add_library(${PROJECT_NAME} SHARED ${SOURCES})

link_directories(
     /usr/local/cuda/targets/aarch64-linux/lib
 )


install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/camera)
# install(DIRECTORY include/ DESTINATION include)

