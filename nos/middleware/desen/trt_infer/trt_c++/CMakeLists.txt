cmake_minimum_required(VERSION 3.16) 
project(de_privacy_infer)


add_definitions(-w)

#Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 11) 
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall -O3 -Wno-deprecated-declarations")

#For finding FindTensorRT.cmake
        set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

#TODO : Specify the path to TensorRT root dir
if (NOT TensorRT_DIR) 
    set(TensorRT_DIR/ usr/local/libs/TensorRT/) 
endif()

#Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda) 
include_directories("${PROJECT_SOURCE_DIR}/")
#We require CUDA, OpenCV, and TensorRT
#find_package(TensorRT REQUIRED)

find_package(CUDA REQUIRED) 

find_package(OpenCV REQUIRED) 
include_directories(/usr/local/cuda/include) 

message(${OpenCV_INCLUDE_DIRS})
#for tensorrt
                    
link_directories("/usr/lib/x86_64-linux-gnu")
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")

aux_source_directory(./src SRCS)
add_executable(de_privacy_infer ${SRCS})
target_link_libraries(de_privacy_infer -lnvinfer)                                        
target_link_libraries(de_privacy_infer  ${OpenCV_LIBS})
target_link_libraries(de_privacy_infer  -lcuda cudart)
