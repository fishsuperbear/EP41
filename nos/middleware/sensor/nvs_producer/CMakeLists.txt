project(nvs_producer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories(
    ${NETA_MIDDLEWARE_DIR}/idl/generated/
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
)

link_directories(
    ${COMPILER_SYSROOT}/lib
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattach*.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/camera_internal_data*.cxx
)

# remove ide files

file(GLOB_RECURSE REMOVE_SOURCES "CVirtualCamProducer.cpp" "CVirtualCamVicProducer.cpp" "CRecoderMaster.cpp")
list(REMOVE_ITEM SOURCES ${REMOVE_SOURCES})

add_executable(nvs_producer
    ${SOURCES}
)

target_link_libraries(nvs_producer
    nvscistream
    nvscibuf
    nvscisync
    nvscievent
    nvsciipc
    nvscicommon
    cuda
    cudart
    neta_log
    nvmedia2d
    nvvic
    nvsipl
    nvsipl_query
    nvmedia_iep_sci
    nvvideo
    neta_cm
    neta_phm
    neta_em_client
    yaml-cpp
    tegrawfd
    nvidia-kms
    nvdc
    nvddk_2d_v2
    nvimp
    nvddk_vic
    diagnosis
)

install(TARGETS nvs_producer RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/conf/
        DESTINATION ${TARGET_PLATFORM}/conf
        PATTERN "*")

# ide logic compile to lib
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../codec/src/nvmedia/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/../../
    ${NETA_MIDDLEWARE_DIR}/idl/generated/
)

set(NV_TOPDIR ${COMPILER_PATH}/..)
set(NV_PLATFORM_DIR ${NV_TOPDIR}/drive-linux)
set(NV_PLATFORM_SDK_INC_DIR ${NV_PLATFORM_DIR}/include)
set(NV_PLATFORM_SDK_LIB_DIR ${NV_PLATFORM_DIR}/lib-target ${COMPILER_PATH}/../usr/lib ${COMPILER_SYSROOT}/lib)

file(GLOB_RECURSE
    IDL_SOURCES
    ${NETA_MIDDLEWARE_DIR}/idl/generated/sensor_reattach*.cxx
)

set(SRC_IDE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../codec/src/nvmedia/utils/scibuf_utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../codec/src/nvmedia/utils/misc_utils.c
    ${CMAKE_CURRENT_SOURCE_DIR}/../../codec/src/nvmedia/utils/log_utils.c
    CVirtualCamProducer.cpp
    CVirtualCamVicProducer.cpp
    CRecoderMaster.cpp
    CProducer.cpp
    CPoolManager.cpp
    CClientCommon.cpp
    CUtils.cpp
    CLateConsumerHelper.cpp
)

add_library(nv_ide SHARED ${SRC_IDE} ${IDL_SOURCES})

target_link_directories(nv_ide
    PUBLIC
    ${NV_PLATFORM_SDK_LIB_DIR}
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)

target_link_libraries(nv_ide
    PUBLIC
    nvscibuf
    nvscisync
    nvmedia_ide_parser
    nvmedia_ide_sci
    nvparser
    nvvideo
    pthread
    nvscistream
    nvmedia2d
    nvvic
    cuda
    cudart
    neta_cm
    tegrawfd
    nvidia-kms
    nvdc
    nvddk_2d_v2
    nvimp
    nvddk_vic
    yaml-cpp
)

install(TARGETS nv_ide LIBRARY DESTINATION ${TARGET_PLATFORM}/lib)
