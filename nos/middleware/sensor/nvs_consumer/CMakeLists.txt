project(nvs_consumer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories(
    ${NETA_MIDDLEWARE_DIR}/
    ${COMPILER_SYSROOT}/include
    ${COMPILER_SYSROOT}/include/nvmedia_6x
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include
)

link_directories(
    ${COMPILER_SYSROOT}/lib
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/
    ${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${COMPILER_PATH}/../usr/lib
)

add_library(nvs_consumer SHARED 
    CClientCommon.cpp
    CConsumer.cpp
    CCudaConsumer.cpp
    CEncConsumer.cpp
    # CNvMediaConsumer.cpp
    CUtils.cpp
    CEncManager.cpp
    CDisplayConsumer.cpp
    COpenWFDController.cpp
)

target_link_libraries(nvs_consumer PUBLIC
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
    tegrawfd
    nvidia-kms
    nvdc
    nvddk_2d_v2
    nvimp
    nvddk_vic
)

add_executable(desay_nvs_recv
    tool/nvs_desay_recv.cpp
)

target_link_libraries(desay_nvs_recv
    nvs_consumer
)

install(TARGETS nvs_consumer LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(TARGETS desay_nvs_recv RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin)

install(FILES 
        ./CChannel.hpp
        ./CClientCommon.hpp
        ./CCmdLineParser.hpp
        ./CConsumer.hpp
        ./CCudaConsumer.hpp
        ./CEncConsumer.hpp
        ./CEventHandler.hpp
        ./CFactory.hpp
        ./CIpcConsumerChannel.hpp
        ./CMaster.hpp
        ./CNvMediaConsumer.hpp
        ./Common.hpp
        ./CProfiler.hpp
        ./CSingleProcessChannel.hpp
        ./CUtils.hpp
        DESTINATION ${TARGET_PLATFORM}/include/sensor/nvs_consumer)