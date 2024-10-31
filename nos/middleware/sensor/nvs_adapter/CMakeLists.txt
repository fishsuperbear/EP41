project(nvs_adapter)

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

add_library(nvs SHARED 
    nvs_block_common.cpp
    nvs_block_ipc_dst.cpp
    nvs_block_pool.cpp
    nvs_block_queue.cpp
    nvs_block_img_producer.cpp
    nvs_block_multicast.cpp
    nvs_block_ipc_src.cpp
    nvs_adapter_producer.cpp
    nvs_utility.cpp
    nvs_block_img_consumer.cpp
    nvs_block_cuda_consumer.cpp
    nvs_adapter_consumer.cpp
    nvs_block_sipl_producer.cpp
)

target_link_libraries(nvs
    PUBLIC
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
)

add_executable(nvs_recv
    tool/nvs_recv.cpp
)

target_link_libraries(nvs_recv
    nvs
)

add_executable(nvs_send
    tool/nvs_send.cpp
)

target_link_libraries(nvs_send
    nvs
)

install(TARGETS nvs LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(TARGETS nvs_recv nvs_send RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin)

install(FILES 
        nvs_adapter_consumer.h  
        nvs_block_common.h         
        nvs_block_img_consumer.h  
        nvs_block_ipc_dst.h  
        nvs_block_multicast.h  
        nvs_block_queue.h    
        nvs_helper.h  
        nvs_utility.h
        nvs_adapter_producer.h  
        nvs_block_cuda_consumer.h  
        nvs_block_img_producer.h  
        nvs_block_ipc_src.h  
        nvs_block_pool.h       
        nvs_event_service.h  
        nvs_logger.h
        nvs_block_sipl_producer.h
        DESTINATION ${TARGET_PLATFORM}/include/sensor/nvs_adapter)