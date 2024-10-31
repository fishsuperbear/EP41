
cmake_minimum_required(VERSION 2.8.12)

project(cm_freqchecker_tool)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib) 
        
add_executable(cm_freqchecker_tool 
        ${CMAKE_CURRENT_SOURCE_DIR}/cm_freqchecker_tool.cpp
)

target_include_directories(cm_freqchecker_tool
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)

target_link_libraries(cm_freqchecker_tool
        pthread
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        neta_cm
        dl
        rt
        netaos::log
        protobuf
        uuid
        globalproto
)

install(
    TARGETS cm_freqchecker_tool RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
