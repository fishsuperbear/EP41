cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "hz_ins")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

add_definitions(-DINS_PVATB_DEBUG_ON)

link_directories(
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib 
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

file(GLOB_RECURSE
    SOURCES_FILE
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    )

add_executable(hz_ins ${SOURCES_FILE})

# third_party/x86/protobuf/include/google/protobuf/port_def.inc
target_include_directories(hz_ins 
    PRIVATE
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
    ${CMAKE_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/../common/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${CMAKE_CURRENT_SOURCE_DIR}/../
    ${NETA_MIDDLEWARE_DIR}/
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include
)

target_link_libraries(hz_ins
    pthread
    dl 
    rt
    neta_log
    yaml-cpp
    neta_cm
    neta_crc
    neta_e2e
    neta_e2exf
    neta_e2exf_cpp
)

install(
    TARGETS 
    hz_ins
    RUNTIME 
    DESTINATION ${TARGET_PLATFORM}/bin)


#for test
install(
    TARGETS
    hz_ins
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/hz_ins/bin
)

install( FILES ${CMAKE_CURRENT_SOURCE_DIR}/../config/hz_ins.yaml
    DESTINATION ${TARGET_PLATFORM}/test/hz_ins/conf
)
