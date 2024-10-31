#module名称
set(MODULE_NAME, "sensor_trans")
if(${BUILD_TYPE} STREQUAL "release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1200 -g -O3")
elseif(${BUILD_TYPE} STREQUAL "debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1200 -g")
endif()

include(${CMAKE_SOURCE_DIR}/netaos_thirdparty/cmake/Packages.cmake)
find_package(Protobuf REQUIRED)

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/common*.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase*.cxx
    ${NETA_MIDDLEWARE_DIR}/idl/generated/chassis_ota_method*.cxx)
    
add_executable(sensor_trans  ${SOURCES}) 

target_include_directories(sensor_trans
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${NETA_MIDDLEWARE_DIR}/
    ${CMAKE_SOURCE_DIR}/
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/include/
    ${CMAKE_SOURCE_DIR}/service/ethstack/sensor_trans/gen/includes/
)

target_link_directories(sensor_trans
    PRIVATE
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
)

target_link_libraries(sensor_trans
    pthread
    dl
    rt 
    fastcdr
    fastrtps
    protobuf
    neta_log
    yaml-cpp
    neta_cm
    hozon_someip
    neta_adf
    neta_cfg
)

install(
    TARGETS
    sensor_trans
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/sensor_trans.yaml
    DESTINATION ${TARGET_PLATFORM}/conf/
)

install(
    TARGETS 
    sensor_trans
    RUNTIME 
    DESTINATION ${TARGET_PLATFORM}/test/sensor_trans/bin/
    )

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/sensor_trans.yaml
    DESTINATION ${TARGET_PLATFORM}/test/sensor_trans/conf/
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/sensor_trans.yaml
    DESTINATION ${TARGET_PLATFORM}/test/adf/conf/
)
