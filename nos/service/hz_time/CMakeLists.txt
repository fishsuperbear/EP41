#module名称
set(MODULE_NAME "hz_time")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC --std=gnu++14 -O3 -Wall -Werror")

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)

add_executable(${MODULE_NAME}  ${SOURCES})

target_include_directories(${MODULE_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include/
    ${NETA_MIDDLEWARE_DIR}/
    ${CMAKE_SOURCE_DIR}/
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    ${CMAKE_SOURCE_DIR}/middleware/cfg/include
)

target_link_directories(${MODULE_NAME}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib/
)

target_link_libraries(${MODULE_NAME}
    PRIVATE
    pthread
    protobuf
    yaml-cpp
    neta_log
    neta_adf
    neta_cm 
    neta_cfg
)

install(
    TARGETS
    ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/conf/${MODULE_NAME}.yaml
    DESTINATION ${TARGET_PLATFORM}/conf/
)

