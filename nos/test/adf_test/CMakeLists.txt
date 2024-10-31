
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "adf_test")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_SOURCE_DIR}/../netaos_thirdparty/cmake/Packages.cmake)
find_package(Protobuf REQUIRED)

file(GLOB_RECURSE
        IDL_SOURCES
        ${NETA_MIDDLEWARE_DIR}/idl/generated/*.cxx)

add_executable(node_test_recv ${CMAKE_CURRENT_SOURCE_DIR}/src/node_test_client_main.cpp ${IDL_SOURCES})
add_executable(node_test_send ${CMAKE_CURRENT_SOURCE_DIR}/src/node_test_send_main.cpp ${IDL_SOURCES})
# add_executable(sensor_test_main ${CMAKE_CURRENT_SOURCE_DIR}/src/sensor_test_main.cpp ${IDL_SOURCES})

target_include_directories(node_test_recv
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

add_definitions(-DNODE_DATA_TEST)

target_include_directories(node_test_send
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
)

# target_include_directories(sensor_test_main
#     PRIVATE
#     ${CMAKE_CURRENT_SOURCE_DIR}/
# )

target_link_libraries(node_test_recv
    PRIVATE
    netaos::adf
    netaos::cm
)
target_link_libraries(node_test_send
    PRIVATE
    netaos::adf
    netaos::cm
)

# target_link_libraries(sensor_test_main
#     PRIVATE
#     netaos::adf
#     netaos::cm
# )

install(
    TARGETS 
    node_test_recv
    node_test_send
    # sensor_test_main
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    TARGETS 
    node_test_recv
    node_test_send
    # sensor_test_main
    RUNTIME
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/adf/bin
)

install(DIRECTORY ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/conf/adf/ 
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/adf/conf
)

install(DIRECTORY ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/adf/
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/unit_test/adf_test_unit/testcase_ADF_001
    USE_SOURCE_PERMISSIONS )
