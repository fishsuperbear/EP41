CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

project("mcu_fault_gen")

set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(COMPILER_PATH /usr/local/orin_sdk/aarch64)
set(CMAKE_C_COMPILER ${COMPILER_PATH}/bin/aarch64-linux-gcc)
set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/bin/aarch64-linux-g++)
set(CMAKE_LINKER ${COMPILER_PATH}/bin/aarch64-linux-ld)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -pthread -Wall")

include_directories(
.
${CMAKE_CURRENT_SOURCE_DIR}/includes
${CMAKE_CURRENT_SOURCE_DIR}/iauto_someip/include
)

#########################################################################
# lib: libMcuFaultService.so
#########################################################################
set(MCUFAULTSERVICE_LIBRARY_NAME McuFaultService)
set(MCUFAULTSERVICE_LIBRARY_SRC_FILES
    src/hozon/netaos/v1/mcufaultservice_proxy.cpp
    src/hozon/netaos/v1/mcufaultservice_manifest.cpp
)

add_library(${MCUFAULTSERVICE_LIBRARY_NAME} SHARED ${MCUFAULTSERVICE_LIBRARY_SRC_FILES})

target_link_libraries(${MCUFAULTSERVICE_LIBRARY_NAME}
    PUBLIC
    ara_com_ics
    someip_e2e_ics
    nlog_ics
    aracore_ics
    dlt_ics
    perftoolkit_ics
    ara_serialize_ics
    ncdump_ics
    someip_ics
    ara_someipbinding_ics
    ncjson_ics
    someip_serializer_ics
    ncore_ics
    ssl
    crypto
    pthread
)

target_link_directories(${MCUFAULTSERVICE_LIBRARY_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/iauto_someip/lib
)

install(TARGETS ${MCUFAULTSERVICE_LIBRARY_NAME}
        LIBRARY
        DESTINATION ${CMAKE_SOURCE_DIR}
)