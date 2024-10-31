CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -pthread -Wall")

set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(COMPILER_PATH /usr/local/orin_sdk/aarch64)
set(CMAKE_C_COMPILER ${COMPILER_PATH}/bin/aarch64-linux-gcc)
set(CMAKE_CXX_COMPILER ${COMPILER_PATH}/bin/aarch64-linux-g++)
set(CMAKE_LINKER ${COMPILER_PATH}/bin/aarch64-linux-ld)

include_directories(
.
${CMAKE_CURRENT_SOURCE_DIR}/includes
${CMAKE_CURRENT_SOURCE_DIR}/../../ethstack/iauto_someip/include
)

link_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../ethstack/iauto_someip/lib
)

#########################################################################
# lib: libswc_IAUTOSAR_AASWC_SocFaultService_Server_SOC.so
#########################################################################
set(AASWC_SOCFAULTSERVICE_SERVER_SOCSWC_DEPLOYMENT_LIBRARY_NAME nos_spm_svr)
set(AASWC_SOCFAULTSERVICE_SERVER_SOCSWC_DEPLOYMENT_LIBRARY_SRC_FILES
src/hozon/netaos/v1/socpowerservice_skeleton.cpp
src/hozon/netaos/v1/socpowerservice_manifest.cpp
)

add_library(${AASWC_SOCFAULTSERVICE_SERVER_SOCSWC_DEPLOYMENT_LIBRARY_NAME} SHARED ${AASWC_SOCFAULTSERVICE_SERVER_SOCSWC_DEPLOYMENT_LIBRARY_SRC_FILES})


target_link_libraries(${AASWC_SOCFAULTSERVICE_SERVER_SOCSWC_DEPLOYMENT_LIBRARY_NAME}
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

# target_link_directories(${AASWC_SOCFAULTSERVICE_SERVER_SOCSWC_DEPLOYMENT_LIBRARY_NAME}
#     PUBLIC
#     ${CMAKE_CURRENT_SOURCE_DIR}/iauto_someip/lib
# )

install(TARGETS ${AASWC_SOCFAULTSERVICE_SERVER_SOCSWC_DEPLOYMENT_LIBRARY_NAME}
        LIBRARY
        DESTINATION ${CMAKE_SOURCE_DIR}/lib
)