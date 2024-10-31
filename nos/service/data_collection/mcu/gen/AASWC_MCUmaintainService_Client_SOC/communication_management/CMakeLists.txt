CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

if(${USER_TARGET_OS} STREQUAL "qnx")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -Wall")
elseif(${USER_TARGET_OS} STREQUAL "linux")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -pthread -Wall")
elseif(${USER_TARGET_OS} STREQUAL "droid")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14 -pthread -Wall")
endif()

include_directories(
.
${CMAKE_CURRENT_SOURCE_DIR}/includes
)

#########################################################################
# lib: libswc_IAUTOSAR_AASWC_MCUmaintainService_Client_SOC.so
#########################################################################
set(AASWC_MCUMAINTAINSERVICE_CLIENT_SOCSWC_DEPLOYMENT_LIBRARY_NAME swc_IAUTOSAR_AASWC_MCUmaintainService_Client_SOC)
set(AASWC_MCUMAINTAINSERVICE_CLIENT_SOCSWC_DEPLOYMENT_LIBRARY_SRC_FILES
src/hozon/netaos/v1/mcumaintainservice_proxy.cpp
src/hozon/netaos/v1/mcumaintainservice_manifest.cpp
)

add_library(${AASWC_MCUMAINTAINSERVICE_CLIENT_SOCSWC_DEPLOYMENT_LIBRARY_NAME} SHARED ${AASWC_MCUMAINTAINSERVICE_CLIENT_SOCSWC_DEPLOYMENT_LIBRARY_SRC_FILES})
target_link_libraries(${AASWC_MCUMAINTAINSERVICE_CLIENT_SOCSWC_DEPLOYMENT_LIBRARY_NAME} aracore_ics ara_com_ics ara_someipbinding_ics ara_serialize_ics )


install(TARGETS ${AASWC_MCUMAINTAINSERVICE_CLIENT_SOCSWC_DEPLOYMENT_LIBRARY_NAME}
        LIBRARY
        DESTINATION /lib/
)