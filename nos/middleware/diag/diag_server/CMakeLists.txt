#module名称
set(MODULE_NAME "diag_server")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

if (${TARGET_PLATFORM} MATCHES "orin")
        set(NVLIB_PATH ${COMPILER_PATH}/../usr/lib/)
        set(NVLIB_IVC nvivc)
        set(NVLIB_FSICOM NvFsiCom)
        set(INTERFACE  ${CMAKE_CURRENT_SOURCE_DIR}/src/publish/*.cpp)
endif()

if(ENABLE_SOMEIP)
        link_directories(
                ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/lib
                ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/lib
                ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/lib
        )
endif()


if(ENABLE_IPC)
        # 使用IPC 通讯
        file(GLOB_RECURSE
                SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/datatransfer/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/event_manager/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/info/*.cpp
                ${INTERFACE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/security/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/service/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/session/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/ipc_transport/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/uds_transport/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/diag_server_transport_service.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/diag_server_transport.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/uds/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/diag_server.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp

                ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/phm.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/phmPubSubTypes.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/phmTypeObject.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/chassis_ota_method.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/chassis_ota_methodPubSubTypes.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/chassis_ota_methodTypeObject.cxx
        )

        #默认编译c和c++文件
        add_executable(${MODULE_NAME} ${SOURCES})

        #用户可在此添加头文件路径
        target_include_directories(${MODULE_NAME}
                PRIVATE
                ${CMAKE_SOURCE_DIR}/middleware/diag/diag_server/include/transport/ipc_transport
                )
else()
        # 使用CM 通讯
        file(GLOB_RECURSE
                SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/datatransfer/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/event_manager/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/info/*.cpp
                ${INTERFACE}
                ${CMAKE_CURRENT_SOURCE_DIR}/src/security/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/service/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/session/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/cm_transport/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/uds_transport/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/diag_server_transport_service.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/transport/diag_server_transport.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/uds/*.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/src/diag_server.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp

                ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/phm.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/phmPubSubTypes.cxx
                ${NETA_MIDDLEWARE_DIR}/idl/generated/phmTypeObject.cxx
        )

        #默认编译c和c++文件
        add_executable(${MODULE_NAME} ${SOURCES})

        #用户可在此添加头文件路径
        target_include_directories(${MODULE_NAME}
                PRIVATE
                ${CMAKE_SOURCE_DIR}/middleware/diag/diag_server/include/transport/cm_transport
        )
endif()

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cpp-ipc/include
        ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/include
        ${CMAKE_CURRENT_SOURCE_DIR}/uds_gen/includes
)

target_link_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib
        ${NVLIB_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}/uds_gen/lib
)

if(ENABLE_SOMEIP)
        target_include_directories(${MODULE_NAME}
                PRIVATE
                ${CMAKE_SOURCE_DIR}/middleware/diag/dosomeip/someip/src-gen
                ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/include
                ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/include
                ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/include
        )
endif()

if (${TARGET_PLATFORM} MATCHES "orin")
set(IAUTO_LIB
        -Wl,--start-group
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
        pthread
        pthread
        ssl
        crypto
        SoCUdsService
        -Wl,--end-group
)
endif()

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        dl
        pthread
        ${NVLIB_IVC}
        ${NVLIB_FSICOM}
        neta_diag_common
        neta_doip
        neta_docan
        neta_diag_sa
        neta_log
        neta_cm
        neta_em_client
        sqlite3
        neta_cfg
        neta_ipc
        neta_devm_client
        ${IAUTO_LIB}
        )

if(ENABLE_SOMEIP)
        target_link_libraries(${MODULE_NAME}
                PRIVATE
                neta_dosomeip
                CommonAPI
                CommonAPI-SomeIP
                vsomeip3
        )
endif()


set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

file(COPY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/lib
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)

file(GLOB CONF_FILES  "${CMAKE_SOURCE_DIR}/middleware/diag/conf/${TARGET_PLATFORM}/*")
file(COPY ${CONF_FILES}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)

# add someip gen lib
install(DIRECTORY ${CMAKE_SOURCE_DIR}/middleware/diag/diag_server/uds_gen/lib/ DESTINATION ${TARGET_PLATFORM}/lib)

#TO DELETE: Just for test
set(LIB_JSON_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib")
file(COPY ${LIB_JSON_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)

set(LIB_FASTDDS_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib")
file(COPY ${LIB_FASTDDS_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)

set(LIB_TINYXML2_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib")
file(COPY ${LIB_TINYXML2_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)

set(LIB_UUID_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/lib")
file(COPY ${LIB_UUID_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)


if(ENABLE_SOMEIP)
        set(LIB_BOOST_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/lib")
        file(COPY ${LIB_BOOST_PATH}
                DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
        )

        set(LIB_COMAPI_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/lib")
        file(COPY ${LIB_COMAPI_PATH}
                DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
        )

        set(LIB_COMAPI_SOMEIP_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/lib")
        file(COPY ${LIB_COMAPI_SOMEIP_PATH}
                DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
        )

        set(LIB_VSOMIP_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/lib")
        file(COPY ${LIB_VSOMIP_PATH}
                DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
        )
endif()

#TO DELETE: Just for test
