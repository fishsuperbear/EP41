#module名称
set(MODULE_NAME "update_manager")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")


link_directories(
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/swm_proxy/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

if (${TARGET_PLATFORM} MATCHES "orin")
        if(ENABLE_IPC)
                # orin 平台 且 使用IPC 通讯
                file(GLOB_RECURSE
                        SOURCES
                        ${CMAKE_SOURCE_DIR}/service/update_manager/common/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/config/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/download/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/file_to_bin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/ipc/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/manager/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/state_machine/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/taskbase/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/update_check/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/upload/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/record/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/cmd_line_upgrade/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/agent/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/mcu/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/main.cpp
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devm.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmTypeObject.cxx
                )
                add_executable(${MODULE_NAME} ${SOURCES})

                target_include_directories(${MODULE_NAME}
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/ipc

                )
                target_link_libraries(${MODULE_NAME}
                        PRIVATE
                        neta_ipc
                )
        else()
                # orin 平台 且 使用CM 通讯
                file(GLOB_RECURSE
                        SOURCES
                        ${CMAKE_SOURCE_DIR}/service/update_manager/common/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/config/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/download/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/file_to_bin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/cm/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/manager/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/state_machine/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/taskbase/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/update_check/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/upload/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/record/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/cmd_line_upgrade/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/agent/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/mcu/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/main.cpp
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devm.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmTypeObject.cxx
                )
                add_executable(${MODULE_NAME} ${SOURCES})

                target_include_directories(${MODULE_NAME}
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/cm

                )
        endif()

elseif(${TARGET_PLATFORM} MATCHES "x86_2004")
        if(ENABLE_IPC)
                # orin 平台 且 使用IPC 通讯
                file(GLOB_RECURSE
                        SOURCES
                        ${CMAKE_SOURCE_DIR}/service/update_manager/common/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/config/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/download/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/file_to_bin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin/interface_update_req_dispatcher.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/ipc/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/manager/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/state_machine/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/taskbase/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/update_check/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/upload/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/record/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/cmd_line_upgrade/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/agent/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/mcu/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/main.cpp
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devm.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmTypeObject.cxx
                )
                add_executable(${MODULE_NAME} ${SOURCES})

                target_include_directories(${MODULE_NAME}
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/ipc

                )
                target_link_libraries(${MODULE_NAME}
                        PRIVATE
                        neta_ipc
                )
        else()
                # orin 平台 且 使用CM 通讯
                file(GLOB_RECURSE
                        SOURCES
                        ${CMAKE_SOURCE_DIR}/service/update_manager/common/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/config/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/download/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/file_to_bin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin/interface_update_req_dispatcher.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/cm/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/manager/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/state_machine/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/taskbase/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/update_check/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/upload/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/record/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/cmd_line_upgrade/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/agent/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/mcu/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/main.cpp
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devm.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmTypeObject.cxx
                )
                add_executable(${MODULE_NAME} ${SOURCES})

                target_include_directories(${MODULE_NAME}
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/orin
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/cm

                )
        endif()
else()
        
        if(ENABLE_IPC)
                #  非 orin 平台 且 使用IPC 通讯
                file(GLOB_RECURSE
                        SOURCES
                        ${CMAKE_SOURCE_DIR}/service/update_manager/common/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/config/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/download/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/file_to_bin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/mdc/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/mdc/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/ipc/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/manager/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/state_machine/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/taskbase/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/update_check/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/upload/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/record/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/cmd_line_upgrade/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/agent/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/mcu/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/main.cpp
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devm.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmTypeObject.cxx
                )
                add_executable(${MODULE_NAME} ${SOURCES})

                target_include_directories(${MODULE_NAME}
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/mdc
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/mdc
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/ipc

                )
                target_link_libraries(${MODULE_NAME}
                        PRIVATE
                        neta_ipc
                )

        else()
                # 非 orin 平台 且 使用CM 通讯
                file(GLOB_RECURSE
                        SOURCES
                        ${CMAKE_SOURCE_DIR}/service/update_manager/common/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/config/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/download/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/file_to_bin/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/mdc/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/mdc/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/cm/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/manager/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/state_machine/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/taskbase/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/update_check/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/upload/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/record/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/cmd_line_upgrade/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/agent/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/mcu/*.cpp
                        ${CMAKE_SOURCE_DIR}/service/update_manager/main.cpp
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devm.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmPubSubTypes.cxx
                        ${NETA_MIDDLEWARE_DIR}/idl/generated/devmTypeObject.cxx
                )
                add_executable(${MODULE_NAME} ${SOURCES})

                target_include_directories(${MODULE_NAME}
                        PRIVATE
                        ${CMAKE_SOURCE_DIR}/service/update_manager/installer/mdc
                        ${CMAKE_SOURCE_DIR}/service/update_manager/ota_api/mdc
                        ${CMAKE_SOURCE_DIR}/service/update_manager/transport/cm

                )
        endif()


endif()

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/service
        ${CMAKE_SOURCE_DIR}/middleware/https/
        ${CMAKE_SOURCE_DIR}/middleware/https/include/
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/common
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/cryp/
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/x509/
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/keys
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/cryp/cryobj/
        ${CMAKE_SOURCE_DIR}/middleware/crypto/include/utility/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/boost/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cpp-ipc/include/
        ${CMAKE_SOURCE_DIR}/middleware/lrzip-0.631
        #${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lrzip/include
)   

if (${TARGET_PLATFORM} MATCHES "x86_2004" OR ${TARGET_PLATFORM} MATCHES "orin")
    set(CAMERA_LIBRARY
        neta_lrzip
        #lzo2
        #lz4
        )
endif()
#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        neta_log
        neta_cm
        neta_em_client
        neta_sttask
        neta_https
        neta_sm_client
        neta_cfg
        neta_zmq_ipc
        ${CAMERA_LIBRARY}
        )

if (${TARGET_PLATFORM} MATCHES "mdc" OR ${TARGET_PLATFORM} MATCHES "mdc-llvm")
        target_include_directories(${MODULE_NAME}
                PRIVATE
                ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/swm_proxy/include
        )

        target_link_libraries(${MODULE_NAME}
                PRIVATE
                swm_proxy
        )

        file(COPY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/swm_proxy/lib/libswm_proxy.so DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib)
        file(COPY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/swm_proxy/etc/SWMProxyProcess DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)
endif()

if (${TARGET_PLATFORM} MATCHES "orin")
        target_link_libraries(${MODULE_NAME}
                PRIVATE
                pps_com
        )
endif()

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

file(GLOB CONF_FILES  "${CMAKE_SOURCE_DIR}/service/update_manager/conf/${TARGET_PLATFORM}/*")
file(COPY ${CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)
