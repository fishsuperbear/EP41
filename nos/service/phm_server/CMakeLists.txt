#module名称
set(MODULE_NAME "phm_server")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

if (${TARGET_PLATFORM} MATCHES "orin")
        set(INTERFACE  ${CMAKE_SOURCE_DIR}/service/phm_server/src/fault_manager/serviceInterface/*.cpp)
endif()

link_directories(
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
        ${NVLIB_PATH}
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib
        ${CMAKE_SOURCE_DIR}/service/phm_server/gen/soc_fault_gen/lib
        ${CMAKE_SOURCE_DIR}/service/phm_server/gen/mcu_fault_gen/lib
)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/gen/mcu_fault_gen/includes
        ${CMAKE_CURRENT_SOURCE_DIR}/gen/soc_fault_gen/includes
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/common/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/fault_lock/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/fault_manager/analysis/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/fault_manager/file/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/fault_manager/interactive/*.cpp
        ${INTERFACE}
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/fault_manager/manager/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/fault_manager/strategy/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/health_manager/manager/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/health_manager/manager/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/health_manager/monitor/*.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/src/phm_server.cpp
        ${CMAKE_SOURCE_DIR}/service/phm_server/main.cpp
        ${EPL_SRC}
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
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/include
        ${NETA_MIDDLEWARE_DIR}/idl/generated/
        ${NETA_MIDDLEWARE_DIR}/cm/include/
        ${NETA_MIDDLEWARE_DIR}/em/include/
        ${NETA_MIDDLEWARE_DIR}/log/include/
        ${NETA_MIDDLEWARE_DIR}/sm/include
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/service
        ${CMAKE_SOURCE_DIR}/middleware/per/struct2x/include
        ${CMAKE_SOURCE_DIR}/middleware/cfg/include
        )

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
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libssl.so.1.1
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/iauto_someip/lib/libcrypto.so.1.1
        McuFaultService
        SocFaultService
        -Wl,--end-group
)
endif()

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        ${NVLIB_IVC}
        ${NVLIB_FSICOM}
        neta_log
        neta_cm
        neta_sm_client
        neta_em_client
        neta_per
        neta_cfg
        ${IAUTO_LIB}
        )


set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

# add someip gen lib
install(DIRECTORY ${CMAKE_SOURCE_DIR}/service/phm_server/gen/mcu_fault_gen/lib/ DESTINATION ${TARGET_PLATFORM}/lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/service/phm_server/gen/soc_fault_gen/lib/ DESTINATION ${TARGET_PLATFORM}/lib)

file(GLOB CONF_FILES  "${CMAKE_SOURCE_DIR}/service/phm_server/conf/*")
file(COPY ${CONF_FILES}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)

if (${TARGET_PLATFORM} MATCHES "orin")
        file(GLOB THIRD_PARTY_FILES ${CMAKE_SOURCE_DIR}/service/phm_server/bin/${TARGET_PLATFORM}/*)
        install(FILES ${THIRD_PARTY_FILES}
                DESTINATION ${TARGET_PLATFORM}/bin
        )
endif()
