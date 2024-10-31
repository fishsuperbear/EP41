#module名称
set(MODULE_NAME "system_monitor")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

link_directories(
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${NETA_SERVICE_DIR}/idl/generated/monitor.cxx
        ${NETA_SERVICE_DIR}/idl/generated/monitorPubSubTypes.cxx
        ${NETA_SERVICE_DIR}/idl/generated/monitorTypeObject.cxx
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/service
        ${CMAKE_SOURCE_DIR}
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        neta_log
        neta_cm
        neta_phm
        neta_cfg
        neta_devm_client
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

file(GLOB CONF_FILES  "${CMAKE_SOURCE_DIR}/service/system_monitor/conf/*")
file(COPY ${CONF_FILES}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)

if (${TARGET_PLATFORM} MATCHES "orin")
        file(GLOB THIRD_PARTY_FILES ${CMAKE_SOURCE_DIR}/service/system_monitor/bin/${TARGET_PLATFORM}/*)
        install(FILES ${THIRD_PARTY_FILES}
                DESTINATION ${TARGET_PLATFORM}/bin
        )
endif()