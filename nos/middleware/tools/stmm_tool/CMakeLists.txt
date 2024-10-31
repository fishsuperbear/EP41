#module名称
set(TOOL_MODULE_NAME "stmm")
set(INFO_MODULE_NAME "stmm_info")

set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        TOOL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/system_monitor_info.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/stmm_tool.cpp
        ${CMAKE_SOURCE_DIR}/service/idl/generated/monitor.cxx
        ${CMAKE_SOURCE_DIR}/service/idl/generated/monitorPubSubTypes.cxx
        ${CMAKE_SOURCE_DIR}/service/idl/generated/monitorTypeObject.cxx
    )

file(GLOB_RECURSE
        INFO_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/system_monitor_info.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/stmm_info.cpp
    )

link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib)

#默认编译c和c++文件
add_executable(${TOOL_MODULE_NAME} ${TOOL_SOURCES})
add_executable(${INFO_MODULE_NAME} ${INFO_SOURCES})

#用户可在此添加头文件路径
target_include_directories(${TOOL_MODULE_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/devm/include/common
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include
        )

target_include_directories(${INFO_MODULE_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/devm/include/common
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        )

#用户可在此添加链接库
target_link_libraries(${TOOL_MODULE_NAME}
        PRIVATE
        neta_cm
        neta_cfg
        neta_devm_client
        )

#用户可在此添加链接库
target_link_libraries(${INFO_MODULE_NAME}
        PRIVATE
        jsoncpp
        neta_devm_client
        )

install(
    TARGETS ${TOOL_MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/
)

install(
    TARGETS ${INFO_MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/
)
