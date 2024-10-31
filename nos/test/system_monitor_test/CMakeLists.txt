#module名称
set(MODULE_NAME "system_monitor_test")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${PRJ_ROOT_DIR}/service/idl/generated/monitor.cxx
        ${PRJ_ROOT_DIR}/service/idl/generated/monitorPubSubTypes.cxx
        ${PRJ_ROOT_DIR}/service/idl/generated/monitorTypeObject.cxx
        # ${PRJ_ROOT_DIR}/proto/soc/system_monitor.pb.cc
    )

link_directories(
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib)

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/service
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        netaos::log
        fastcdr
        fastrtps
        neta_cm
        foonathan_memory-0.7.1
        tinyxml2
        dl
        rt
        uuid
        globalproto
        protobuf
        netaos::cfg
        )

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)