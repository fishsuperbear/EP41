#module名称
set(MODULE_NAME "neta_diag_agent")

#lib名称
set(LIB_NAME neta_diag_agent)

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
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/diag.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/diagTypeObject.cxx
    )

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        neta_cm
        neta_log
        )

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/common/diag_agent_def.h
DESTINATION ${TARGET_PLATFORM}/include/diag/diag_agent/include/common)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/handler/
DESTINATION ${TARGET_PLATFORM}/include/diag/diag_agent/include/handler)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/service/
DESTINATION ${TARGET_PLATFORM}/include/diag/diag_agent/include/service)