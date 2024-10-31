#module名称
set(MODULE_NAME "neta_dosomeip")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

#默认编译c和c++文件
add_library(${MODULE_NAME} SHARED ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/someip/src
        ${CMAKE_CURRENT_SOURCE_DIR}/someip/src-gen
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/include
)

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        neta_diag_common
        neta_log
        jsoncpp
        CommonAPI 
        CommonAPI-SomeIP 
        vsomeip3
)

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

file(GLOB CONF_FILES  "${CMAKE_SOURCE_DIR}/middleware/diag/dosomeip/someip/etc/*")
file(COPY ${CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)

file(GLOB CONF_FILES  "${CMAKE_SOURCE_DIR}/middleware/diag/dosomeip/etc/*")
file(COPY ${CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)
