#module名称
set(MODULE_NAME "neta_log")
#lib名称
set(LIB_NAME neta_log)

# add_definitions("-lpthread")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(APPEND filters "")
set(SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} "${filters}")

if(ENABLE_LOG_SERVER)
    add_definitions(-DBUILD_LOG_SERVER_ENABLE)
endif()

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

set(LIB_ZIPPER_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/lib")
file(COPY ${LIB_ZIPPER_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)


#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

if (${TARGET_PLATFORM} MATCHES "orin")
#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${SRC_DIRS}
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/spdlog/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/logblock/include
)

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/logblock/lib/
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        neta_zmq_ipc
        jsoncpp
        zipper
        neta_logblock_helper
        logblockreader
        logblockwriter
        hw_platform
        )
else()
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${SRC_DIRS}
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/spdlog/include
        )

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/lib/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        neta_zmq_ipc
        jsoncpp
        zipper
        )
endif()

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${TARGET_PLATFORM}/include/log/include)



