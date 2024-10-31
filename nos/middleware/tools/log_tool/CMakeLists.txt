set(HZ_LOG_TOOLS_NAME "log")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

list(APPEND filters "")
set(SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} "${filters}")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

link_directories(
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SET_LOG_LEVEL_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/hz_log_tools.cpp)

#默认编译c和c++文件
add_executable(${HZ_LOG_TOOLS_NAME} ${SET_LOG_LEVEL_SOURCES})

target_include_directories(${HZ_LOG_TOOLS_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zipper/include
)

target_link_libraries(${HZ_LOG_TOOLS_NAME}
        PUBLIC
        pthread
        zipper
        )

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${HZ_LOG_TOOLS_NAME} PRIVATE -Wall)

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${HZ_LOG_TOOLS_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")


#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${HZ_LOG_TOOLS_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/
)