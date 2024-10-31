#module名称
set(MODULE_NAME "hz_log_test")

# add_definitions("-lpthread")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")


list(APPEND filters "")
set(SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} "${filters}")

set(CMAKE_INSTALL_RPATH "${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        HZ_LOG_TEST_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/hz_log_test.cpp)

link_directories(${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cpp-ipc/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
        )


set(UNIT_TEST_NAME "hz_log_unit_test")
add_executable(${UNIT_TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/hz_log_unit_test.cpp)
target_compile_options(${UNIT_TEST_NAME} PRIVATE -Wall)
#用户可在此添加头文件路径
target_include_directories(${UNIT_TEST_NAME}
        PRIVATE
        ${SRC_DIRS}
        ${PRJ_ROOT_DIR}/middleware/log/include
        )
#用户可在此添加链接库
target_link_libraries(${UNIT_TEST_NAME}
        PRIVATE
        netaos::log
        )
install(
        TARGETS ${UNIT_TEST_NAME} RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin/hz_log_test/
)

set(MODULE_NAME_1W "hz_log_performance_test")
add_executable(${MODULE_NAME_1W} ${CMAKE_CURRENT_SOURCE_DIR}/hz_log_performance_test.cpp)
target_compile_options(${MODULE_NAME_1W} PRIVATE -Wall)
#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME_1W}
        PRIVATE
        ${SRC_DIRS}
        ${PRJ_ROOT_DIR}/middleware/log/include
        ${PRJ_ROOT_DIR}/middleware/logblock_helper/include
        )
#用户可在此添加链接库
target_link_libraries(${MODULE_NAME_1W}
        PRIVATE
        netaos::log
        )
install(
        TARGETS ${MODULE_NAME_1W} RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin/hz_log_test/
)

set(OPERATION_TEST_NAME "log_server_operation_test")
add_executable(${OPERATION_TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/log_server_operation_test.cpp)
target_compile_options(${OPERATION_TEST_NAME} PRIVATE -Wall)
#用户可在此添加头文件路径
target_include_directories(${OPERATION_TEST_NAME}
        PRIVATE
        ${SRC_DIRS}
        ${PRJ_ROOT_DIR}/middleware/log/include
        )
#用户可在此添加链接库
target_link_libraries(${OPERATION_TEST_NAME}
        PRIVATE
        netaos::log
        )
install(
        TARGETS ${OPERATION_TEST_NAME} RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin/hz_log_test/
)


set(COMPRESS_TEST_NAME "log_server_compress_test")
add_executable(${COMPRESS_TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/log_server_compress_test.cpp)
target_compile_options(${COMPRESS_TEST_NAME} PRIVATE -Wall)
#用户可在此添加头文件路径
target_include_directories(${COMPRESS_TEST_NAME}
        PRIVATE
        ${SRC_DIRS}
        ${PRJ_ROOT_DIR}/middleware/log/include
        )
#用户可在此添加链接库
target_link_libraries(${COMPRESS_TEST_NAME}
        PRIVATE
        netaos::log
        )
install(
        TARGETS ${COMPRESS_TEST_NAME} RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin/hz_log_test/
)

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${HZ_LOG_TEST_SOURCES})


#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${SRC_DIRS}
        ${PRJ_ROOT_DIR}/middleware/log/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        netaos::log
        )


#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")


#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/hz_log_test/
)

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${HZ_LOG_TOOLS_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/tools/hz_log_tools/
)


set(LOG_CFG_FILE_PATH "${CMAKE_SOURCE_DIR}/hz_log_test/log_cfg.json")
file(COPY ${LOG_CFG_FILE_PATH}
        DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/bin/hz_log_test/etc/  FOLLOW_SYMLINK_CHAIN
)
