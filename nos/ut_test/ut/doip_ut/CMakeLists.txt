#module名称
set(MODULE_NAME "doip_ut")

# 代码覆盖率报告编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")

link_directories(
    ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib
    ${CMAKE_SOURCE_DIR}/../third_party/${TARGET_PLATFORM}/gtest/lib
    ${CMAKE_SOURCE_DIR}/../third_party/${TARGET_PLATFORM}/jsoncpp/lib
)
#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB
        SOURCES
        ${CMAKE_SOURCE_DIR}/../ut_test/ut/doip_ut/*.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/config/doip_config.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/base/doip_util.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/base/doip_thread.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/base/doip_select.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/base/doip_netlink.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/base/doip_data_queue.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/base/doip_event_loop.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/base/doip_connection.cpp
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/src/socket/doip_socket_os.cpp
)


#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/../middleware
        ${CMAKE_SOURCE_DIR}/../middleware/diag/doip/include
        ${CMAKE_SOURCE_DIR}/../third_party/${TARGET_PLATFORM}/gtest/include
        ${CMAKE_SOURCE_DIR}/../third_party/${TARGET_PLATFORM}/jsoncpp/include
)


#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        jsoncpp
        netaos::log
        gtest
        pthread
)

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

