#module名称
set(MODULE_NAME "ut_helloworld")
add_definitions(-DUT=)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib/
)
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
        )

target_link_libraries(${MODULE_NAME}
        PRIVATE
        gtest
        gtest_main
        )
#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/example/ut_helloworld
)

