#module名称
set(MODULE_NAME "e2e_test_001")
add_definitions(-DUT=)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib/
)


#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/middleware/crc/include
        ${PRJ_ROOT_DIR}/middleware/e2e/e2e/include
        ${PRJ_ROOT_DIR}/middleware/e2e/e2exf/include
        ${PRJ_ROOT_DIR}/middleware/e2e/e2exf_cpp/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        neta_crc
        neta_e2e
        neta_e2exf
        neta_e2exf_cpp
        gtest
        gtest_main
        )

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed,--disable-new-dtags")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/e2e_test_unit/e2e_test_001/bin/
)

# set(SCRIPTS_PATH ${CMAKE_SOURCE_DIR}/unit_test/e2e_test_unit/e2e_test_001/scripts)
# set(CONFIG_PATH ${CMAKE_SOURCE_DIR}/unit_test/e2e_test_unit/e2e_test_001/conf)
file(COPY ${SCRIPTS_PATH} ${CONFIG_PATH}
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/test/unit_test/e2e_test_unit/e2e_test_001)