
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "testcase_DEVM_TEST_010")

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib)

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
    )

add_executable(${MODULE_NAME} ${SOURCES})

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        )

target_link_libraries(${MODULE_NAME}
        jsoncpp
        gtest
        )
install(
        TARGETS ${MODULE_NAME} RUNTIME
        DESTINATION ${TARGET_PLATFORM}/test/unit_test/devm_test_unit/${MODULE_NAME}/bin/
)
