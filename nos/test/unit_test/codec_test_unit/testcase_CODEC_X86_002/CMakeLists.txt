if(${TARGET_PLATFORM} MATCHES "x86_2004")

get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(MODULE_NAME ${CURRENT_DIR_NAME})

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

target_link_libraries(${MODULE_NAME}
    gtest
    gtest_main
)

#用户可在此添加链接参数
# set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/${MAIN_DIR_NAME}/${CURRENT_DIR_NAME}/bin
)

endif()
