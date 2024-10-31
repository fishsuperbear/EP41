#module名称
set(MODULE_NAME "avp_map_manage_test")

# set(EXECUTABLE_OUTPUT_PATH ${PROJECT_PATH}/output/${TARGET_PLATFORM}/bin)

#默认编译c和c++文件
add_executable(${MODULE_NAME}
        test.cpp
)

#头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/include
)

#指定库的路径 
target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}
)

#添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        avp_map_manage
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)