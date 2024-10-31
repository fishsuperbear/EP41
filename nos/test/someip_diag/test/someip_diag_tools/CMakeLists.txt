set(SOMEIP_TOOLS_NAME "someip_diag_tools")

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
)

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

#默认编译c和c++文件
add_executable(${SOMEIP_TOOLS_NAME} 
        ${SOURCES}
        ../../src-gen/v1/commonapi/DoSomeIPSomeIPProxy.cpp
	../../src-gen/v1/commonapi/DoSomeIPSomeIPDeployment.cpp
        )

target_include_directories(${SOMEIP_TOOLS_NAME}
    PRIVATE
    ${PRJ_ROOT_DIR}/middleware
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
)

target_link_libraries(${SOMEIP_TOOLS_NAME}
        PRIVATE
        CommonAPI 
        CommonAPI-SomeIP 
        vsomeip3
        jsoncpp
        )

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${SOMEIP_TOOLS_NAME} PRIVATE -Wall)

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${SOMEIP_TOOLS_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${SOMEIP_TOOLS_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/tools/someip_diag_tools/
)

set(COMPLETE_FILE_PATH "${CMAKE_SOURCE_DIR}/")
file(COPY ${COMPLETE_FILE_PATH}/someip_diag/test/someip_diag_tools/someip_tools.json
        DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/tools/someip_diag_tools/  FOLLOW_SYMLINK_CHAIN
)

