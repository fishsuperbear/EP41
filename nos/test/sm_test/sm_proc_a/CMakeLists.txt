#module名称
set(MODULE_NAME "sm_proc_a")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

link_directories(${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_SOURCE_DIR}/middleware/em/utils/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/middleware/em/include
        ${PRJ_ROOT_DIR}/middleware/em/utils
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        netaos::log
        netaos::cm
        netaos::sm
        netaos::em
        dl
        rt
        )

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/sm_test_unit/sm_test_common/emproc/sm_proc_a/bin/
)

set(EM_CONFIG_FILE_PATH "${CMAKE_SOURCE_DIR}/sm_test/sm_proc_a/etc")
file(COPY ${EM_CONFIG_FILE_PATH}
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/test/unit_test/sm_test_unit/sm_test_common/emproc/sm_proc_a  FOLLOW_SYMLINK_CHAIN
)
