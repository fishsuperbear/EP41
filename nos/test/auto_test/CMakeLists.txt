#module名称
set(MODULE_NAME "auto_test")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(DEFAULT_COMPILE -std=c++14 -O0 -g -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

message("${CMAKE_CURRENT_SOURCE_DIR}")
#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/Common/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestModule/uds/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestModule/docan/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestModule/doip/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestModule/dosomeip/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/TestModule/phm/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/testmain.cpp
        )

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include
        ${PRJ_ROOT_DIR}/test/auto_test/Common
        ${PRJ_ROOT_DIR}/test/auto_test/TestModule/uds
        ${PRJ_ROOT_DIR}/test/auto_test/TestModule/docan
        ${PRJ_ROOT_DIR}/test/auto_test/TestModule/doip
        ${PRJ_ROOT_DIR}/test/auto_test/TestModule/dosomeip
        ${PRJ_ROOT_DIR}/test/auto_test/TestModule/phm
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        netaos::log
        netaos::cm
        netaos::phm
        netaos::sm
        neta_diag_sa
        )

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/auto_test/bin
)
install(DIRECTORY ${PRJ_ROOT_DIR}/test/auto_test/Conf
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/auto_test
)
install(DIRECTORY ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/auto_test
)
install(FILES ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_diag_sa.so
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/auto_test/lib
)
install(FILES ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_log.so
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/auto_test/lib
)
install(DIRECTORY ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/auto_test
)
install(DIRECTORY ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/auto_test
)





