#module名称
set(MODULE_NAME "um_check_test")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

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
        ${CMAKE_SOURCE_DIR}/../service/update_manager/common/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/config/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/download/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/file_to_bin/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/installer/orin/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/ota_api/orin/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/transport/cm/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/manager/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/state_machine/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/taskbase/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/update_check/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/upload/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/record/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/cmd_line_upgrade/*.cpp
        ${CMAKE_SOURCE_DIR}/../service/update_manager/agent/*.cpp
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/../middleware
        ${CMAKE_SOURCE_DIR}/../service
        ${CMAKE_SOURCE_DIR}/../service/update_manager
        ${CMAKE_SOURCE_DIR}/../service/update_manager/installer/orin
        ${CMAKE_SOURCE_DIR}/../service/update_manager/ota_api/orin
        ${CMAKE_SOURCE_DIR}/../service/update_manager/transport/cm
        ${CMAKE_SOURCE_DIR}/../middleware/https/
        ${CMAKE_SOURCE_DIR}/../middleware/https/include/
        ${CMAKE_SOURCE_DIR}/../middleware/crypto/include/
        ${CMAKE_SOURCE_DIR}/../middleware/crypto/include/common
        ${CMAKE_SOURCE_DIR}/../middleware/crypto/include/cryp/
        ${CMAKE_SOURCE_DIR}/../middleware/crypto/include/x509/
        ${CMAKE_SOURCE_DIR}/../middleware/crypto/include/keys
        ${CMAKE_SOURCE_DIR}/../middleware/crypto/include/cryp/cryobj/
        ${CMAKE_SOURCE_DIR}/../middleware/crypto/include/utility/
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/curl/include
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/zipper/include
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/boost/include
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include
        ${CMAKE_SOURCE_DIR}/../netaos_thirdparty/${TARGET_PLATFORM}/cpp-ipc/include/
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        netaos::log
        netaos::cm
        netaos::em
        netaos::sm
        neta_sttask
        neta_securehttp
        netaos::cfg
        gtest
        gtest_main
        )

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed,--disable-new-dtags")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/test/unit_test/um_test_unit/bin/
)
