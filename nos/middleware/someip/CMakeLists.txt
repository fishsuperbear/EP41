#module名称
set(MODULE_NAME "neta_someip")
#lib名称
set(LIB_NAME neta_someip)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g ")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

#查找目录下所有的头文件
function(include_headerfiles root_dir)
    if (IS_DIRECTORY ${root_dir})
        include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            include_headerfiles(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

#用户可在此添加头文件路径
include_directories(${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_headerfiles(${CMAKE_CURRENT_SOURCE_DIR}/base)
include_headerfiles(${CMAKE_CURRENT_SOURCE_DIR}/extend)
include_headerfiles(${CMAKE_CURRENT_SOURCE_DIR}/protocol)


#用户可在此添加头文件路径
#target_include_directories(${MODULE_NAME}
#        PRIVATE
#        ${CMAKE_SOURCE_DIR}/middleware
#        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include
#        )

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
    SOURCES
  # ${CMAKE_CURRENT_SOURCE_DIR}/base/bufferlist/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/config/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/file/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/list/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/log/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/looper/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/map/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/transmit/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/base/sync/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/extend/e2e/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/extend/network/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/extend/tp/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/app_context/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/endpoint/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/init/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/ipc/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/message/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/route_data/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/rpc/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/sd/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/protocol/serializer/*.cpp
    )


#默认编译c和c++文件
add_library(${MODULE_NAME} SHARED ${SOURCES})

#用户可在此添加链接库目录
target_link_directories(${MODULE_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64
    )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libssl.so.3
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libcrypto.so.3
        )


set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")


install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)



#####################################################

#可执行程序名称
set(EXECUTABLE_NAME "neta_someipd")

#用户可在此添加头文件路径
#target_include_directories(${EXECUTABLE_NAME}
#        PRIVATE
#        ${CMAKE_SOURCE_DIR}/middleware
#        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include
#        )

#用户可在此添加头文件路径
include_directories(${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/ne_someip_daemon_main.cpp
    )


#用户可在此添加链接库目录
link_directories(
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

#默认编译c和c++文件
add_executable(${EXECUTABLE_NAME} ${SOURCES})

#用户可在此添加链接库
target_link_libraries(${EXECUTABLE_NAME}
        PRIVATE
        pthread
        neta_someip
        neta_cm
        neta_em_client
        )

install(TARGETS ${EXECUTABLE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

set(SOMEIPD_CONFIG_PATH "${CMAKE_SOURCE_DIR}/middleware/someip/conf")


file(COPY ${SOMEIPD_CONFIG_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)