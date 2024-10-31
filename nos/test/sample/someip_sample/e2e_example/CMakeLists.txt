cmake_minimum_required(VERSION 2.8)

set(MODULE_NAME "provide_demo")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "PRJ_ROOT_DIR: ${PRJ_ROOT_DIR}")

# #查找目录下所有的头文件
# function(include_headerfiles root_dir)
#     if (IS_DIRECTORY ${root_dir})
#         include_directories(${root_dir})
#     endif()
#
#     file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
#     foreach(sub ${ALL_SUB})
#         if (IS_DIRECTORY ${root_dir}/${sub})
#             include_headerfiles(${root_dir}/${sub})
#         endif()
#     endforeach()
# endfunction()
#
# #用户可在此添加头文件路径
# include_directories(${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/include)
# include_directories(${PRJ_ROOT_DIR}/middleware/someip/include)
# include_headerfiles(${PRJ_ROOT_DIR}/middleware/someip/base)
# include_headerfiles(${PRJ_ROOT_DIR}/middleware/someip/extend)
# include_headerfiles(${PRJ_ROOT_DIR}/middleware/someip/protocol)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/someip_provide_usage_demo.cpp
    )

add_executable(${MODULE_NAME}
    ${SOURCES}
)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/
        ${PRJ_ROOT_DIR}/middleware/someip/include
        )


#用户可在此添加链接库目录
link_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)

target_link_libraries(${MODULE_NAME}
    pthread
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libneta_someip.so
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libssl.so.3
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/openssl-3.0.2/lib64/libcrypto.so.3
)


#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
