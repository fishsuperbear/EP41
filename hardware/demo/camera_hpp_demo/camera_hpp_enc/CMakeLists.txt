
cmake_minimum_required(VERSION 3.5.1)
# include("/drive/env.cmake")

project(camera_hpp_enc VERSION 1.0.0)


#自动查找头文件路径函数(没有去重)
macro(FIND_INCLUDE_DIR result curdir)  #定义函数,2个参数:存放结果result；指定路径curdir；
    file(GLOB_RECURSE children "${curdir}/*.hpp" "${curdir}/*.h" )  #遍历获取{curdir}中*.hpp和*.h文件列表
    # message(STATUS "children= ${children}")                               #打印*.hpp和*.h的文件列表
    set(dirlist "")                                                     #定义dirlist中间变量，并初始化
    foreach(child ${children})                                          #for循环
        string(REGEX REPLACE "(.*)/.*" "\\1" LIB_NAME ${child})         #字符串替换,用/前的字符替换/*h
        if(IS_DIRECTORY ${LIB_NAME})                                    #判断是否为路径
            LIST(APPEND dirlist ${LIB_NAME})                            #将合法的路径加入dirlist变量中
        endif()                                                         #结束判断
    endforeach()                                                        #结束for循环
    set(${result} ${dirlist})                                           #dirlist结果放入result变量中
endmacro()                                                              #函数结束

FIND_INCLUDE_DIR(INCLUDE_VIDEO_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../../hal/code/interface)
# FIND_INCLUDE_DIR(INCLUDE_GLOBAL_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/porting/)
# FIND_INCLUDE_DIR(INCLUDE_IMPL_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../hal/code/hal_impl)


set(LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../hal/code/lib)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address")

include_directories(
    # ${CMAKE_CURRENT_SOURCE_DIR}/../../../hal/code/interface/hpp/intf_camera/1.0
    ${INCLUDE_VIDEO_DIR_LIST}
    # ${INCLUDE_GLOBAL_DIR_LIST}
    # ${INCLUDE_IMPL_DIR_LIST}
)

file(GLOB_RECURSE HW_HAL_VS_STCS
    "${CMAKE_CURRENT_SOURCE_DIR}/camera_hpp_enc/enctest.cpp"
    )

#link directorie of lib
link_directories(
    ${LIB_DIR}
    /usr/local/cuda/targets/aarch64-linux/lib
)

add_executable(${PROJECT_NAME} ${HW_HAL_VS_STCS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,-rpath,/lib64 ${LINK_FLAGS_DEF}")

target_link_libraries(${PROJECT_NAME}
    hal_camera
    cuda
    cudart
)
