cmake_minimum_required(VERSION 3.5.1)

project(hw_nvmedia_imx728 VERSION 1.0.0)

#自动查找头文件路径函数(没有去重)
macro(FIND_INCLUDE_DIR result curdir)  #定义函数,2个参数:存放结果result；指定路径curdir；
    file(GLOB_RECURSE children "${curdir}/*.hpp" "${curdir}/*.h" )	#遍历获取{curdir}中*.hpp和*.h文件列表
    # message(STATUS "children= ${children}")								#打印*.hpp和*.h的文件列表
    set(dirlist "")														#定义dirlist中间变量，并初始化
    foreach(child ${children})											#for循环
        string(REGEX REPLACE "(.*)/.*" "\\1" LIB_NAME ${child})			#字符串替换,用/前的字符替换/*h
        if(IS_DIRECTORY ${LIB_NAME})									#判断是否为路径
            LIST(APPEND dirlist ${LIB_NAME})							#将合法的路径加入dirlist变量中
        endif()															#结束判断
    endforeach()														#结束for循环
    set(${result} ${dirlist})											#dirlist结果放入result变量中
endmacro()																#函数结束

FIND_INCLUDE_DIR(INCLUDE_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../code)
FIND_INCLUDE_DIR(INCLUDE_DRIVE_OS_DIR_LIST ${CMAKE_CURRENT_SOURCE_DIR}/../../../driveos)

set(LIB_DIR ../../code/lib)
#set(DRIVE_OS_DIR /home/jiangxing/driveos)

include_directories(
    /home/jiangxing/driveos/drive/drive-linux/include/nvmedia_6x
    /home/jiangxing/driveos/drive/drive-linux/include
    /home/jiangxing/driveos/usr/local/cuda-11.4/targets/aarch64-linux/include
    ${INCLUDE_DIR_LIST}
    # ${INCLUDE_DRIVE_OS_DIR_LIST}
)

file(GLOB_RECURSE HW_NVMEDIA_IMX728_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/hw_nvmedia_imx728_vs/hw_nvmedia_imx728_vs/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../code/modules/*.cpp"
)

#link directorie of lib
link_directories(
	/home/jiangxing/driveos/drive/drive-linux/lib-target
	/home/jiangxing/driveos/usr/lib/aarch64-linux-gnu
    /home/jiangxing/driveos/usr/local/cuda-11.4/targets/aarch64-linux/lib
    ${LIB_DIR}
)

add_library(${PROJECT_NAME} SHARED ${HW_NVMEDIA_IMX728_SRCS})

target_link_libraries(${PROJECT_NAME}
    libhw_hal_vs.so
    libhw_platform_vs.so
    libnvsipl.so
    libnvsipl_query.so
    libnvmedia_iep_sci.so
    libnvscistream.so
    libnvscibuf.so
    libnvscisync.so
    libnvscievent.so
    libnvsciipc.so
    libnvscicommon.so
    libcuda.so
    libcudart.so
    libpthread.so
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

add_definitions(-D__HW_NVMEDIA_PROJ_USING_IMX728)

