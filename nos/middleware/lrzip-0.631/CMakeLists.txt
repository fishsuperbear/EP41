
#源码中没有此文件，自己增加；生成lib库的方式对外提供
#lrunzip.h和lrunzip另外自己增加，并修改main函数为LrzipMain()
#github下载源码后执行./configure生成对应的config.h，之后再cmake

cmake_minimum_required(VERSION 3.0)

set(MODULE_NAME "neta_lrzip")
project(${MODULE_NAME})

add_definitions(-DHAVE_CONFIG_H)
add_compile_options(-Wall -Wextra)

# 添加库的源代码文件
file(GLOB_RECURSE
    SOURCES
    main.c
    lzma/C/Threads.c
    lzma/C/LzmaDec.c
    lzma/C/7zCrc.c
    lzma/C/LzmaEnc.c
    lzma/C/Alloc.c
    lzma/C/LzFindMt.c
    lzma/C/LzFind.c
    lzma/C/LzmaLib.c
    runzip.c
    liblrzip.c
    rzip.c
    aes.c
    lrzip.c
    md5.c
    stream.c
    sha4.c
    util.c
    libzpaq/libzpaq.cpp
)

add_library(${MODULE_NAME} SHARED ${SOURCES})

target_compile_options(${MODULE_NAME} PRIVATE -O3)

target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware/lrzip-0.631
        ${CMAKE_SOURCE_DIR}/middleware/lrzip-0.631/lzma/C
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lz4/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lzo2/include
)

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lz4/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lzo2/lib
)
# #用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        stdc++
)

target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        lzo2
        lz4
        m
        #bz2
        z
)

# 指定安装路径
install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)
install(FILES lrunzip.h
        DESTINATION ${TARGET_PLATFORM}/include/lrzip/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lzo2/lib/
        DESTINATION ${TARGET_PLATFORM}/lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/lz4/lib/
        DESTINATION ${TARGET_PLATFORM}/lib)


