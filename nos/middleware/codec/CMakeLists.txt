#module名称
set(MODULE_NAME "neta_codec")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_SKIP_RPATH  TRUE)

if (${TARGET_PLATFORM} MATCHES "orin")
# ============================== setup cuda ===============================
    set(CUDA_ROOT ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/aarch64-linux)
    set(CMAKE_CUDA_COMPILER ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/bin/nvcc)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER} -arch=sm_87 -Xcompiler -fPIC" CACHE STRING "" FORCE)
    enable_language(CUDA)
    # =============================== end setup cuda ==========================
    set(NV_TOPDIR ${COMPILER_PATH}/..)
    set(NV_PLATFORM_DIR ${NV_TOPDIR}/drive-linux)
    set(NV_PLATFORM_SDK_INC_DIR ${NV_PLATFORM_DIR}/include)
    set(NV_PLATFORM_SDK_LIB_DIR ${NV_PLATFORM_DIR}/lib-target ${COMPILER_PATH}/../usr/lib ${COMPILER_SYSROOT}/lib)
    set(CODEC_LIBRARY avcodec avutil avformat swresample swscale nvscibuf nvscisync nvmedia_iep_sci nvvideo pthread nvparser nvmedia_ide_parser nvmedia_ide_sci nv_ide cudart)
    set(CODEC_LIBRARY_DIRS ${NV_PLATFORM_SDK_LIB_DIR} ${CUDA_ROOT}/lib)
    message("==========CODEC_LIBRARY_DIRS: ${CODEC_LIBRARY_DIRS}=============")
    set(CODEC_PRIVATE_INC_DIRS 
        ${COMPILER_SYSROOT}/include/nvmedia_6x
        src/nvmedia/utils src/nvmedia/iep src/nvmedia/ide src/orin
        ${NV_PLATFORM_SDK_INC_DIR}
        ${CUDA_ROOT}/include
        )
    add_definitions(-DPLAT_ORIN -DNVMEDIA_NVSCI_ENABLE -D_FILE_OFFSET_BITS=64 -DNV_IS_SAFETY=0)
    #把目录下的c和c++所有源文件分别保存在变量中
    file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nvmedia/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nvmedia/iep/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/nvmedia/utils/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda_kernel/*.cu
        )
elseif (${TARGET_PLATFORM} MATCHES "x86_2004")
    # only for x86
    set(CODEC_PRIVATE_INC_DIRS
        ${NETA_MIDDLEWARE_DIR}/../
        ${CMAKE_SOURCE_DIR}/
    )

    set(CODEC_LIBRARY avcodec avformat avutil swresample swscale)
    add_definitions(-DPLAT_X86)
    #把目录下的c和c++所有源文件分别保存在变量中
    file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp)

    # find cuda files
    file(GLOB_RECURSE CUDA_LIBS ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/libcudart.so*)

elseif (${TARGET_PLATFORM} MATCHES "mdc-llvm")
    add_definitions(-DPLAT_MDC)
    #把目录下的c和c++所有源文件分别保存在变量中
    file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mdc/*.cpp)
endif()

#默认编译c和c++文件
add_library(${MODULE_NAME} SHARED 
        ${SOURCES} 
        ${CMAKE_CURRENT_SOURCE_DIR}/src/empty/encoder_empty.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/empty/decoder_empty.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/decoder_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/encoder_factory.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/codec_logger.cpp
        )


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_SYSROOT}/include
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/ffmpeg/include/
        ${CODEC_PRIVATE_INC_DIRS}
        ${CMAKE_SOURCE_DIR}
)

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/ffmpeg/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
        ${CODEC_LIBRARY_DIRS}
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        neta_log
        ${CODEC_LIBRARY}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${TARGET_PLATFORM}/include/codec/include)

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

if (${TARGET_PLATFORM} MATCHES "orin")
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/ffmpeg/lib
            DESTINATION ${TARGET_PLATFORM})

    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/src/nvmedia/utils/scibuf_utils.h DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include/codec/include)
elseif (${TARGET_PLATFORM} MATCHES "x86_2004")
    install(DIRECTORY ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/ffmpeg/lib
            DESTINATION ${TARGET_PLATFORM})

    file(COPY ${CUDA_LIBS} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib FOLLOW_SYMLINK_CHAIN)
endif()
