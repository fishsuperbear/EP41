project(nvsipl_camera VERSION 1.0.0)

#set(NV_PLATFORM_DIR /drive/drive-linux)
#set(NV_PLATFORM_SDK_INC_DIR ${NV_PLATFORM_DIR}/include)
#set(NV_PLATFORM_SDK_LIB_DIR ${NV_PLATFORM_DIR}/lib-target)


include_directories(
    /home/jiangxing/driveos/drive/drive-linux/include/nvmedia_6x
    /home/jiangxing/driveos/drive/drive-linux/include
)

file(GLOB NVSIPL_SRCS 
    "*.cpp"
)

#link directorie of lib
link_directories(
	/home/jiangxing/driveos/drive/drive-linux/lib-target
	/home/jiangxing/driveos/usr/lib/aarch64-linux-gnu
)

add_executable(${PROJECT_NAME} ${NVSIPL_SRCS})
target_link_libraries(${PROJECT_NAME}
    libnvsipl_query.so
    libnvsipl.so
    libnvscistream.so
    libtegrawfd.so
    libnvmedia2d.so
    libnvscibuf.so
    libnvscisync.so

    libdl.so
    libpthread.so
)
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)

