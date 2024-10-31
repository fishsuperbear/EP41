project(nvsipl_multicast VERSION 1.0.0)

#set(DRIVE_OS_DIR /home/jiangxing/driveos)

include_directories(
    /home/jiangxing/driveos/drive/drive-linux/include/nvmedia_6x
    /home/jiangxing/driveos/drive/drive-linux/include
    /home/jiangxing/driveos/usr/local/cuda-11.4/targets/aarch64-linux/include
)

file(GLOB MULTICAST_SRCS 
    "*.cpp"
)

#link directorie of lib
link_directories(
	/home/jiangxing/driveos/drive/drive-linux/lib-target
	/home/jiangxing/driveos/usr/lib/aarch64-linux-gnu
    /home/jiangxing/driveos/usr/local/cuda-11.4/targets/aarch64-linux/lib
)

add_executable(${PROJECT_NAME} ${MULTICAST_SRCS})
target_link_libraries(${PROJECT_NAME}
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


