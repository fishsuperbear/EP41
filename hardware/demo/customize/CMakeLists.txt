cmake_minimum_required(VERSION 3.5.1)

project(customize VERSION 1.0.0)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address")
set(DRIVE_OS_DIR /home/orin/wjf/project/hardware/driveos)

include_directories(
    ${DRIVE_OS_DIR}/v6.0.6/include
    ${DRIVE_OS_DIR}/v6.0.6/include/nvmedia_6x
    # /home/jiangxing/driveos/usr/local/cuda-11.4/targets/aarch64-linux/include
    src
)

file(GLOB MULTICAST_SRCS
    "src/*.cpp"
)

#link directorie of lib

link_directories(
    /home/jiangxing/driveos/drive/drive-linux/lib-target
    /home/jiangxing/driveos/usr/lib/aarch64-linux-gnu
    /usr/local/cuda-11.4/targets/aarch64-linux/lib
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
    nvplayfair
)

install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)


