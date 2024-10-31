project(simple_udp_sync)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${NETA_MIDDLEWARE_DIR}
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)

link_directories(

)

add_executable(udp_tsync_master
    udp_tsync_master.cpp
)

add_executable(udp_tsync_slave
    udp_tsync_slave.cpp
)

target_link_libraries(udp_tsync_master pthread)
target_link_libraries(udp_tsync_slave pthread)

install(TARGETS udp_tsync_master udp_tsync_slave RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin)