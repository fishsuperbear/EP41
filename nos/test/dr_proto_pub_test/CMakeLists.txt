
cmake_minimum_required(VERSION 2.8.12)

project(dr_proto_pub_test)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib) 
        
add_executable(dr_proto_pub_test 
        ${CMAKE_CURRENT_SOURCE_DIR}/dr_proto_pub_test.cpp
        ${PRJ_ROOT_DIR}/proto/dead_reckoning/dr.pb.cc)

target_include_directories(dr_proto_pub_test
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)


target_link_libraries(dr_proto_pub_test
        pthread
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        neta_cm
        dl
        rt
        netaos::log
        protobuf
        uuid
        globalproto
)

install(
    TARGETS dr_proto_pub_test  RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
