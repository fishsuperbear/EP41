
cmake_minimum_required(VERSION 2.8.12)

project(lidar_test)

link_directories(lidar_test
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/third_party/${TARGET_PLATFORM}/pcl/lib
        ${PRJ_ROOT_DIR}/third_party/${TARGET_PLATFORM}/boost/lib) 
        
add_executable(lidar_test 
        ${CMAKE_CURRENT_SOURCE_DIR}/lidar_test.cpp
        ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc)


target_include_directories(lidar_test
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${PRJ_ROOT_DIR}/third_party/${TARGET_PLATFORM}/pcl/include/pcl-1.11
        ${PRJ_ROOT_DIR}/third_party/${TARGET_PLATFORM}/eigen3/include
        ${PRJ_ROOT_DIR}/third_party/${TARGET_PLATFORM}/boost/include
)


target_link_libraries(lidar_test
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
        pcl_io                                                                      
        pcl_common
        pcl_io_ply
        boost_filesystem
        boost_iostreams
        # lboost
)


file(GLOB_RECURSE libs
        ${PRJ_ROOT_DIR}/third_party/${TARGET_PLATFORM}/boost/lib/*
        ${PRJ_ROOT_DIR}/third_party/${TARGET_PLATFORM}/pcl/lib/*
        )

install(FILES 
        ${libs}
        DESTINATION ${TARGET_PLATFORM}/lib
        )
        
install(
    TARGETS lidar_test RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)


