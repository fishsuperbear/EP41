
cmake_minimum_required(VERSION 2.8.12)

project(proto_cm_test)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib) 
        
add_executable(proto_cm_test_pub 
        ${CMAKE_CURRENT_SOURCE_DIR}/pub_proto_main.cpp
        ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc)
add_executable(proto_cm_intra_pub 
        ${CMAKE_CURRENT_SOURCE_DIR}/pub_proto_intra.cpp
        ${PRJ_ROOT_DIR}/proto/localization/localization.pb.cc
        ${PRJ_ROOT_DIR}/proto/perception/perception_obstacle.pb.cc
        ${PRJ_ROOT_DIR}/proto/planning/planning.pb.cc
        )
add_executable(proto_cm_test_sub 
        ${CMAKE_CURRENT_SOURCE_DIR}/sub_proto_main.cpp
        ${PRJ_ROOT_DIR}/proto/test/soc/for_test.pb.cc)

target_include_directories(proto_cm_test_pub
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)
target_include_directories(proto_cm_intra_pub
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)


target_include_directories(proto_cm_test_sub
        PRIVATE
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)

target_link_libraries(proto_cm_test_pub
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

target_link_libraries(proto_cm_test_sub
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
target_link_libraries(proto_cm_intra_pub
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
    TARGETS proto_cm_test_pub proto_cm_test_sub proto_cm_intra_pub RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
