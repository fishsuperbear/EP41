
cmake_minimum_required(VERSION 2.8.12)

project(proto_sm_test)

link_directories(
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib) 
        
add_executable(sm_test_pub 
        ${CMAKE_CURRENT_SOURCE_DIR}/sm_test_pub.cpp
        ${PRJ_ROOT_DIR}/proto/soc/chassis.pb.cc
        ${PRJ_ROOT_DIR}/proto/statemachine/state_machine.pb.cc)

add_executable(park_test 
        ${CMAKE_CURRENT_SOURCE_DIR}/park_test.cpp)

add_executable(mock_chassis 
        ${CMAKE_CURRENT_SOURCE_DIR}/mock_chassis.cpp)

target_include_directories(sm_test_pub
        PRIVATE
        ${CMAKE_BINARY_DIR}/proto
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)

target_include_directories(park_test
        PRIVATE
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/middleware/
        ${PRJ_ROOT_DIR}/middleware/cfg/include/
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
)

target_include_directories(mock_chassis
        PRIVATE
        ${PRJ_ROOT_DIR}
        ${PRJ_ROOT_DIR}/middleware/
        ${PRJ_ROOT_DIR}/middleware/cfg/include/
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
)

target_link_libraries(sm_test_pub
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

target_link_libraries(park_test
        pthread
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        dl
        rt
        yaml-cpp
        netaos::log
        protobuf
        netaos::cfg
        uuid
        globalproto
)

target_link_libraries(mock_chassis
        pthread
        fastcdr
        fastrtps
        foonathan_memory-0.7.1
        tinyxml2
        dl
        rt
        yaml-cpp
        netaos::log
        protobuf
        netaos::cfg
        uuid
        globalproto
)

install(
    TARGETS sm_test_pub RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    TARGETS park_test RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    TARGETS mock_chassis RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)