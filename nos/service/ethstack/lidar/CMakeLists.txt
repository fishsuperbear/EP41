set(MODULE_NAME "neta_lidar")

set(LIB_NAME neta_lidar)

link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

if(${BUILD_TYPE} STREQUAL "release")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
endif()

list(APPEND filters "")

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmTypeObject.cxx
        )

add_executable(${MODULE_NAME} ${SOURCES})

target_link_directories(${MODULE_NAME}
    PUBLIC
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${COMPILER_SYSROOT}/lib
    ${COMPILER_PATH}/../usr/lib
    ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/lib  
    ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/pcl/lib       
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib/                      
    )


target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
        ${NETA_MIDDLEWARE_DIR}/
        ${CMAKE_SOURCE_DIR}/service/ethstack/lidar
        ${CMAKE_SOURCE_DIR}/
        ${COMPILER_SYSROOT}/include
        ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/pcl/include/pcl-1.11
        ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/eigen3/include
        ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/include
        ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/boost/include/boost
        ${CMAKE_SOURCE_DIR}/service/
        ${PRJ_ROOT_DIR}/middleware    
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include/
        # ${CMAKE_SOURCE_DIR}/middleware/cfg/include
        )                       
    

target_link_libraries(${MODULE_NAME}
        PRIVATE
        fastcdr
        fastrtps
        neta_cm
        neta_log
        yaml-cpp
        neta_codec
        pcl_io                                                                      
        pcl_common
        # boost
        neta_em_client
        pcl_io_ply
        boost_filesystem
        boost_iostreams
        neta_phm
        neta_cfg
        jsoncpp
        )                                                                 

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed -pie")


install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)


# file(GLOB_RECURSE correct
#         ${CMAKE_CURRENT_SOURCE_DIR}/conf/*    
#         )

# install(FILES 
#         ${correct}
#         DESTINATION ${TARGET_PLATFORM}/conf/
#         )

#for test
# install(
#     TARGETS
#     ${MODULE_NAME}
#     RUNTIME
#     DESTINATION ${TARGET_PLATFORM}/test/neta_lidar/bin
#     )


