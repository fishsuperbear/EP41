#module名称
set(MODULE_NAME fsmcore)

if(${BUILD_TYPE} STREQUAL "release")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O3 ")
endif()

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    ${CMAKE_SOURCE_DIR}/proto/statemachine/fsm*.pb.cc
)

#默认编译c和c++文件
add_library(${MODULE_NAME} SHARED ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include        
    ${NETA_MIDDLEWARE_DIR}/
    ${COMPILER_SYSROOT}/include
    ${CMAKE_SOURCE_DIR}/
    ${CMAKE_SOURCE_DIR}/middleware/
    ${CMAKE_SOURCE_DIR}/middleware/adf/include
    ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/bdb/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)

target_link_directories(${MODULE_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/bdb/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)

target_link_libraries(${MODULE_NAME}
    PRIVATE
    foonathan_memory-0.7.1
    tinyxml2
    fastcdr
    fastrtps
    neta_adf
    neta_cm
    neta_log
    neta_phm
    yaml-cpp
    neta_codec
    neta_em_client
    protobuf
    db_cxx
)

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed -pie")

install(
    FILES ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/bdb/lib/libdb_cxx-6.2.so
    DESTINATION ${TARGET_PLATFORM}/lib/
)

install(
    FILES ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/bdb/lib/libdb_cxx.so
    DESTINATION ${TARGET_PLATFORM}/lib/
)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/zstd/lib/
    DESTINATION ${TARGET_PLATFORM}/lib/ FILES_MATCHING  PATTERN
    "libzstd.so.*"
)

if(${TARGET_PLATFORM} STREQUAL "x86_2004")
    install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/cuda/targets/x86_64-linux/lib/
        DESTINATION ${TARGET_PLATFORM}/lib/ FILES_MATCHING  PATTERN
        "libcudart.so.*"
    )
endif()

install(
    TARGETS ${MODULE_NAME} LIBRARY
    DESTINATION ${TARGET_PLATFORM}/lib/
)

add_subdirectory(ihbc_tsr_tlr)
add_subdirectory(guard_mode)
add_subdirectory(moving_object_detection)
#add_subdirectory(test)
