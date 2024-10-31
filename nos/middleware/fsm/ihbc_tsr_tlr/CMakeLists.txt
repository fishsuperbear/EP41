#module名称
set(MODULE_NAME ihbc_tsr_tlr)

if(${BUILD_TYPE} STREQUAL "release")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++17 -O3 ")
endif()


#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(
    ${MODULE_NAME}
    PRIVATE
    ${NETA_MIDDLEWARE_DIR}/
    ${COMPILER_SYSROOT}/include
    ${CMAKE_SOURCE_DIR}/
    ${CMAKE_SOURCE_DIR}/middleware/
    ${CMAKE_SOURCE_DIR}/middleware/adf/include
    ${CMAKE_SOURCE_DIR}/middleware/fsm/utils
    ${CMAKE_SOURCE_DIR}/middleware/fsm/include
    ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/bdb/include
    ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/fast-dds/include
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include/
)

#用户添加module依赖
add_dependencies(${MODULE_NAME} fsmcore)

target_link_directories(${MODULE_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/third_party/${TARGET_PLATFORM}/bdb/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
)

#用户可在此添加链接库
target_link_libraries(
    ${MODULE_NAME}
    neta_adf
    neta_cm
    foonathan_memory-0.7.1
    tinyxml2
    fastcdr
    fastrtps
    yaml-cpp
    protobuf
    db_cxx
    fsmcore
)

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed,--disable-new-dtags")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

file(GLOB LOG_CONF_FILES  "conf/*.json")
file(COPY ${LOG_CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)

file(GLOB PB_CONF_FILES  "conf/*.pb.conf")
file(COPY ${PB_CONF_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)

file(GLOB YAML_FILES  "conf/*.yaml")
file(COPY ${YAML_FILES} DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf)
