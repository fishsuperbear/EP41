cmake_minimum_required(VERSION 3.0)

project(avp_map_manage)

# module名称
set(MODULE_NAME "avp_map_manage")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")

set(PROJECT_PATH ${CMAKE_SOURCE_DIR})

# 设置动态库生成路径
# set(LIBRARY_OUTPUT_PATH ${PROJECT_PATH}/output/${TARGET_PLATFORM}/lib)

# 编译map_manage
add_library(${MODULE_NAME}
    SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/map_manage.cpp
    ${PROJECT_PATH}/proto/map/avp_map_origin.pb.cc
)

target_include_directories(${MODULE_NAME}
    PUBLIC
    ${PROJECT_PATH}
    ${PROJECT_PATH}/middleware/log/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${PROJECT_PATH}/third_party/${TARGET_PLATFORM}/yaml-cpp/include
    ${PROJECT_PATH}/third_party/${TARGET_PLATFORM}/protobuf/include
)

target_link_directories(${MODULE_NAME}
    PUBLIC
    ${PROJECT_PATH}/output/${TARGET_PLATFORM}/lib
    ${PROJECT_PATH}/third_party/${TARGET_PLATFORM}/yaml-cpp/lib
    ${PROJECT_PATH}/third_party/${TARGET_PLATFORM}/protobuf/lib
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
    yaml-cpp
    protobuf
    pthread
    neta_log
)

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed -Wl,-rpath,/app/lib -Wl,--disable-new-dtags")

add_subdirectory(test)


install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/map_manage.h
DESTINATION ${TARGET_PLATFORM}/include/avp_map_manage/include)