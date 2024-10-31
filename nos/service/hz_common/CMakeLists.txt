#module名称
set(MODULE_NAME "hz_common")

#lib名称
set(LIB_NAME hz_common)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_SOURCE_DIR}/service/canstack/common/*.cpp
)

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

add_definitions(-DCHASSIS_DEBUG_ON)

#用户可在此添加头文件路径
include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/include/
        ${CMAKE_SOURCE_DIR}/service/canstack/common/include/
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
        )
        
#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        neta_log
)

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION include/hz_common)