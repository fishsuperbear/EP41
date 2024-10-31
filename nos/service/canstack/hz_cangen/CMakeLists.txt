#module名称
set(MODULE_NAME "hz_cangen")

#lib名称
set(LIB_NAME hz_cangen)

set(CANGEN ${CMAKE_SOURCE_DIR}/service/canstack/hz_cangen)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

add_definitions(-DCHASSIS_DEBUG_ON)

#用户可在此添加头文件路径
include_directories(
        ${SRC_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/include)

#用户可在此添加链接库
target_link_libraries(${LIB_NAME}
        PRIVATE
)

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${TARGET_PLATFORM}/include/hz_cangen)