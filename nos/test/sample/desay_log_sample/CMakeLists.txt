
#module名称
set(MODULE_NAME "nos_log")

set(NETAOS_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../output)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)

#默认编译c和c++文件
add_library(${MODULE_NAME} SHARED ${SOURCES})

set_target_properties(nos_log PROPERTIES LINKER_LANGUAGE CXX)

target_link_directories(${MODULE_NAME}
PRIVATE
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
PRIVATE
        neta_log
)

install(
    TARGETS ${MODULE_NAME} 
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)










