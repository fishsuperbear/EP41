#lib名称
set(LIB_NAME neta_logblock_helper)

#把目录下的c所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})

target_include_directories(${LIB_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/logblock/include
)

target_link_directories(${LIB_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/logblock/lib        
)

# #用户可在此添加链接库
target_link_libraries(${LIB_NAME}
        PUBLIC
        logblockreader
        logblockwriter
        hw_platform
)

set_target_properties(${LIB_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

set(LIB_LOGBLOCK_PATH "${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/logblock/lib")
file(COPY ${LIB_LOGBLOCK_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)

install(TARGETS ${LIB_NAME} LIBRARY
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${TARGET_PLATFORM}/include/logblock_helper/)
