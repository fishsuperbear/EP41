#lib名称
set(LIB_NAME neta_crc)

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
)

# #用户可在此添加链接库
# target_link_libraries(${MODULE_NAME}
#         )

set_target_properties(${LIB_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${LIB_NAME} LIBRARY
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/crc.h
        DESTINATION ${TARGET_PLATFORM}/include/crc/include/)