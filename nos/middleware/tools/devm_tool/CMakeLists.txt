#module名称
set(MODULE_NAME "devm")

link_directories(
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fastrtps/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB
        SOURCES
        ${CMAKE_SOURCE_DIR}/middleware/tools/devm_tool/*.cpp
        ${CMAKE_SOURCE_DIR}/middleware/tools/devm_tool/src/*.cpp
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/devm.cxx
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/devmTypeObject.cxx
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/devmPubSubTypes.cxx
)


#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/tools/devm_tool/include
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include/devm/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fastrtps/include
)


#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        neta_log
        neta_cm
        fastcdr
        fastrtps
        neta_devm_client
        jsoncpp
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

