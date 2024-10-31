#module名称
set(MODULE_NAME "ota_api")

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Werror")
link_directories(
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_SOURCE_DIR}/middleware/ota_comp_api/src/*.c
        ${CMAKE_SOURCE_DIR}/middleware/ota_comp_api/src/*.cxx
        ${CMAKE_SOURCE_DIR}/middleware/ota_comp_api/src/*.cpp
        ${CMAKE_SOURCE_DIR}/middleware/ota_comp_api/src/*.cc
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/devm.cxx
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/devmTypeObject.cxx
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/devmPubSubTypes.cxx
)


#默认编译c和c++文件
add_library(${MODULE_NAME} SHARED ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/ota_comp_api/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
)


#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        neta_log
        neta_cm
        neta_em_client
        fastcdr
        fastrtps
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} DESTINATION ${TARGET_PLATFORM}/lib)
install(FILES ${CMAKE_SOURCE_DIR}/middleware/ota_comp_api/include/ota_api.h
        DESTINATION ${TARGET_PLATFORM}/include/ota_comp_api/include)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/middleware/ota_comp_api/conf/
        DESTINATION ${TARGET_PLATFORM}/conf)

