#module名称
set(MODULE_NAME "devm_server")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

link_directories(
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/lib
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_SOURCE_DIR}/service/devm_server/src/*.cpp
        ${CMAKE_SOURCE_DIR}/service/devm_server/src/*.cc
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
        ${CMAKE_SOURCE_DIR}/service
        ${CMAKE_SOURCE_DIR}/service/devm_server/include
        ${CMAKE_SOURCE_DIR}/service/devm_server/include/common
        ${CMAKE_SOURCE_DIR}/middleware/per
        ${CMAKE_SOURCE_DIR}/middleware/per/include
        ${CMAKE_SOURCE_DIR}/middleware/per/proto
        ${CMAKE_SOURCE_DIR}/middleware/per/struct2x
        ${CMAKE_SOURCE_DIR}/middleware/per/struct2x/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/sqlite/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include
        #${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/nlohmann_json/include/nlohmann
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/nlohmann_json/include
)


#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        jsoncpp
        neta_log
        #neta_cm
        neta_em_client
        #fastcdr
        #fastrtps
        neta_cfg
        neta_zmq_ipc
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

if (${TARGET_PLATFORM} MATCHES "orin")
file(GLOB THIRD_PARTY_FILES ${CMAKE_SOURCE_DIR}/service/devm_server/bin/${TARGET_PLATFORM}/*)
install(FILES ${THIRD_PARTY_FILES}
    DESTINATION ${TARGET_PLATFORM}/bin
)
endif()
