#module名称
set(MODULE_NAME "extwdg")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

link_directories(
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/lib
        ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_SOURCE_DIR}/service/extwdg/*.cpp
        ${CMAKE_SOURCE_DIR}/service/extwdg/*.cc
        ${CMAKE_SOURCE_DIR}/service/extwdg/src/*.cpp
        ${CMAKE_SOURCE_DIR}/service/extwdg/src/*.cc
)


#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/service
        ${CMAKE_SOURCE_DIR}/service/extwdg/include/
        ${CMAKE_SOURCE_DIR}/service/extwdg/include/common/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/nlohmann_json/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include/zmq/
)


#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        neta_log
        neta_cm
        neta_phm
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

if (${TARGET_PLATFORM} MATCHES "orin")
file(GLOB THIRD_PARTY_FILES ${CMAKE_SOURCE_DIR}/service/extwdg/bin/${TARGET_PLATFORM}/*)
file(GLOB EXTWDGCONFIG ${CMAKE_SOURCE_DIR}/service/extwdg/conf/*)
install(FILES ${THIRD_PARTY_FILES}
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(FILES ${EXTWDGCONFIG}
    DESTINATION ${TARGET_PLATFORM}/conf
)

# file(COPY ${EXTWDGCONFIG}
#     DESTINATION ${TARGET_PLATFORM}/conf
# )
endif()