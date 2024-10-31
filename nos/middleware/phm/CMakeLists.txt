#module名称
set(MODULE_NAME "neta_phm")
#lib名称
set(LIB_NAME neta_phm)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebase.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebasePubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/servicebaseTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/phm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/phmPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/phmTypeObject.cxx
    )

#默认编译c和c++文件
add_library(${LIB_NAME} SHARED ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/middleware/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include/
        ${NETA_MIDDLEWARE_DIR}/idl/generated/
        ${NETA_MIDDLEWARE_DIR}/cm/include/
        )

target_link_directories(${MODULE_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PUBLIC
        pthread
        neta_log
        neta_cm
        neta_sm_client
        yaml-cpp
        )


set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(TARGETS ${MODULE_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include
        DESTINATION ${TARGET_PLATFORM}/include/phm/)

set(PHM_CONFIG_PATH "${CMAKE_SOURCE_DIR}/middleware/phm/conf")

file(COPY ${PHM_CONFIG_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)