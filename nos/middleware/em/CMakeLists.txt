#module名称
set(MODULE_NAME "execution_manager")

add_definitions("-D EM_DEBUG_ON")

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)

# set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror")
set(DEFAULT_COMPILE -std=c++14 -O0 -g3 -Wall -fPIC)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")


link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib/
)

#把目录下的c和c++所有源文件分别保存在变量中
file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc

        ${NETA_MIDDLEWARE_DIR}/idl/generated/em.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/emPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/emTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/sm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/smPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/smTypeObject.cxx
        ${CMAKE_SOURCE_DIR}/middleware/em/src/exec_server.cpp
        ${CMAKE_SOURCE_DIR}/middleware/em/src/exec_server_impl.cpp            
        ${CMAKE_SOURCE_DIR}/middleware/em/src/exec_server_impl_zmq.cpp            
        ${CMAKE_SOURCE_DIR}/middleware/sm/src/state_server.cpp
        ${CMAKE_SOURCE_DIR}/middleware/sm/src/state_server_impl.cpp
        ${CMAKE_SOURCE_DIR}/middleware/sm/src/sm_types.cpp
    )

#默认编译c和c++文件
add_executable(${MODULE_NAME} ${SOURCES})


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/middleware
        ${CMAKE_SOURCE_DIR}/middleware/em/inc
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include

        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        neta_log
        neta_cm
        dl
        rt
        neta_sm_server
        )

set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

set(EM_CONFIG_FILE_PATH "${CMAKE_SOURCE_DIR}/middleware/em/conf")

install(TARGETS ${MODULE_NAME} RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

file(COPY ${EM_CONFIG_FILE_PATH}
        DESTINATION ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}  FOLLOW_SYMLINK_CHAIN
)


#库名称
set(CLIENT_LIB_NAME neta_em_client)

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/exec_client*.cpp
        ${NETA_MIDDLEWARE_DIR}/em/utils/cJSON.c
        ${NETA_MIDDLEWARE_DIR}/idl/generated/em.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/emPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/emTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/sm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/smPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/smTypeObject.cxx)

#默认编译c和c++文件
add_library(${CLIENT_LIB_NAME} SHARED ${SOURCES})

target_include_directories(${CLIENT_LIB_NAME}
        PUBLIC
        ${NETA_MIDDLEWARE_DIR}
        ${NETA_MIDDLEWARE_DIR}/em/utils
        # ${NETA_MIDDLEWARE_DIR}/idl/generated/
        # ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        # ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/include
)

target_link_directories(${CLIENT_LIB_NAME}
        PUBLIC
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include/
        ${NETA_MIDDLEWARE_DIR}
)

target_link_libraries(${CLIENT_LIB_NAME}
        PUBLIC
        pthread
        neta_cm
        neta_log
)

install(TARGETS ${CLIENT_LIB_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

file(GLOB_RECURSE
        EM_INCLUDE_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/proctypes.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/exec_client.h)

install(FILES ${EM_INCLUDE_FILE}
        DESTINATION ${TARGET_PLATFORM}/include/em/include)
