
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")
set(SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} "${filters}")

set(CMAKE_INSTALL_RPATH "${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/lib
        )

set(IPC_CLIENT "zmq_ipc_client")
#默认编译c和c++文件
add_executable(${IPC_CLIENT}
        ${CMAKE_CURRENT_SOURCE_DIR}/zmq_ipc_client.cpp
        )

#用户可在此添加头文件路径
target_include_directories(${IPC_CLIENT}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include/
        )

#用户可在此添加链接库
target_link_libraries(${IPC_CLIENT}
        PRIVATE
        pthread
        zmq
        neta_zmq_ipc
        protobuf
        )

#用户添加module依赖
#add_dependencies(${IPC_CLIENT} USER_DEPENDENCIES)
install(
    TARGETS ${IPC_CLIENT} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/
)


set(IPC_SERVER "zmq_ipc_server")
#默认编译c和c++文件
add_executable(${IPC_SERVER}
        ${CMAKE_CURRENT_SOURCE_DIR}/zmq_ipc_server.cpp
        )

#用户可在此添加头文件路径
target_include_directories(${IPC_SERVER}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/zmq/include/
        )

#用户可在此添加链接库
target_link_libraries(${IPC_SERVER}
        PRIVATE
        pthread
        zmq
        neta_zmq_ipc
        protobuf
        )

#用户添加module依赖
#add_dependencies(${IPC_SERVER} USER_DEPENDENCIES)
install(
    TARGETS ${IPC_SERVER} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/
)

install(DIRECTORY ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/cpp-ipc/lib/
        DESTINATION ${TARGET_PLATFORM}/lib)
