#module名称
set(MODULE_NAME "neta_sm")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

#库名称
set(SERVER_LIB_NAME neta_sm_server)
set(CLIENT_LIB_NAME neta_sm_client)

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${NETA_MIDDLEWARE_DIR}/em/utils/*.c
        ${NETA_MIDDLEWARE_DIR}/em/src/*.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/em.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/emPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/emTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/sm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/smPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/smTypeObject.cxx)

include_directories(
        ${NETA_MIDDLEWARE_DIR}/idl/generated/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/include/
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/libuuid-1.0.3/include/
        ${NETA_MIDDLEWARE_DIR}
)

#默认编译c和c++文件
add_library(${SERVER_LIB_NAME} SHARED ${SOURCES})
add_library(${CLIENT_LIB_NAME} SHARED ${SOURCES})

#用户可在此添加链接库
target_link_libraries(${SERVER_LIB_NAME}
        PUBLIC
        pthread
        neta_cm
        neta_log
        neta_zmq_ipc
)

target_link_libraries(${CLIENT_LIB_NAME}
        PUBLIC
        neta_cm
        neta_log
        neta_zmq_ipc
)

install(TARGETS ${SERVER_LIB_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)
install(TARGETS ${CLIENT_LIB_NAME} LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib)

file(GLOB_RECURSE
        SM_INCLUDE_FILE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/state_client.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/state_client_zmq.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/state_client_impl.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/state_client_impl_zmq.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sm_types.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/sm_logger.h
        )

install(FILES ${SM_INCLUDE_FILE}
        DESTINATION ${TARGET_PLATFORM}/include/sm/include)