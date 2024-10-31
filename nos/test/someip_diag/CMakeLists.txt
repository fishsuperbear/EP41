cmake_minimum_required(VERSION 2.8.12)
set(MODULE_NAME "someip_diag")
set(SERVER_NAME "dosomeip_server")
set(CLIENT_NAME "dosomeip_client")


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ldl")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
    src-gen
    src
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/include
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/include
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/include

)
link_directories(
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/commonapi_someip/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/vsomeip/lib
)
add_executable(${CLIENT_NAME}
	src/DoSomeIPClient.cpp
	src-gen/v1/commonapi/DoSomeIPSomeIPProxy.cpp
	src-gen/v1/commonapi/DoSomeIPSomeIPDeployment.cpp
)
target_link_libraries(${CLIENT_NAME} CommonAPI CommonAPI-SomeIP vsomeip3)

add_executable(${SERVER_NAME}
	src/DoSomeIPServer.cpp
	src/DoSomeIPStubImpl.cpp	
	src-gen/v1/commonapi/DoSomeIPSomeIPStubAdapter.cpp
	src-gen/v1/commonapi/DoSomeIPSomeIPDeployment.cpp
)
target_link_libraries(${SERVER_NAME} CommonAPI CommonAPI-SomeIP vsomeip3)

install(
    TARGETS ${CLIENT_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    TARGETS ${SERVER_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

add_subdirectory(test)
