cmake_minimum_required(VERSION 2.8.12)
set(MODULE_NAME "someip")

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
add_executable(HelloWorldClient
	src/HelloWorldClient.cpp
	src-gen/v1/commonapi/HelloWorldSomeIPProxy.cpp
	src-gen/v1/commonapi/HelloWorldSomeIPDeployment.cpp
)
target_link_libraries(HelloWorldClient CommonAPI CommonAPI-SomeIP vsomeip3)

add_executable(HelloWorldService
	src/HelloWorldService.cpp
	src/HelloWorldStubImpl.cpp	
	src-gen/v1/commonapi/HelloWorldSomeIPStubAdapter.cpp
	#src-gen/v1/commonapi/HelloWorldStubDefault.cpp
	src-gen/v1/commonapi/HelloWorldSomeIPDeployment.cpp
)
target_link_libraries(HelloWorldService CommonAPI CommonAPI-SomeIP vsomeip3)

install(
    TARGETS HelloWorldClient RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

install(
    TARGETS HelloWorldService RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
