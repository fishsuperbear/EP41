set(CLIENT_NAME "someip_client")

include_directories(
)

add_executable(${CLIENT_NAME}
	client.cpp
    ../../src-gen/v1/commonapi/DoSomeIPSomeIPProxy.cpp
	../../src-gen/v1/commonapi/DoSomeIPSomeIPDeployment.cpp
)

target_link_libraries(${CLIENT_NAME} 
    CommonAPI 
    CommonAPI-SomeIP 
    vsomeip3
    # neta_dosomeip
)

install(
    TARGETS ${CLIENT_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
