set(SERVER_NAME "someip_server")

link_directories(
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
)

add_executable(${SERVER_NAME}
    server.cpp
)

target_include_directories(${SERVER_NAME}
    PRIVATE
    ${PRJ_ROOT_DIR}/middleware
)

target_link_libraries(${SERVER_NAME} 
    CommonAPI 
    CommonAPI-SomeIP 
    vsomeip3
    neta_dosomeip
    netaos::log
)

install(
    TARGETS ${SERVER_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)


