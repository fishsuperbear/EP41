
set(MODULE_NAME "someip_svc_test")
set(SOMEIP_SVC_TEST someip_svc_test)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

file(GLOB_RECURSE SOMEIP_SVC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_service.cpp
)

add_executable(${MODULE_NAME} ${SOMEIP_SVC_FILES})


#用户可在此添加链接库目录
link_directories(
)

target_link_libraries(${MODULE_NAME}
    pthread
)

install(TARGETS ${MODULE_NAME}
    RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)
