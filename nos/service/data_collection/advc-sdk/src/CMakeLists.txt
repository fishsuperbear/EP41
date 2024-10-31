project(advcsdk)

file(GLOB sdk_common_header "${CMAKE_SOURCE_DIR}/include/*.h")
file(GLOB sdk_syscommon_header "../${CMAKE_SOURCE_DIR}/third_party/include/openssl/*.h")
file(GLOB sdk_op_header "${CMAKE_SOURCE_DIR}/include/op/*.h")
file(GLOB sdk_util_header "${CMAKE_SOURCE_DIR}/include/util/*.h")
file(GLOB sdk_encrypt_header "${CMAKE_SOURCE_DIR}/include/encrypt/*.h")
file(GLOB sdk_header
        ${sdk_common_header}
        ${sdk_syscommon_header}
        ${sdk_op_header}
        ${sdk_request_header}
        ${sdk_response_header}
        ${sdk_util_header}
        ${sdk_encrypt_header})

file(GLOB sdk_common_src "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB sdk_op_src "${CMAKE_SOURCE_DIR}/src/op/*.cpp")
file(GLOB sdk_util_src "${CMAKE_SOURCE_DIR}/src/util/*.cpp")
file(GLOB sdk_encrypt_src "${CMAKE_SOURCE_DIR}/src/encrypt/*.cpp")
file(GLOB sdk_src
        ${sdk_common_src}
        ${sdk_op_src}
        ${sdk_request_src}
        ${sdk_response_src}
        ${sdk_trsf_src}
        ${sdk_util_src}
        ${sdk_encrypt_src})

message("Sdk header: ${sdk_header}")
message("Sdk src: ${sdk_src}")

set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/include/ ${POCO_INCLUDE_DIR} ${OPENSSL_INCLUDE_DIR})
add_library(${PROJECT_NAME} STATIC ${sdk_src} ${sdk_header} ${OPENSSL_INCLUDE_DIR} ../include/advc_model.h)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "advcsdk")

if (BUILD_SHARED_LIB)
    message(STATUS "Build shared lib")
    link_directories(${POCO_LINK_DIR} ${OPENSSL_LINK_DIR}) #这一行要放到add_library前面
    add_library(${PROJECT_NAME}-shared SHARED ${sdk_src})
    include_directories(${CMAKE_SOURCE_DIR}/include/ ${POCO_INCLUDE_DIR} ${OPENSSL_INCLUDE_DIR})
    target_link_libraries(${PROJECT_NAME}-shared ${POCO_LIBS} ${OPENSSL_LIBS}  ${SYSTEM_LIBS})
endif ()
