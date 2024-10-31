cmake_minimum_required(VERSION 3.16.3)

project(DDSHelloWorldExample VERSION 1 LANGUAGES CXX)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

message(STATUS "Configuring HelloWorld example...")

file(GLOB DDS_HELLOWORLD_EXAMPLE_SOURCES_CXX "*.cxx")
file(GLOB DDS_HELLOWORLD_EXAMPLE_SOURCES_CPP "*.cpp")

add_executable(DDSHelloWorldExample ${DDS_HELLOWORLD_EXAMPLE_SOURCES_CXX} ${DDS_HELLOWORLD_EXAMPLE_SOURCES_CPP})

target_include_directories(DDSHelloWorldExample
        PRIVATE
        ${NETA_MIDDLEWARE_DIR}/
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
        ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include/fastdds/thirdparty/optionparser
        )

target_compile_definitions(DDSHelloWorldExample PRIVATE
    $<$<AND:$<NOT:$<BOOL:${WIN32}>>,$<STREQUAL:"${CMAKE_BUILD_TYPE}","Debug">>:__DEBUG>
    $<$<BOOL:${INTERNAL_DEBUG}>:__INTERNALDEBUG> # Internal debug activated.
)
target_link_libraries(DDSHelloWorldExample
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastcdr.so
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib/libfastrtps.so
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/tinyxml2/lib/libtinyxml2.so

)
install(TARGETS DDSHelloWorldExample
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)

install(FILES HelloWorld.idl DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/share/data_type)