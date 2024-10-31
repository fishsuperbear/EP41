cmake_minimum_required(VERSION 3.10.2)
set(MODULE_NAME "ota_api_sample")
project(ota_api_sample)

# set env before compile
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../netaos_thirdparty/)
set(NETAOS_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../output)


set(SAMPLE_NAME "ota_api_sample")

file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    )

link_directories(
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib)

add_executable(${MODULE_NAME} ${SOURCES})

target_include_directories(${MODULE_NAME}  
    PRIVATE
    ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/include
    ${PRJ_ROOT_DIR}/middleware

)

target_link_libraries(${MODULE_NAME}  
    PRIVATE
    pthread
    ota_api
    netaos::log
    netaos::em
    netaos::cm
)

#用户可在此添加链接参数
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)