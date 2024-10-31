set(MODULE_NAME "derived")

set(LIB_NAME derived)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -rdynamic -fPIC --std=gnu++14 -O2 -Wall -Werror")
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_SOURCE_DIR}/../netaos_thirdparty/cmake/Packages.cmake)
find_package(Protobuf REQUIRED)

link_directories(
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/lib
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/lib
)

file(GLOB_RECURSE
    SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${NETA_MIDDLEWARE_DIR}/idl/generated/*.cxx
    # ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/src/class_register.cpp    
)

add_library(${LIB_NAME} SHARED ${SOURCES})
# add_executable(${LIB_NAME} ${SOURCES})

target_include_directories(${MODULE_NAME}
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/yaml-cpp/include
    ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/fast-dds/include
    ${PRJ_ROOT_DIR}/
)

target_link_directories(${MODULE_NAME}
    PRIVATE
    pthread
    dl
    rt
    neta_adf
    yaml-cpp
    fastrtps
    neta_log
    neta_cm
    fastcdr
    fastrtps
)

install(TARGETS ${MODULE_NAME} LIBRARY
    DESTINATION ${TARGET_PLATFORM}/lib )

install(FILES ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib/libderived.so 
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/adf_main/lib
)

install(FILES ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/bin/adf_main
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/adf_main/bin
)

install(DIRECTORY ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/conf/adf/ 
    DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/test/adf_main/conf
)
