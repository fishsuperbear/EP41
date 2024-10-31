set(MODULE_NAME "soc_to_mcu")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1200")

link_directories(
    ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/lib
    ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/lib
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/service/ethstack/gen/includes
        ${CMAKE_SOURCE_DIR}/service/ethstack/gen/includes/hozon/netaos/v1
        ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/jsoncpp/include
        ${CMAKE_SOURCE_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/protobuf/include
        ${CMAKE_SOURCE_DIR}/middleware/cm/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)



aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC_FILES)
file(GLOB_RECURSE
        SRC_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
        ${CMAKE_SOURCE_DIR}/middleware/idl/generated/*.cxx
        # ${CMAKE_SOURCE_DIR}/proto/*/*.cc
        )

add_executable(${MODULE_NAME}
    ${SRC_FILES}
)

target_link_libraries(${MODULE_NAME}
    PUBLIC
    pthread
    zmq
    protobuf
    hozon_someip
    protobuf
    neta_cm
    neta_log
    neta_phm
    neta_cm
    neta_em_client
    globalproto
)

install(TARGETS ${MODULE_NAME}
    RUNTIME
	DESTINATION ${TARGET_PLATFORM}/bin
)