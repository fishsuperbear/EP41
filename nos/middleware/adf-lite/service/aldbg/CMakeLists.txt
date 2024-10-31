cmake_minimum_required(VERSION 3.10.2)

project(aldbg)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${NETA_MIDDLEWARE_DIR}/tools/data_tools/common/rcutils/include
    ${NETA_MIDDLEWARE_DIR}/tools/data_tools/common/rcpputils/include
    ${NETA_MIDDLEWARE_DIR}
)

add_library(adf-lite-dbg SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/executor/aldbg_executor.cpp
    ${CMAKE_SOURCE_DIR}/proto/test/soc/dbg_msg.pb.cc
    ${NETA_MIDDLEWARE_DIR}/adf-lite/service/aldbg/player/lite_player.cpp
    ${NETA_MIDDLEWARE_DIR}/adf-lite/service/aldbg/utility/utility.cpp
)

target_link_libraries(adf-lite-dbg
    PUBLIC
    adf-lite
    protobuf
    neta_sm_client
    # rcpputils
    )

install(TARGETS adf-lite-dbg LIBRARY DESTINATION ${TARGET_PLATFORM}/lib)

add_executable(aldbg
    ${CMAKE_CURRENT_SOURCE_DIR}/cli/aldbg.cpp
    ${NETA_MIDDLEWARE_DIR}/adf-lite/service/aldbg/utility/utility.cpp
    ${CMAKE_SOURCE_DIR}/proto/test/soc/dbg_msg.pb.cc
)

target_link_libraries(aldbg PRIVATE adf-lite-dbg protobuf pthread)

install(TARGETS aldbg RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)