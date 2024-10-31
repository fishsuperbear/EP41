cmake_minimum_required(VERSION 3.10.2)

project(parking_fusion)

include_directories(
    ${PRJ_ROOT_DIR}
    ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include
    ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

add_library(parking_fusion_test SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/parking_fusion_executor.cpp)

target_link_libraries(parking_fusion_test netaos::adf-lite)

install(TARGETS parking_fusion_test LIBRARY DESTINATION ${TARGET_PLATFORM}/test/emproc_adf_test/adf-lite-sample/lib)
