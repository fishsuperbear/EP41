cmake_minimum_required(VERSION 3.16.3)
set(MOUDLE_NAME compress_test)

find_library(EXTERNAL_LIB_PATH libneta_compress.so PATHS ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib)

message(${CMAKE_SOURCE_DIR})

add_executable(compress_test
    test.cpp)

target_include_directories(compress_test
PRIVATE
    ${CMAKE_SOURCE_DIR}/../service/data_collection/common/compress/include/
    )

target_link_libraries(compress_test
    ${EXTERNAL_LIB_PATH})

install(TARGETS compress_test
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)