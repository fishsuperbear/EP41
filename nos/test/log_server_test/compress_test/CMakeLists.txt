
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")


list(APPEND filters "")
set(SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} "${filters}")

set(CMAKE_INSTALL_RPATH "${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")


link_directories(${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        )

add_executable(delete_when_compress 
        ${CMAKE_CURRENT_SOURCE_DIR}/delete_when_compress.cpp
)

target_include_directories(delete_when_compress
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(delete_when_compress
        pthread
        netaos::log
)

install(
    TARGETS delete_when_compress RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)
