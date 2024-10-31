
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

add_executable(mcu_log_test 
        ${CMAKE_CURRENT_SOURCE_DIR}/mcu_log_test.cpp
)

target_include_directories(mcu_log_test
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(mcu_log_test
        pthread
        netaos::log
)

install(
    TARGETS mcu_log_test RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)
