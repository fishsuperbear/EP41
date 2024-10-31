
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")


list(APPEND filters "")
set(SRC_DIRS ${CMAKE_CURRENT_SOURCE_DIR} "${filters}")

set(CMAKE_INSTALL_RPATH "${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "SRC_DIRS: ${SRC_DIRS}")


link_directories(${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/lib
        )

add_executable(tool_test_proc1 
        ${CMAKE_CURRENT_SOURCE_DIR}/tool_test_proc1.cpp
)

target_include_directories(tool_test_proc1
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(tool_test_proc1
        pthread
        netaos::log
)

install(
    TARGETS tool_test_proc1 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin//log_test
)


add_executable(tool_test_proc2
        ${CMAKE_CURRENT_SOURCE_DIR}/tool_test_proc2.cpp
)

target_include_directories(tool_test_proc2
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(tool_test_proc2
        pthread
        netaos::log
)

install(
    TARGETS tool_test_proc2 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin//log_test
)

add_executable(log_level_test
        ${CMAKE_CURRENT_SOURCE_DIR}/log_level_test.cpp
)

target_include_directories(log_level_test
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(log_level_test
        pthread
        netaos::log
)

install(
    TARGETS log_level_test RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin//log_test
)

set(START_TEST_PATH "${CMAKE_SOURCE_DIR}/log_server_test/tool_test/start_test.sh")
file(COPY ${START_TEST_PATH}
        DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/bin/log_test/  FOLLOW_SYMLINK_CHAIN
)
