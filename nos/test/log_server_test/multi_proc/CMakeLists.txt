
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

add_executable(proc_001 
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_001.cpp
)

target_include_directories(proc_001
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_001
        pthread
        netaos::log
)

install(
    TARGETS proc_001 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_002
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_002.cpp
)

target_include_directories(proc_002
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_002
        pthread
        netaos::log
)

install(
    TARGETS proc_002 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_003
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_003.cpp
)

target_include_directories(proc_003
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_003
        pthread
        netaos::log
)

install(
    TARGETS proc_003 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_004
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_004.cpp
)

target_include_directories(proc_004
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_004
        pthread
        netaos::log
)

install(
    TARGETS proc_004 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_005
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_005.cpp
)

target_include_directories(proc_005
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_005
        pthread
        netaos::log
)

install(
    TARGETS proc_005 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_006
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_006.cpp
)

target_include_directories(proc_006
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_006
        pthread
        netaos::log
)

install(
    TARGETS proc_006 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_007
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_007.cpp
)

target_include_directories(proc_007
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_007
        pthread
        netaos::log
)

install(
    TARGETS proc_007 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_008
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_008.cpp
)

target_include_directories(proc_008
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_008
        pthread
        netaos::log
)

install(
    TARGETS proc_008 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_009
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_009.cpp
)

target_include_directories(proc_009
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_009
        pthread
        netaos::log
)

install(
    TARGETS proc_009 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

add_executable(proc_010
        ${CMAKE_CURRENT_SOURCE_DIR}/proc_010.cpp
)

target_include_directories(proc_010
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware/log/include
)

target_link_libraries(proc_010
        pthread
        netaos::log
)

install(
    TARGETS proc_010 RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin/log_test
)

set(RUN_PATH "${CMAKE_SOURCE_DIR}/log_server_test/multi_proc/run.sh")
file(COPY ${RUN_PATH}
        DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/bin/log_test/  FOLLOW_SYMLINK_CHAIN
)


