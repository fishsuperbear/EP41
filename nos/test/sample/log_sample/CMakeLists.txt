cmake_minimum_required(VERSION 3.10.2)

project(log_sample)

# set env before compile
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../netaos_thirdparty/)
set(NETAOS_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../output)


set(SAMPLE_NAME "log_sample")

add_executable(${SAMPLE_NAME} 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/log_sample.cpp
)

target_include_directories(${SAMPLE_NAME}  
    PRIVATE
    ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/include
    ${PRJ_ROOT_DIR}/middleware/log/include

)

target_link_libraries(${SAMPLE_NAME}  
    PRIVATE
    netaos::log
)

install(
    TARGETS
    ${SAMPLE_NAME}  
    RUNTIME
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/log-sample/bin
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/conf/log_cfg.json
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/log-sample/conf
)

install(FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/log_sample.cpp
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/log-sample/src
)

install(FILES  
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    DESTINATION ${NETAOS_OUTPUT_DIR}/${TARGET_PLATFORM}/test/log-sample/
)