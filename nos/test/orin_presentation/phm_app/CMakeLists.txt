
cmake_minimum_required(VERSION 2.8.12)

set(MODULE_NAME "phm_app")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_EXTENSIONS OFF)


file(GLOB_RECURSE
        SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avm.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmPubSubTypes.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/avmTypeObject.cxx)

add_executable(phm_app ${SOURCES})

target_include_directories(phm_app
        PRIVATE
        ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/include/
)

target_link_libraries(phm_app
        pthread
        netaos::log
        netaos::phm
        netaos::cm
)

install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

set(LOG_CFG_FILE_PATH "${CMAKE_SOURCE_DIR}/orin_presentation/phm_app/app_sub_phm_config.yaml")
file(COPY ${LOG_CFG_FILE_PATH}
        DESTINATION ${PRJ_ROOT_DIR}/output/${TARGET_PLATFORM}/conf/  FOLLOW_SYMLINK_CHAIN
)
