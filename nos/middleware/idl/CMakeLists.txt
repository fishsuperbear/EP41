project(idl)

include(${CMAKE_SOURCE_DIR}/cmake/neta_fastddsgen.cmake)

file(GLOB_RECURSE IDL_FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/data_type/*.idl)

FASTDDS_GENERATE_CPP(IDL_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/data_type 
        IDL_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/generated ${IDL_FILES})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/generated/
        DESTINATION ${TARGET_PLATFORM}/include/idl/generated)
