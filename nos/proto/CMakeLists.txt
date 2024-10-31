cmake_minimum_required(VERSION 3.12.1)

# -----just for build proto single------
if(GLOBALPROTO_SINGLE_MODULE_COMPILE)
   set(3RD_ROOT_PROTOC ${CMAKE_CURRENT_LIST_DIR}/third_party/third_party/x86/protobuf)
   include(${CMAKE_CURRENT_LIST_DIR}/third_party/third_party/cmake/FindProtobuf.cmake)
  include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${PLATFORM_3RD}/protobuf/include   )
# ---- build in submodule----- default! no not set above value,and set PLATFORM_3RD in your main cmake by dif platform
else()
   set(3RD_ROOT_PROTOC ${PLATFORM_3RD}/../x86/protobuf)
   include(${PLATFORM_3RD}/../cmake/FindProtobuf.cmake)
   include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../)
   include_directories(${PLATFORM_3RD}/protobuf/include )
endif()
# -----build select end-----


set(3RD_PROTO_LIB "${PLATFORM_3RD}/protobuf/lib/libprotobuf.so")

SET(CMAKE_CXX_FLAGS "-w ${CMAKE_CXX_FLAGS}")
message("Proto Compiler Options: " ${CMAKE_CXX_FLAGS})
project(globalproto)

file(
  GLOB
  PROTOS
  "localization/*.proto"
  "common/*.proto"
  "canbus/*.proto"
  "map/*.proto"
  "control/*.proto"
  "dead_reckoning/*.proto"
  "drivers/*.proto"
  "hmi/*.proto"
  "dreamview/*.proto"
  "planning/*.proto"
  "planning/math/*.proto"
  "perception/*.proto"
  "prediction/*.proto"
  "routing/*.proto"
  "storytelling/*.proto"
  "transform/*.proto"
  "fsm/*.proto"
  "sensors/*.proto"
  "soc/*.proto"
  "test/soc/*.proto"
  "statemachine/*.proto"
  "local_mapping/*.proto"
  )
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTOS})


# --- here for adf easy release---
if (BUILD_BY_ADF)
    message("Build by adf")
    set(LIB_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
else()
    set(LIB_OUTPUT_DIR "lib")
endif()

add_library(${PROJECT_NAME} SHARED ${PROTO_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-Wl,--disable-new-dtags")
set_target_properties(${PROJECT_NAME} PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(${PROJECT_NAME} ${3RD_PROTO_LIB})

install(
  TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${LIB_OUTPUT_DIR}
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION .)
