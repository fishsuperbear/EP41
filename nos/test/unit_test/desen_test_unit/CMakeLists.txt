
set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage --coverage")
set(DEFAULT_COMPILE -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -fPIC -fPIE -pie -fstack-protector-all -Wtrampolines)

get_filename_component(MAIN_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)

# common lib dir
link_directories(
  ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/lib/
)

# common include dir
include_directories(
  ${PRJ_ROOT_DIR}/netaos_thirdparty/${TARGET_PLATFORM}/gtest/include

)

set(CMAKE_SKIP_RPATH  TRUE)

file(GLOB SUB_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *)
foreach(DIR ${SUB_DIRS})
  if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${DIR})
    add_subdirectory(${DIR})
  endif()
endforeach()


