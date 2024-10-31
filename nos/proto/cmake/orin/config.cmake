
add_definitions(-DUSE_PLATFORM_ORIN)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/orin/toolchain.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/orin/dependence.cmake)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
)