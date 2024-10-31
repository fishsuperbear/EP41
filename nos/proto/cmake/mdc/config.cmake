
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mdc/toolchain.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/mdc/dependence.cmake)

unset(USE_PLATFORM_MDC CACHE)
option(USE_PLATFORM_MDC "use platform mdc" ON)
add_definitions(-DUSE_PLATFORM_MDC)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
)