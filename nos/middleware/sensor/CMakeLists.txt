project(sensor)

link_directories(
    ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib
)

if (${TARGET_PLATFORM} MATCHES "orin")
    add_subdirectory(nvs_adapter)
    add_subdirectory(camera)
    add_subdirectory(nvs_consumer)
    add_subdirectory(nvs_producer)
endif()
