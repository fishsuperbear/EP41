add_subdirectory(aldbg)
install(FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/rpc/lite_rpc.h
    DESTINATION ${TARGET_PLATFORM}/include/adf-lite/service/rpc)