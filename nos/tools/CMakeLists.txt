project(tools)

if (${TARGET_PLATFORM} MATCHES "orin")
    file(GLOB TOOLS_BIN 
        ${CMAKE_CURRENT_SOURCE_DIR}/killall/*
        ${CMAKE_CURRENT_SOURCE_DIR}/memtester/*
        ${CMAKE_CURRENT_SOURCE_DIR}/perf/bin/*
    )
    install(FILES ${TOOLS_BIN}
            DESTINATION ${TARGET_PLATFORM}/bin
            PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
            GROUP_EXECUTE GROUP_READ
            WORLD_EXECUTE WORLD_READ
    )

    file(GLOB TOOLS_LIB 
        ${CMAKE_CURRENT_SOURCE_DIR}/perf/lib/*
    )
    install(FILES ${TOOLS_LIB}
            DESTINATION ${TARGET_PLATFORM}/lib
    )
endif()

install(DIRECTORY FlameGraph DESTINATION  ${TARGET_PLATFORM}/tools
        FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
                        GROUP_EXECUTE GROUP_READ
                        WORLD_EXECUTE WORLD_READ)