if (${TARGET_PLATFORM} MATCHES "orin")
    install(FILES netaosConfig-orin.cmake
        DESTINATION ${TARGET_PLATFORM}/lib/cmake/netaos
        RENAME  netaosConfig.cmake
    )
elseif (${TARGET_PLATFORM} MATCHES "x86_2004")
    install(FILES netaosConfig-x86.cmake
        DESTINATION ${TARGET_PLATFORM}/lib/cmake/netaos
        RENAME  netaosConfig.cmake
    )
elseif (${TARGET_PLATFORM} MATCHES "mdc-llvm")
    install(FILES netaosConfig-mdc-llvm.cmake
        DESTINATION ${TARGET_PLATFORM}/lib/cmake/netaos
        RENAME  netaosConfig.cmake
    )
elseif (${TARGET_PLATFORM} MATCHES "j5")
    install(FILES netaosConfig-j5.cmake
        DESTINATION ${TARGET_PLATFORM}/lib/cmake/netaos
        RENAME  netaosConfig.cmake
    )
endif()

install(FILES netaosTargets.cmake
    DESTINATION ${TARGET_PLATFORM}/lib/cmake/netaos
)