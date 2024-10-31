#module名称
set(MODULE_NAME "nvsipl_multicast")

# add_library(${MODULE_NAME} INTERFACE IMPORTED)
# set_target_properties(${MODULE_NAME} PROPERTIES
#   INTERFACE_INCLUDE_DIRECTORIES "${NETA_MIDDLEWARE_DIR}/sensor/multicast;${COMPILER_SYSROOT}/include;${COMPILER_SYSROOT}/include/nvmedia_6x;${COMPILER_PATH}/../usr/local/cuda-11.4/targets/aarch64-linux/include;"
#   INTERFACE_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/lib"
#   INTERFACE_LINK_LIBRARIES "cudart;nvsipl;nvmedia_iep_sci;nvscistream;nvmedia2d;nvscibuf;nvscisync;nvscievent;nvsciipc;nvscicommon;cuda;tegrawfd;nvsipl_query;pthread"
# )

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libnvsipl_multicast.so
        DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION ${TARGET_PLATFORM}/include/multicast/include)

