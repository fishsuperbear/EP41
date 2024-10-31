
cmake_minimum_required(VERSION 2.8.12)

get_filename_component(CURRENT_DIR_NAME ${CMAKE_CURRENT_LIST_DIR} NAME)
set(MODULE_NAME ${CURRENT_DIR_NAME})

# install(
#     TARGETS ${MODULE_NAME}.sh RUNTIME
#     DESTINATION ${TARGET_PLATFORM}/test/unit_test/nvs_producer_test_unit/${CURRENT_DIR_NAME}
# )
file(COPY ${MODULE_NAME}.sh
    DESTINATION  ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/test/unit_test/nvs_producer_test_unit/${CURRENT_DIR_NAME}  FOLLOW_SYMLINK_CHAIN)