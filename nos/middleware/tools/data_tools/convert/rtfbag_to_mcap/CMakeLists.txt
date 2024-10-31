cmake_minimum_required(VERSION 3.12)
set(RTF_TO_MCAP_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(convert_02280324_to_mcap)
add_subdirectory(convert_04300723_to_mcap)
install(FILES topic_info_mapping.json DESTINATION ${TARGET_PLATFORM}/conf/bag/convert/rtfbag_to_mcap)

install(
DIRECTORY ./MDC_Ubuntu_X86
DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/conf/bag)