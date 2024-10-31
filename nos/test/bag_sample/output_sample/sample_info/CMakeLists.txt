cmake_minimum_required(VERSION 3.16.3)
set(MOUDLE_NAME sample_info)
add_executable(${MOUDLE_NAME} sample_info.cpp)
#/home/sw/work/netaos_o/nos/output/x86_2004/替换成工具包目录
target_include_directories(${MOUDLE_NAME}
  PRIVATE /home/sw/work/netaos_o/nos/output/x86_2004/include/data_tools/bag/)
target_link_libraries(
    ${MOUDLE_NAME} /home/sw/work/netaos_o/nos/output/x86_2004/lib/libneta_bag.so)
install(TARGETS ${MOUDLE_NAME}
    RUNTIME DESTINATION ${TARGET_PLATFORM}/bin)