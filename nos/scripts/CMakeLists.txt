cmake_minimum_required(VERSION 3.10)

install(
  FILES env_setup.sh start.sh nos_bash_complt.sh start_camera.sh set_irq_affiniy.sh
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/scripts)

file(COPY nos_tool.sh
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${TARGET_PLATFORM}/scripts)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/nfs/
        DESTINATION ${TARGET_PLATFORM}/conf/nfs/)