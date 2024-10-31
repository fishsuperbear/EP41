cmake_minimum_required(VERSION 3.10)

project(netaos)
file(COPY cgroup.yaml 
    DESTINATION  ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/conf  FOLLOW_SYMLINK_CHAIN)
file(COPY notify_main
    DESTINATION  ${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/bin  FOLLOW_SYMLINK_CHAIN)
#file(COPY cgroup.yaml 
#    DESTINATION  ${RELEASE_DIR}/conf  FOLLOW_SYMLINK_CHAIN)
#file(COPY notify_main
#    DESTINATION  ${RELEASE_DIR}/bin  FOLLOW_SYMLINK_CHAIN)
