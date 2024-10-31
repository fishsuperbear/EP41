# /**
# 
# * Copyright @ 2020 - 2027 iAuto Software(Shanghai) Co., Ltd.
# 
# * All Rights Reserved.
# 
# *
# 
# * Redistribution and use in source and binary forms, with or without
# 
# * modification, are NOT permitted except as agreed by
# 
# * iAuto Software(Shanghai) Co., Ltd.
# 
# *
# 
# * Unless required by applicable law or agreed to in writing, software
# 
# * distributed under the License is distributed on an "AS IS" BASIS,
# 
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 
# */
cmake_minimum_required(VERSION 2.8)

project(bufferlist${IFEATURE_OSAL_SUFFIX})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=gnu++14")

include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)

set(SRC_BUFFER_FILES
	${CMAKE_SOURCE_DIR}/src/memory.c
	${CMAKE_SOURCE_DIR}/src/buffer.c
)

add_library(bufferlist${IFEATURE_OSAL_SUFFIX} SHARED ${SRC_BUFFER_FILES})

install(TARGETS bufferlist${IFEATURE_OSAL_SUFFIX}
    LIBRARY
    DESTINATION /usr/lib
)


################################ test ################################
set(SRC_LIST_FILES
    ${CMAKE_SOURCE_DIR}/tests/tst_list.c
)

set(SRC_MEMORY_FILES
    ${CMAKE_SOURCE_DIR}/tests/tst_memory.c
)


set(SRC_BUFFER_FILES
    ${CMAKE_SOURCE_DIR}/tests/tst_buffer.c
)

set(SRC_PERFORMANCE_FILES
    ${CMAKE_SOURCE_DIR}/tests//tst_performance.cpp
)

set(SRC_EXAMPLE_BUFFER_FILES
    ${CMAKE_SOURCE_DIR}/tests//example_buffer.c
)

set(SRC_EXAMPLE_MEMORY_FILES
    ${CMAKE_SOURCE_DIR}/tests//example_memory.c
)


add_executable(tst_list${IFEATURE_OSAL_SUFFIX} ${SRC_LIST_FILES})
target_link_libraries(tst_list${IFEATURE_OSAL_SUFFIX} bufferlist${IFEATURE_OSAL_SUFFIX})

add_executable(tst_memory${IFEATURE_OSAL_SUFFIX} ${SRC_MEMORY_FILES})
target_link_libraries(tst_memory${IFEATURE_OSAL_SUFFIX} bufferlist${IFEATURE_OSAL_SUFFIX})

add_executable(tst_bufferlist${IFEATURE_OSAL_SUFFIX} ${SRC_BUFFER_FILES})
target_link_libraries(tst_bufferlist${IFEATURE_OSAL_SUFFIX} bufferlist${IFEATURE_OSAL_SUFFIX})

add_executable(tst_performance${IFEATURE_OSAL_SUFFIX} ${SRC_PERFORMANCE_FILES})
target_link_libraries(tst_performance${IFEATURE_OSAL_SUFFIX} bufferlist${IFEATURE_OSAL_SUFFIX})

add_executable(example_bufferlist${IFEATURE_OSAL_SUFFIX} ${SRC_EXAMPLE_BUFFER_FILES})
target_link_libraries(example_bufferlist${IFEATURE_OSAL_SUFFIX} bufferlist${IFEATURE_OSAL_SUFFIX})

add_executable(example_memory${IFEATURE_OSAL_SUFFIX} ${SRC_EXAMPLE_MEMORY_FILES})
target_link_libraries(example_memory${IFEATURE_OSAL_SUFFIX} bufferlist${IFEATURE_OSAL_SUFFIX})


install(TARGETS tst_list${IFEATURE_OSAL_SUFFIX}
    RUNTIME
    DESTINATION /usr/bin
)

install(TARGETS tst_memory${IFEATURE_OSAL_SUFFIX}
    RUNTIME
    DESTINATION /usr/bin
)

install(TARGETS tst_bufferlist${IFEATURE_OSAL_SUFFIX}
    RUNTIME
    DESTINATION /usr/bin
)

install(TARGETS tst_performance${IFEATURE_OSAL_SUFFIX}
    RUNTIME
    DESTINATION /usr/bin
)

install(TARGETS example_bufferlist${IFEATURE_OSAL_SUFFIX}
    RUNTIME
    DESTINATION /usr/bin
)

install(TARGETS example_memory${IFEATURE_OSAL_SUFFIX}
    RUNTIME
    DESTINATION /usr/bin
)



