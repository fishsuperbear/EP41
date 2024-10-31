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

project(e2e)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/libcrc SOMEIP_SRC_E2E_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/libe2e SOMEIP_SRC_E2E_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/libe2exf SOMEIP_SRC_E2E_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/e2eProtect SOMEIP_SRC_E2E_FILES)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/runtimeE2E SOMEIP_SRC_E2E_FILES)

add_library(csomeip_e2e${IFEATURE_OSAL_SUFFIX} SHARED ${SOMEIP_SRC_E2E_FILES})
target_link_libraries(csomeip_e2e${IFEATURE_OSAL_SUFFIX})

install(TARGETS csomeip_e2e${IFEATURE_OSAL_SUFFIX}
        LIBRARY
        DESTINATION /usr/lib
)
