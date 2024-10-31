
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-depth=1200")
include_directories(
    .
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CMAKE_CURRENT_SOURCE_DIR}/includes/hozon/netaos/v1
    ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/include
)

# link_directories(${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib)

set(ADASSWCSWC_DEPLOYMENT_LIBRARY_NAME hozon_someip)

file(GLOB_RECURSE
    ADASSWCSWC_DEPLOYMENT_LIBRARY_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/hozon/netaos/v1/*.cpp
)

# 设置共享库目录
set(SHARED_LIB_DIR " ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib")

# 查找所有的共享库文件
file(GLOB SHARED_LIBS "${SHARED_LIB_DIR}/*.so")

add_library(${ADASSWCSWC_DEPLOYMENT_LIBRARY_NAME} SHARED ${ADASSWCSWC_DEPLOYMENT_LIBRARY_SRC_FILES})
target_link_libraries(${ADASSWCSWC_DEPLOYMENT_LIBRARY_NAME}
    PUBLIC
    -Wl,--start-group
    ara_com_ics            
    someip_e2e_ics  
    nlog_ics
    aracore_ics            
    dlt_ics          
    perftoolkit_ics
    ara_serialize_ics      
    ncdump_ics       
    someip_ics
    ara_someipbinding_ics  
    ncjson_ics       
    someip_serializer_ics 
    ncore_ics
    pthread 
    ssl 
    crypto
    -Wl,--end-group
 )

target_link_directories(${ADASSWCSWC_DEPLOYMENT_LIBRARY_NAME}
    PUBLIC
    ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib
)

install(TARGETS ${ADASSWCSWC_DEPLOYMENT_LIBRARY_NAME}
        LIBRARY
        DESTINATION ${TARGET_PLATFORM}/lib
)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/lib/
DESTINATION ${TARGET_PLATFORM}/lib)

install(DIRECTORY ${CMAKE_SOURCE_DIR}/service/ethstack/iauto_someip/bin/
DESTINATION ${TARGET_PLATFORM}/bin)
