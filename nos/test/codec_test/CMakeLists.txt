if (${TARGET_PLATFORM} MATCHES "orin")

########################## decoder_functional_test ##########################
#module名称
set(MODULE_NAME "decoder_functional_test")
set(CMAKE_CXX_FLAGS " -g -O0")
set(CMAKE_C_FLAGS " -g -O0")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")

file(GLOB PROTO_SRC ${NETA_MIDDLEWARE_DIR}/../proto/*/*.pb.cc)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")


#默认编译c和c++文件
add_executable(${MODULE_NAME}
        test_decoder_functional.cpp
        ${CMAKE_SOURCE_DIR}/../proto/soc/sensor_image.pb.cc
        ${CMAKE_SOURCE_DIR}/../proto/common/header.pb.cc
        ${CMAKE_SOURCE_DIR}/../proto/common/error_code.pb.cc
)

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_SYSROOT}/include
        ${PRJ_ROOT_DIR}/middleware
        ${COMPILER_SYSROOT}/include/nvmedia_6x
        ${PRJ_ROOT_DIR}/
        )

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        nvmedia2d
        nvvic
        nvparser
        nvmedia_ide_parser
        nvmedia_ide_sci
        netaos::codec
        globalproto
        pthread
        nvscibuf
        netaos::log
        netaos::cm
)

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
        TARGETS ${MODULE_NAME} RUNTIME
        DESTINATION ${TARGET_PLATFORM}/bin
)

########################## decoder_performance_test ##########################
#module名称
set(MODULE_NAME "decoder_performance_test")
set(CMAKE_CXX_FLAGS " -g -O0")
set(CMAKE_C_FLAGS " -g -O0")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")

file(GLOB PROTO_SRC ${NETA_MIDDLEWARE_DIR}/../proto/*/*.pb.cc)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")


#默认编译c和c++文件
add_executable(${MODULE_NAME}
        test_decoder_performance.cpp
  ${PROTO_SRC}
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx  ${PROTO_SRC}
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
  ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
  ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/nvmedia/utils/scibuf_utils.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/nvmedia/utils/misc_utils.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/nvmedia/utils/log_utils.c
  ${CMAKE_SOURCE_DIR}/../proto/soc/sensor_image.pb.cc
  ${CMAKE_SOURCE_DIR}/../proto/common/header.pb.cc
  ${CMAKE_SOURCE_DIR}/../proto/common/error_code.pb.cc
  )

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)

set(NV_TOPDIR ${COMPILER_PATH}/..)
set(NV_PLATFORM_DIR ${NV_TOPDIR}/drive-linux)
set(NV_PLATFORM_SDK_INC_DIR ${NV_PLATFORM_DIR}/include)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${NETA_MIDDLEWARE_DIR}/tools/data_tools/bag/include
        ${NETA_MIDDLEWARE_DIR}/../
        ${NETA_MIDDLEWARE_DIR}/
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/orin/utils
        ${COMPILER_SYSROOT}/include/nvmedia_6x
        ${NV_PLATFORM_SDK_INC_DIR}
        ${COMPILER_SYSROOT}/include
        ${PRJ_ROOT_DIR}/
        )

# target_link_directories(${MODULE_NAME} PRIVATE netaos::codec)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libjsoncpp.so.25
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libpcap.so.1
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libglobalproto.so
        netaos::log
        nvparser
        nvmedia_ide_parser
        nvmedia_ide_sci
        nvmedia2d
        nvvic
        netaos::codec
        netaos::bag
        class_loader
        rcutils
        ament_index_cpp
        yaml-cpp
        uuid
        globalproto
        netaos::cm
        )

set(NV_TOPDIR ${COMPILER_PATH}/..)
set(NV_PLATFORM_DIR ${NV_TOPDIR}/drive-linux)
set(NV_PLATFORM_SDK_INC_DIR ${NV_PLATFORM_DIR}/include)
target_link_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_PATH}/../usr/lib
        ${COMPILER_SYSROOT}/lib
        ${NV_PLATFORM_DIR}/lib-target
        )

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

########################## encoder_nvmedia_test ##########################
#module名称
set(MODULE_NAME "encoder_nvmedia_test")
set(CMAKE_CXX_FLAGS " -g -O0")
set(CMAKE_C_FLAGS " -g -O0")

set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/output/${TARGET_PLATFORM}/lib")

file(GLOB PROTO_SRC ${NETA_MIDDLEWARE_DIR}/../proto/*/*.pb.cc)
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")


#默认编译c和c++文件
add_executable(${MODULE_NAME}
        test_encoder_nvmedia.cpp
        ${PROTO_SRC}
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx  ${PROTO_SRC}
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobuf.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/common.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/commonTypeObject.cxx
        ${NETA_MIDDLEWARE_DIR}/idl/generated/cm_protobufPubSubTypes.cxx
        ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/nvmedia/utils/scibuf_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/nvmedia/utils/misc_utils.c
        ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/nvmedia/utils/log_utils.c
        ${CMAKE_SOURCE_DIR}/../proto/soc/sensor_image.pb.cc
        ${CMAKE_SOURCE_DIR}/../proto/common/header.pb.cc
        ${CMAKE_SOURCE_DIR}/../proto/common/error_code.pb.cc
  )

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)

set(NV_TOPDIR ${COMPILER_PATH}/..)
set(NV_PLATFORM_DIR ${NV_TOPDIR}/drive-linux)
set(NV_PLATFORM_SDK_INC_DIR ${NV_PLATFORM_DIR}/include)

#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${NETA_MIDDLEWARE_DIR}/tools/data_tools/bag/include
        ${NETA_MIDDLEWARE_DIR}/../
        ${NETA_MIDDLEWARE_DIR}/
        ${NETA_THIRDPARTY_DIR}/${TARGET_PLATFORM}/protobuf/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../middleware/codec/src/orin/utils
        ${COMPILER_SYSROOT}/include/nvmedia_6x
        ${NV_PLATFORM_SDK_INC_DIR}
        ${COMPILER_SYSROOT}/include
        ${PRJ_ROOT_DIR}/
        )

# target_link_directories(${MODULE_NAME} PRIVATE netaos::codec)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        pthread
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libjsoncpp.so.25
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libpcap.so.1
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libglobalproto.so
        netaos::log
        nvparser
        nvmedia_ide_parser
        nvmedia_ide_sci
        nvmedia2d
        nvvic
        netaos::codec
        netaos::bag
        class_loader
        rcutils
        ament_index_cpp
        yaml-cpp
        uuid
        globalproto
        )

set(NV_TOPDIR ${COMPILER_PATH}/..)
set(NV_PLATFORM_DIR ${NV_TOPDIR}/drive-linux)
set(NV_PLATFORM_SDK_INC_DIR ${NV_PLATFORM_DIR}/include)
target_link_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_PATH}/../usr/lib
        ${COMPILER_SYSROOT}/lib
        ${NV_PLATFORM_DIR}/lib-target
        )

#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

endif()

########################## encoder_cpu_test ##########################
set(MODULE_NAME "encoder_cpu_test")

#默认编译c和c++文件
add_executable(${MODULE_NAME}
        test_encoder_cpu.cpp
        ${CMAKE_SOURCE_DIR}/../proto/soc/sensor_image.pb.cc
        ${CMAKE_SOURCE_DIR}/../proto/common/header.pb.cc
        ${CMAKE_SOURCE_DIR}/../proto/common/error_code.pb.cc
)

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/
        )

# target_link_directories(${MODULE_NAME} PRIVATE netaos::codec)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libjsoncpp.so.25
        netaos::codec
        netaos::log
        netaos::cm
        globalproto
        )
target_link_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_PATH}/../usr/lib
        ${COMPILER_SYSROOT}/lib
        ${NV_PLATFORM_DIR}/lib-target
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib
)
#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

###########################################################################



########################## encoder_cpu_test ##########################
set(MODULE_NAME "decoder_cpu_test")

#默认编译c和c++文件
add_executable(${MODULE_NAME}
        test_decoder_cpu.cpp
)

#用户可在此添加编译选项
#eg: target_compile_options(${MODULE_NAME} PRIVATE -Wall USER_COMPILE_OPTIONS)
target_compile_options(${MODULE_NAME} PRIVATE -Wall)


#用户可在此添加头文件路径
target_include_directories(${MODULE_NAME}
        PRIVATE
        ${PRJ_ROOT_DIR}/middleware
        ${PRJ_ROOT_DIR}/
        )

# target_link_directories(${MODULE_NAME} PRIVATE netaos::codec)

#用户可在此添加链接库
target_link_libraries(${MODULE_NAME}
        PRIVATE
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib/libjsoncpp.so.25
        netaos::codec
        netaos::log
        netaos::cm
        globalproto
        )
target_link_directories(${MODULE_NAME}
        PRIVATE
        ${COMPILER_PATH}/../usr/lib
        ${COMPILER_SYSROOT}/lib
        ${NV_PLATFORM_DIR}/lib-target
        ${CMAKE_SOURCE_DIR}/../output/${TARGET_PLATFORM}/lib
)
#用户可在此添加链接参数
#eg: set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed USER_LINK_OPTIONS")
set_target_properties(${MODULE_NAME} PROPERTIES LINK_FLAGS "-Wl,--as-needed")

#用户添加module依赖
#add_dependencies(${MODULE_NAME} USER_DEPENDENCIES)
install(
    TARGETS ${MODULE_NAME} RUNTIME
    DESTINATION ${TARGET_PLATFORM}/bin
)

###########################################################################