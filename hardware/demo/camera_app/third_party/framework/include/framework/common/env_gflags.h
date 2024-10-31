// Copyright (c) 2008, Hozonauto Inc.
// All rights reserved.

// 增加 libcyber.so 中定义的 gflags 变量，在可执行文件中通过环境变量进行配置
// cyber log 中配置日志严重级别、多线程日志的线程数、是否输出到屏幕等，
// 需要支持环境变量配置

#pragma once

#include <string>
#include <string.h>        // for memchr
#include <stdlib.h>        // for getenv

#include "gflags/gflags.h"

namespace hozonauto {
namespace netaos {

/*****************************************************************************/
/* 优先从环境变量中，读取日志配置                                              */
/*****************************************************************************/
#define EnvToString(envname, default)   \
  (!getenv(envname) ? (default) : getenv(envname))

#define EnvToBool(envname, default)   \
  (!getenv(envname) ? (default) : memchr("tTyY1\0", getenv(envname)[0], 6) != NULL)

#define EnvToInt(envname, default)  \
  (!getenv(envname) ? (default) : strtol(getenv(envname), NULL, 10))

#define EnvToUint(envname, default)  \
  (!getenv(envname) ? (default) : strtol(getenv(envname), NULL, 10))

/*****************************************************************************/
/* 封装一层 gflags 定义变量的方式，支持从环境变量中读取                         */
/*****************************************************************************/
#define NETA_DEFINE_bool(name, value, meaning) \
  DEFINE_bool(name, EnvToBool("NETA_" #name, value), meaning)

#define NETA_DEFINE_int32(name, value, meaning) \
  DEFINE_int32(name, EnvToInt("NETA_" #name, value), meaning)

#define NETA_DEFINE_uint32(name, value, meaning) \
  DEFINE_uint32(name, EnvToUint("NETA_" #name, value), meaning)

#define NETA_DEFINE_string(name, value, meaning) \
  DEFINE_string(name, EnvToString("NETA_" #name, value), meaning)

}
}