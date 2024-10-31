/**
 * @log
 */

#ifndef CYBER_COMMON_LOG_H_
#define CYBER_COMMON_LOG_H_

#include <cstdarg>
#include <string>

#include "framework/base/macros.h"
#include "framework/binary.h"
#include "framework/log_interface/macro.h"

#ifndef MODULE_NAME
extern char *__progname;
#define MODULE_NAME    ((netaos::framework::binary::GetName().empty()) ?  \
+  __progname : netaos::framework::binary::GetName().c_str())
#endif

#define ADEBUG   LOG_INTERFACE(MODULE_NAME, DEBUG)
#define AINFO    LOG_INTERFACE(MODULE_NAME, INFO)
#define AWARN    LOG_INTERFACE(MODULE_NAME, WARNING)
#define AERROR   LOG_INTERFACE(MODULE_NAME, ERROR)
#define AFATAL   LOG_INTERFACE(MODULE_NAME, FATAL)

#define AINFO_IF(cond) ALOG_IF(INFO, cond, MODULE_NAME)
#define AWARN_IF(cond) ALOG_IF(WARN, cond, MODULE_NAME)
#define AERROR_IF(cond) ALOG_IF(ERROR, cond, MODULE_NAME)
#define AFATAL_IF(cond) ALOG_IF(FATAL, cond, MODULE_NAME)

#define ALOG_IF(severity, cond, module) \
    static_cast<void>(0), \
    !(cond) ? (void)0  : netaos::framework::loginterface::LogMessageVoidify() & LOG_INTERNAL(module, severity)

#define SPD_CHECK(condition) \
    ALOG_IF(FATAL, cyber_unlikely(!(condition)), MODULE_NAME) \
        << "Check failed: " #condition ""

#define SPD_CHECK_EQ(val1, val2) SPD_CHECK(val1 == val2)
#define SPD_CHECK_NE(val1, val2) SPD_CHECK(val1 != val2)
#define SPD_CHECK_LE(val1, val2) SPD_CHECK(val1 <= val2)
#define SPD_CHECK_LT(val1, val2) SPD_CHECK(val1 < val2)
#define SPD_CHECK_GE(val1, val2) SPD_CHECK(val1 >= val2)
#define SPD_CHECK_GT(val1, val2) SPD_CHECK(val1 > val2)

#undef CHECK
#undef CHECK_EQ
#undef CHECK_NE
#undef CHECK_LE
#undef CHECK_LT
#undef CHECK_GE
#undef CHECK_GT
#define CHECK(condition)        SPD_CHECK(condition)
#define CHECK_EQ(val1, val2)    SPD_CHECK_EQ(val1, val2)
#define CHECK_NE(val1, val2)    SPD_CHECK_NE(val1, val2)
#define CHECK_LE(val1, val2)    SPD_CHECK_LE(val1, val2)
#define CHECK_LT(val1, val2)    SPD_CHECK_LT(val1, val2)
#define CHECK_GE(val1, val2)    SPD_CHECK_GE(val1, val2)
#define CHECK_GT(val1, val2)    SPD_CHECK_GT(val1, val2)

#define ACHECK(cond)      CHECK(cond)

#undef SPDLOG_FIRST_N
#define SPDLOG_FIRST_N(severity, n) SPD_LOG_FIRST_N(severity, n)

#undef SPDLOG_EVERY_N
#define SPDLOG_EVERY_N(severity, n) SPD_LOG_EVERY_N(severity, n)

#define AINFO_EVERY(freq)    SPD_LOG_EVERY_N(INFO, freq)
#define AWARN_EVERY(freq)    SPD_LOG_EVERY_N(WARNING, freq)
#define AERROR_EVERY(freq)   SPD_LOG_EVERY_N(ERROR, freq)

#define SPD_LOG_FIRST_N(severity, n) \
    static int SPDLOG_OCCURRENCES = 0; \
    if (SPDLOG_OCCURRENCES <= (n)) \
        ++SPDLOG_OCCURRENCES; \
    if (SPDLOG_OCCURRENCES <= (n))  \
        LOG_INTERFACE(MODULE_NAME, severity)

#define SPD_LOG_EVERY_N(severity, n) \
    static int SPDLOG_OCCURRENCES = 0, SPDLOG_OCCURRENCES_MOD_N = 0; \
    ++SPDLOG_OCCURRENCES; \
    if (++SPDLOG_OCCURRENCES_MOD_N > (n)) SPDLOG_OCCURRENCES_MOD_N -= (n); \
    if (SPDLOG_OCCURRENCES_MOD_N == 1) \
        LOG_INTERFACE(MODULE_NAME, severity)

#if !defined(RETURN_IF_NULL)
#define RETURN_IF_NULL(ptr)          \
  if (ptr == nullptr) {              \
    AWARN << #ptr << " is nullptr."; \
    return;                          \
  }
#endif

#if !defined(RETURN_VAL_IF_NULL)
#define RETURN_VAL_IF_NULL(ptr, val) \
  if (ptr == nullptr) {              \
    AWARN << #ptr << " is nullptr."; \
    return val;                      \
  }
#endif

#if !defined(RETURN_IF)
#define RETURN_IF(condition)           \
  if (condition) {                     \
    AWARN << #condition << " is met."; \
    return;                            \
  }
#endif

#if !defined(RETURN_VAL_IF)
#define RETURN_VAL_IF(condition, val)  \
  if (condition) {                     \
    AWARN << #condition << " is met."; \
    return val;                        \
  }
#endif

#if !defined(_RETURN_VAL_IF_NULL2__)
#define _RETURN_VAL_IF_NULL2__
#define RETURN_VAL_IF_NULL2(ptr, val) \
  if (ptr == nullptr) {               \
    return (val);                     \
  }
#endif

#if !defined(_RETURN_VAL_IF2__)
#define _RETURN_VAL_IF2__
#define RETURN_VAL_IF2(condition, val) \
  if (condition) {                     \
    return (val);                      \
  }
#endif

#if !defined(_RETURN_IF2__)
#define _RETURN_IF2__
#define RETURN_IF2(condition) \
  if (condition) {            \
    return;                   \
  }
#endif

#endif  // CYBER_COMMON_LOG_H_
