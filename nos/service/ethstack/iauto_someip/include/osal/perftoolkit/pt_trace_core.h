#ifndef INCLUDE_OSAL_PERFTOOLKIT_PT_TRACE_CORE_H_
#define INCLUDE_OSAL_PERFTOOLKIT_PT_TRACE_CORE_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include <stdarg.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef __QNXNTO__
#include <process.h>
#include <sys/trace.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif


#ifndef __cplusplus
#define PT_TRACE_LIKELY(exp)    (__builtin_expect(!!(exp), 1))
#define PT_TRACE_UNLIKELY(exp)  (__builtin_expect(!!(exp), 0))
#else
#define PT_TRACE_LIKELY(exp)    (__builtin_expect(!!(exp), true))
#define PT_TRACE_UNLIKELY(exp)  (__builtin_expect(!!(exp), false))
#endif

#define PT_TRACE_FLAG_START 0x8000000000000000ull

extern unsigned long g_pt_trace_enabled_flag;
extern volatile uint64_t g_pt_trace_enabled_tags;

/* 每行记录最大1024字节 */
#define PT_TRACE_BUFFER_MAX_SIZE 1024

#ifdef __QNXNTO__
#define PT_TRACE_EVENT(str) TraceEvent(_NTO_TRACE_INSERTUSRSTREVENT, 1000, str)
#else
extern int _pt_trace_maker_writer_fd_;
#define PT_TRACE_EVENT(str) if PT_TRACE_LIKELY(-1 != _pt_trace_maker_writer_fd_) \
    write(_pt_trace_maker_writer_fd_, str, strlen(str))
#endif

#define PT_TRACE_CORE_BEGIN(str) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "B|%u|%s", (uint32_t)getpid(), str);\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_BEGINF(format, ...) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        int ret = snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "B|%u|", (uint32_t)getpid());\
        if (ret > 0 && ret < PT_TRACE_BUFFER_MAX_SIZE - 1) {\
            ret = snprintf(_pt_trace_buf+ret, sizeof(_pt_trace_buf)-ret, format, ##__VA_ARGS__);\
        }\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_BEGINVF(format, arglist) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        int ret = snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "B|%u|", (uint32_t)getpid());\
        if (ret > 0 && ret < PT_TRACE_BUFFER_MAX_SIZE - 1) {\
            ret = vsnprintf(_pt_trace_buf+ret, sizeof(_pt_trace_buf)-ret, format, arglist);\
        }\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_END() do { \
    char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
    snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "E|%u", (uint32_t)getpid());\
    PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_INSTANT(str) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "I|%u|%s", (uint32_t)getpid(), str);\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_INSTANTF(format, ...) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        int ret = snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "I|%u|", (uint32_t)getpid());\
        if (ret > 0 && ret < PT_TRACE_BUFFER_MAX_SIZE - 1) {\
            ret = snprintf(_pt_trace_buf+ret, sizeof(_pt_trace_buf)-ret, format, ##__VA_ARGS__);\
        }\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_INSTANTVF(format, arglist) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        int ret = snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "I|%u|", (uint32_t)getpid());\
        if (ret > 0 && ret < PT_TRACE_BUFFER_MAX_SIZE - 1) {\
            ret = vsnprintf(_pt_trace_buf+ret, sizeof(_pt_trace_buf)-ret, format, arglist);\
        }\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_ASYNC_START(name, cookie) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "S|%u|%s|%lu", (uint32_t)getpid(), name, (uint64_t)cookie);\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_ASYNC_FINAL(name, cookie) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "F|%u|%s|%lu", (uint32_t)getpid(), name, (uint64_t)cookie);\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_COUNTER(name, value) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "C|%u|%s|%" PRId64, (uint32_t)getpid(), name, (int64_t)value);\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_INFO(name, str) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "N|%u|%s|%s", (uint32_t)getpid(), name, str);\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_INFOF(name, format, ...) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        int ret = snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "N|%u|%s|", (uint32_t)getpid(), name);\
        if (ret > 0 && ret < PT_TRACE_BUFFER_MAX_SIZE - 1) {\
            ret = snprintf(_pt_trace_buf+ret, sizeof(_pt_trace_buf)-ret, format, ##__VA_ARGS__);\
        }\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_INFOVF(name, format, arglist) do { \
        char _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE];\
        int ret = snprintf(_pt_trace_buf, sizeof(_pt_trace_buf), "N|%u|%s|", (uint32_t)getpid(), name);\
        if (ret > 0 && ret < PT_TRACE_BUFFER_MAX_SIZE - 1) {\
            ret = vsnprintf(_pt_trace_buf+ret, sizeof(_pt_trace_buf)-ret, format, arglist);\
        }\
        _pt_trace_buf[PT_TRACE_BUFFER_MAX_SIZE-1] = '\0';\
        PT_TRACE_EVENT(_pt_trace_buf);\
    }while(0)

#define PT_TRACE_CORE_UNIQUE_NAME2_(var, line) var##line
#define PT_TRACE_CORE_UNIQUE_NAME1_(line) PT_TRACE_CORE_UNIQUE_NAME2_(_pt_trace_name_anonymous_, line)
#define PT_TRACE_CORE_UNIQUE_NAME PT_TRACE_CORE_UNIQUE_NAME1_(__LINE__)

// #define PT_TRACE_CORE_DEFINE_MODULE(name) uint64_t _pt_trace_core_module_name_##name = 0;\
// __attribute__((constructor)) static void pt_trace_core_module_ctor_() {\
//     pt_trace_core_register_module(#name,&_pt_trace_core_module_name_##name);\
// }\
// __attribute__((destructor)) static void pt_trace_core_module_dtor_() {\
//     pt_trace_core_unregister_module(#name);\
// }

// #define PT_TRACE_CORE_DECAL_FILTER(name, id) \
// inline static uint64_t pt_trace_core_check_enabled() {\
//     extern uint64_t _pt_trace_core_module_name_##name;\
//     return _pt_trace_core_module_name_##name & (1ull<<id);\
// }

// #ifdef __cplusplus
// extern "C" {
// void pt_trace_core_register_module(const char* name, uint64_t* filter);
// void pt_trace_core_unregister_module(const char* name);
// }
// #endif

#endif  // INCLUDE_OSAL_PERFTOOLKIT_PT_TRACE_CORE_H_
