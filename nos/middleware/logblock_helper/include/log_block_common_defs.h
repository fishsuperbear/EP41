// 
// Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
// 
/// @file log_block_common_defs.h
/// @brief 
/// @author xumengjun@hozonauto.com
/// @version 1.0.0
/// @date 2023-11-07

#ifndef ___LOG_BLOCK_COMMON_DEFS_H__
#define ___LOG_BLOCK_COMMON_DEFS_H__

#include <cstdlib>
#include <unistd.h>
#include <time.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <string>

namespace hozon {
namespace netaos {
namespace logblock {

#if __GNUC__ >= 3
    #define logblock_likely(x) (__builtin_expect((x), 1))
    #define logblock_unlikely(x) (__builtin_expect((x), 0))
#else
    #define logblock_likely(x) (x)
    #define logblcok_unlikely(x) (x)
#endif

inline std::string GetHostName() {
    const uint32_t name_size = 64;
    char hostname[name_size] = {0};
    if (gethostname(hostname, sizeof(hostname))) {
        return "default";
    }
    return std::string(hostname);
}

inline unsigned long long GetTscNS() {
    unsigned long long io_tsc_ns;
    uint64_t tsc;
    __asm__ __volatile__ ("mrs %[tsc], cntvct_el0" : [tsc] "=r" (tsc));
    io_tsc_ns = tsc * 32;
    return io_tsc_ns;
}

inline unsigned long long GetRealTimeNS() {
    /*
    struct timespec now;
    clock_gettime(CLOCK_REALTIME, &now);
    return now.tv_sec * 1000000000ull + now.tv_nsec;
    */
    return GetTscNS();
}

inline unsigned long long GetVirtualTimeNS() {
    /*
    struct timespec now;
    clock_gettime(CLOCK_VIRTUAL, &now);
    return now.tv_sec * 1000000000ull + now.tv_nsec;
    */
    return GetTscNS();
}

static unsigned int LOG_HEADER_MAGIC_NUM = 0xfb709394;
static unsigned int ALIGN_PAGE_SIZE = 4096;
static unsigned int ALIGN_STEP_SIZE = 8;

static const std::string &CURRENT_HOST_NAME = GetHostName();
static unsigned int CURRENT_PROCESS_ID = getpid();
static thread_local unsigned int CURRENT_THREAD_ID = syscall(SYS_gettid);
static thread_local unsigned int CURRENT_ID = 0;

enum LOG_DATA_TYPE : int {
    DATA_TYPE_COMMON_LOG = 0,
    DATA_TYPE_NONE = 1
};

struct LogHeader {
  unsigned int magic_num;                   // magic number
  unsigned int id;                          // sequence number
  unsigned short version;                   // version
  unsigned int data_type;                   // data_type, used for deserializing logbody
  unsigned int header_len;                  // log header length
  unsigned int len;                         // log body length
  unsigned int process_id;                  // process id
  unsigned int thread_id;                   // thread id
  unsigned long long curr_realtime_ns;      // CLOCK_REALTIME
  unsigned long long curr_virtualtime_ns;   // CLOCK_VIRTUAL
  char appid[32];                           // appid
  LogHeader() : magic_num(LOG_HEADER_MAGIC_NUM), id(0), version(0),
                data_type(0), header_len(0), process_id(0), thread_id(0),
                curr_realtime_ns(0), curr_virtualtime_ns(0) {}
};

struct LogBody {
    unsigned int reserved;  // reserved field
    char content[];         // content
};

} // namespace logblock
} // namespace netaos
} // namespace hozon

#endif // ___LOG_BLOCK_COMMON_DEFS_H__
