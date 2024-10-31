#ifndef PROC_TYPES_H
#define PROC_TYPES_H

#include <string>

namespace hozon {
namespace netaos {
namespace em {

using namespace std;

enum class ProcessState : uint32_t {
    IDLE = 0,
    STARTING = 1,
    RUNNING = 2,
    TERMINATING = 3,
    TERMINATED = 4,
    ABORTED = 5
};

enum class ExecutionState : uint32_t {
    kDefault = 0,
    kRunning = 1,
    kTerminating = 2
};

enum class ResultCode : int32_t {
    kSuccess = 0,        /* 执行成功 */
    kFailed = -1,        /* 执行失败 */
    kInvalid = -2,       /* 入参非法 */
    kRejected = -3,      /* 请求拒绝 */
    kTimeout = -4        /* 请求超时 */
};

struct ProcessInfo {
    uint32_t group;
    std::string procname;
    ProcessState procstate;
};

#define REQUEST_CODE_REPORT_STATE 0x8001
#define REPLY_CODE_REPORT_STATE 0x9001

#define ENVRION_NAME "ENVIRON_APPNAME="

}}}
#endif
