#ifndef SM_TYPES_H
#define SM_TYPES_H

#include <string>
#include <map>

namespace hozon {
namespace netaos {
namespace sm {

#define REQUEST_CODE_REGISTER_PREPROCESS_FUNC 0x8001
#define REQUEST_CODE_REGISTER_POSTPROCESS_FUNC 0x8002
#define REQUEST_CODE_SWITCH_MODE 0x8003
#define REQUEST_CODE_GET_CURR_MODE 0x8004
#define REQUEST_CODE_SET_DEFAULT_MODE 0x8005
#define REQUEST_CODE_GET_PROC_INFO 0x8101
#define REQUEST_CODE_GET_MODE_LIST 0x8102
#define REQUEST_CODE_PROC_RESTART 0x8103
#define REQUEST_CODE_STOP_MODE 0x8104
#define REQUEST_CODE_GET_MODE_LIST_DETAIL_INFO 0x8105
#define REPLY_CODE_REGISTER_PREPROCESS_FUNC 0x9001
#define REPLY_CODE_REGISTER_POSTPROCESS_FUNC 0x9002
#define REPLY_CODE_SWITCH_MODE 0x9003
#define REPLY_CODE_GET_CURR_MODE 0x9004
#define REPLY_CODE_SET_DEFAULT_MODE 0x9005
#define REPLY_CODE_GET_PROC_INFO 0x9101
#define REPLY_CODE_GET_MODE_LIST 0x9102
#define REPLY_CODE_PROC_RESTART 0x9103
#define REPLY_CODE_STOP_MODE 0x9104
#define REQUEST_CODE_PREPROCESS_FUNC 0x6001
#define REQUEST_CODE_POSTPROCESS_FUNC 0x6002
#define REPLY_CODE_PREPROCESS_FUNC 0x7001
#define REPLY_CODE_POSTPROCESS_FUNC 0x7002

#define ENVRION_NAME "ENVIRON_APPNAME="
#define TOPIC_SUFFIX_NAME "__processfunc_callback"

enum class SmResultCode : int32_t {
    kSuccess = 0,        /* 执行成功 */
    kFailed = -1,        /* 执行失败 */
    kInvalid = -2,       /* 入参非法 */
    kRejected = -3,      /* 请求拒绝 */
    kPreFailed = -4,     /* 前处理失败 */
    kTimeout = -5        /* 请求超时 */
};

enum class ProcessMode : uint32_t {
    PREPROCESS = 0,
    POSTPROCESS = 1
};

std::string FormatType(const uint32_t& type);
} // namespace sm
} // namespace netaos
} // namespace hozon
#endif
