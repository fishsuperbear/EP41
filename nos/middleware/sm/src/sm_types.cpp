#include <string>
#include <map>
#include "sm/include/sm_types.h"

namespace hozon {
namespace netaos {
namespace sm {

std::map<uint32_t, std::string> _cmd_description = {
    {REQUEST_CODE_REGISTER_PREPROCESS_FUNC, "REQUEST_CODE_REGISTER_PREPROCESS_FUNC"},
    {REQUEST_CODE_REGISTER_POSTPROCESS_FUNC, "REQUEST_CODE_REGISTER_POSTPROCESS_FUNC"},
    {REQUEST_CODE_SWITCH_MODE, "REQUEST_CODE_SWITCH_MODE"},
    {REQUEST_CODE_GET_CURR_MODE, "REQUEST_CODE_GET_CURR_MODE"},
    {REQUEST_CODE_SET_DEFAULT_MODE, "REQUEST_CODE_SET_DEFAULT_MODE"},
    {REQUEST_CODE_GET_PROC_INFO, "REQUEST_CODE_GET_PROC_INFO"},
    {REQUEST_CODE_GET_MODE_LIST, "REQUEST_CODE_GET_MODE_LIST"},
    {REQUEST_CODE_PROC_RESTART, "REQUEST_CODE_PROC_RESTART"},
    {REQUEST_CODE_STOP_MODE, "REQUEST_CODE_STOP_MODE"},
    {REPLY_CODE_REGISTER_PREPROCESS_FUNC, "REPLY_CODE_REGISTER_PREPROCESS_FUNC"},
    {REPLY_CODE_REGISTER_POSTPROCESS_FUNC, "REPLY_CODE_REGISTER_POSTPROCESS_FUNC"},
    {REPLY_CODE_SWITCH_MODE, "REPLY_CODE_SWITCH_MODE"},
    {REPLY_CODE_GET_CURR_MODE, "REPLY_CODE_GET_CURR_MODE"},
    {REPLY_CODE_SET_DEFAULT_MODE, "REPLY_CODE_SET_DEFAULT_MODE"},
    {REPLY_CODE_GET_PROC_INFO, "REPLY_CODE_GET_PROC_INFO"},
    {REPLY_CODE_GET_MODE_LIST, "REPLY_CODE_GET_MODE_LIST"},
    {REPLY_CODE_PROC_RESTART, "REPLY_CODE_PROC_RESTART"},
    {REPLY_CODE_STOP_MODE, "REPLY_CODE_STOP_MODE"},
    {REQUEST_CODE_PREPROCESS_FUNC, "REQUEST_CODE_PREPROCESS_FUNC"},
    {REQUEST_CODE_POSTPROCESS_FUNC, "REQUEST_CODE_POSTPROCESS_FUNC"},
    {REPLY_CODE_PREPROCESS_FUNC, "REPLY_CODE_PREPROCESS_FUNC"},
    {REPLY_CODE_POSTPROCESS_FUNC, "REPLY_CODE_POSTPROCESS_FUNC"}
};

std::string FormatType(const uint32_t& type)
{
    char s[100];
    sprintf(s, "[%X](%s)", type, _cmd_description[type].c_str());
    return s;
}

} // namespace sm
} // namespace netaos
} // namespace hozon