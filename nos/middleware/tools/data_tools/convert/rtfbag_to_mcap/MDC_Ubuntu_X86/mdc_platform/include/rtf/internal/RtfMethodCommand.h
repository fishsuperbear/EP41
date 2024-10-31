/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This is the head file of class RtfEventCommand.
 *      RtfEventCommand will provide all of the Rtf event command. Command format is like this:
 *      rftevent help, rtfevent list, rtfevent info [event name], and so on ...
 * Create: 2019-11-19
 * Notes: NA
 */

#ifndef RTF_METHOD_COMMAND_H
#define RTF_METHOD_COMMAND_H

#include "rtf/internal/RtfCommand.h"

namespace rtf {
namespace rtfmethod {
// 错误码定义
constexpr int32_t RTF_METHOD_RET_OK      = 0;
constexpr int32_t RTF_METHOD_RET_FAILED  = -1;

// 命令等级
constexpr unsigned long RTF_METHOD_FIRST_CMD   = 1UL; // 一级命令字，如：rtfevent
constexpr unsigned long RTF_METHOD_SECOND_CMD  = 2UL; // 二级命令字，如：rtfevent list中的list
constexpr unsigned long RTF_METHOD_THIRD_CMD   = 3UL; // 三级命令字，如：rtfevent list -h中的-h

// 命令名存储下标
constexpr unsigned int RTF_METHOD_FIRST_CMD_IDX   = 0U; // 一级命令字在vector中存储的下标，如：rtfevent
constexpr unsigned int RTF_METHOD_SECOND_CMD_IDX  = 1U; // 二级命令字在vector中存储的下标，如：rtfevent list中的list
constexpr unsigned int RTF_METHOD_THIRD_CMD_IDX   = 2U; // 二级命令字在vector中存储的下标，如：rtfevent list -h中的-h

// 命令行列表
enum RtfMethodCmds : int32_t {
    RTF_METHOD_CMD_HELP = 1, // Print rtfevent command help information
    RTF_METHOD_CMD_LIST,     // Print all event list
    RTF_METHOD_CMD_INFO,     // Print event publish and subscribe information
    RTF_METHOD_CMD_TYPE,     // Show the detail information of an event
    RTF_METHOD_CMD_CALL,

    RTF_METHOD_CMD_UNKNOWN
};

// 类定义
class RtfMethodCommand : public rtf::RtfCommand {
public:
    RtfMethodCommand() noexcept;
    ~RtfMethodCommand() override = default;

    // 将命令行参数输入，执行命令并输出答应结果
    int32_t ExecuteCommand(const ara::core::Vector<ara::core::String>& paraList) override;

protected:
    // 打印命令帮助信息
    virtual void PrintHelpInfo() noexcept;

private:
    // 用于存储命令行输入参数字符串与对应的枚举值
    ara::core::Map<ara::core::String, int32_t> parameterMap_ = {
        {"--help",  RTF_METHOD_CMD_HELP},
        {"-h",      RTF_METHOD_CMD_HELP},
    };
};
} // end of namespace rtfevent
} // end of namespace rtf
#endif
