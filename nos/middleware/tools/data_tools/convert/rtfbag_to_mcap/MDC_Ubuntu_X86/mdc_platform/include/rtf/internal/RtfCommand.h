/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2019. All rights reserved.
 * Description:
 *      This file is the implement of class RtfCommand.
 *      RtfCommand will provide rtf command parse function.
 * Create: 2019-11-29
 * Notes: NA
 */
#ifndef RTF_COMMAND_H
#define RTF_COMMAND_H

#include <iostream>

#include "ara/core/map.h"
#include "ara/core/string.h"
#include "ara/core/vector.h"

namespace rtf {
enum RtfCmdOptNameType : int32_t {
    OPT_TYPE_SHORT_NAME = 1,  // option type is short name
    OPT_TYPE_LONG_NAME,

    OPT_TYPE_UNKNOWN
};

struct RtfCmdOpt{
    int32_t optID;
    bool hasValue; // 是否需要带选项值，true表示需求，false标识只需要选项名
    ara::core::String optShortName;
    ara::core::String optLongName;
    ara::core::String optVal;
};

struct RtfCmdPara{
    int32_t paraID;
    ara::core::Vector<ara::core::String> paraVal;
};

constexpr size_t RTF_CMD_OPT_SHORT_NAME_START_IDX  = 1;    // 短选项名称在参数字符串中从第几个下标开始，举例：‘-h’
constexpr size_t RTF_CMD_OPT_LONG_NAME_START_IDX   = 2;    // 短选项名称在参数字符串中从第几个下标开始，举例：‘--help’

const ara::core::String RTF_CMD_OPT_YES = "Y";      // 无参数类命令选项在用户输入命令中有携带下来
constexpr int32_t RTF_INVALID_CMD_OPT           = __INT32_MAX__;    // 无效的命令选项ID
constexpr unsigned int RTF_MAX_OPT_SHORT_NAME_LEN_HAS_VAL = 2U; // 当选项参数用短名且为带参数选项时，用户能输入的选项字符串最大长度，如‘-w’

// 错误码定义
constexpr int32_t RTF_CMD_RET_OK      = 0;
constexpr int32_t RTF_CMD_RET_FAILED  = -1;
constexpr int32_t RTF_CMD_RET_PARA_ERROR = -2;

constexpr unsigned long MAX_EVENT_METHOD_LENGTH = 5000UL;  // 外部输入字符串最大长度

// 命令行处理和解析类定义
class RtfCommand {
public:
    RtfCommand() noexcept = default;
    virtual ~RtfCommand() = default;

    // 将命令行参数输入，执行命令并输出答应结果
    virtual int32_t ExecuteCommand(const ara::core::Vector<ara::core::String>& paraList);

    /*
    * 功能说明：对用户输入的命令行解析（命令行）
    * 输入参数：
    *       inputParaList -- 用户输入的命令行信息，已经存放在vector<string>中
    *       cmdParaStartIdx -- 命令参数在inputParaList中的起始下标。如：rtfevent hz -w event，命令占用了0/1位置，此时起始参数下标为2
    * 输入输出参数：
    *       optList     -- 命令选项列表，解析完成后会将选项值输出到该函数参数中
    *       paraList    -- 命令参数列表，命令解析完成后会将参数内容输出到该函数参数中
    * 返回值：如果解析成功则范围0.
    * 注意事项：
    *       1.该命令不会校验用户输入的参数或选项的逻辑合理性，如：xxx命令如果A/B参数是互斥的，该函数不会校验A/B的互斥性。
    *       2.如果cmdParaStartIdx输入大于inputParaList的大小，则返回成功（返回值=0）。
    */
    int32_t ParseCmdLine(const ara::core::Vector<ara::core::String> &inputParaList,
                    const size_t cmdParaStartIdx,
                    ara::core::Vector<RtfCmdOpt> &optList,
                    ara::core::Vector<RtfCmdPara> &paraList) const;

private:
    struct ParaVecList {
        ara::core::Vector<RtfCmdPara>& paraList;
        ara::core::Vector<RtfCmdOpt>& optList;
        ara::core::Vector<size_t>& multiValueOptId;
    };
    int32_t ParaOptVal(const ara::core::Vector<ara::core::String>& inputParaList,
                   ParaVecList paraVecList, size_t i, int32_t& preListID, int32_t& preOptID) const;
    // 判断该参数是否属于命令的配置选项，注意选项如果采用短名称可以将多个无参数选项放在一起"-wxx"，长名则只能1个
    RtfCmdOptNameType IsOption(const ara::core::String arg) const;
    // 选项如果采用短名称可以将多个无参数选项放在一起"-wxx"，长名则只能1个
    int32_t IsOptionCompliant(const ara::core::String arg, ara::core::Vector<RtfCmdOpt> &optList) const;
    // 从选型参数中提取内容并赋值
    int32_t GetOptionVal(const ara::core::String arg, ara::core::Vector<RtfCmdOpt> &optList, int32_t &optIDofNextPara) const;
    int32_t GetMultiValueOpt(size_t& paraIdx, ara::core::Vector<size_t>& multiValueOptId,
        ara::core::Vector<RtfCmdPara> &paraList) const;
    int32_t GetShortNameOptVal(const ara::core::String arg, ara::core::Vector<RtfCmdOpt> &optList,
                           int32_t &optIDofNextPara) const;
    int32_t GetLongNameOptVal(const ara::core::String arg,
                          ara::core::Vector<RtfCmdOpt> &optList, int32_t &optIDofNextPara) const;
    int32_t IsShortNameOptCompliant(const ara::core::String arg, ara::core::Vector<RtfCmdOpt> &optList) const noexcept;
    RtfCmdOpt* GetOptInfoByShortName(const char optShortName, ara::core::Vector<RtfCmdOpt> &optList) const noexcept;
    int32_t IsLongNameOptCompliant(const ara::core::String arg, ara::core::Vector<RtfCmdOpt> &optList) const;
    RtfCmdOpt* GetOptInfoByLongName(const ara::core::String arg, ara::core::Vector<RtfCmdOpt> &optList) const;
    RtfCmdOpt* GetOptInfoByOptID(const int32_t optID, ara::core::Vector<RtfCmdOpt> &optList) const noexcept;
};
} // end of namespace rtf
#endif
