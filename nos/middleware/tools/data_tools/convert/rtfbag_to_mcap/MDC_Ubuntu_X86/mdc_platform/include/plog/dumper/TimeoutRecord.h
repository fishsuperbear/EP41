/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef SRC_PLOG_API_PLOG_DUMPER_TIMEOUTRECORD_H
#define SRC_PLOG_API_PLOG_DUMPER_TIMEOUTRECORD_H

#include <vector>
#include <string>

typedef struct
{
    uint32_t writerPid;
    uint32_t readerPid;
    uint64_t msgGuid;
} msgInfo;

typedef struct
{
    uint32_t writerPid;
    uint32_t readerPid;
    std::string Guid;
} MsgInfo;

namespace rbs {
    namespace plog {


        class TimeoutRecord {
        public:
            TimeoutRecord() = default;

            ~TimeoutRecord();

            std::vector<std::string> GetTimeoutRecordInfo(msgInfo msg);

            std::vector<std::string> GetTimeoutRecordInfo(MsgInfo msg);
        };
    }
}

#endif // SRC_PLOG_API_PLOG_DUMPER_TIMEOUTRECORD_H