/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 */

#ifndef SRC_PLOG_API_PLOG_DUMPER_LATENCYLOGDUMPER_H
#define SRC_PLOG_API_PLOG_DUMPER_LATENCYLOGDUMPER_H

#include <memory>
#include <string>
#include <unistd.h>

namespace rbs {
namespace plog {
class LatencyLogDumperImpl;

class LatencyLogDumper {
public:
    LatencyLogDumper();

    ~LatencyLogDumper();

    void SetOutput(std::string path, bool separateProcessStorage = false);

    bool SetRotationParams(uint32_t nFile, uint32_t sizeInMb);

    void AddTargetPid(pid_t pid);

    void AddTargetProcessName(std::string processName);

    int StartDump(uint64_t timeInMs);

    static std::string GetLastPrintedInfo();

private:
    std::unique_ptr<LatencyLogDumperImpl> impl_;
};
}
}

#endif // SRC_PLOG_API_PLOG_DUMPER_LATENCYLOGDUMPER_H
