/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 * Description:
 */

#ifndef DIAG_AGENT_ROUTINE_CONTROL_H
#define DIAG_AGENT_ROUTINE_CONTROL_H

#include <mutex>
#include <vector>
#include "diag/diag_agent/include/common/diag_agent_def.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent{

class DiagAgentRoutineControl {
public:
    DiagAgentRoutineControl() {}
    virtual ~DiagAgentRoutineControl() {}

    virtual bool Start(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData) = 0;
    virtual bool Stop(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData) {return false;}
    virtual bool Result(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData) {return false;}

private:
    DiagAgentRoutineControl(const DiagAgentRoutineControl &);
    DiagAgentRoutineControl & operator = (const DiagAgentRoutineControl &);
};

}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_AGENT_ROUTINE_CONTROL_H
