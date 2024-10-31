/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2022. All rights reserved.
 * Description:
 */

#ifndef DIAG_AGENT_DATA_IDENTIFIER_H
#define DIAG_AGENT_DATA_IDENTIFIER_H

#include <mutex>
#include <vector>
#include "diag/diag_agent/include/common/diag_agent_def.h"

namespace hozon {
namespace netaos {
namespace diag {
namespace diag_agent{

class DiagAgentDataIdentifier {
public:
    DiagAgentDataIdentifier() {}
    virtual ~DiagAgentDataIdentifier() {}

    virtual bool Read(const uint16_t did, std::vector<uint8_t>& resData) {return false;}
    virtual bool Write(const uint16_t did, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData) {return false;}

private:
    DiagAgentDataIdentifier(const DiagAgentDataIdentifier &);
    DiagAgentDataIdentifier & operator = (const DiagAgentDataIdentifier &);
};

}  // namespace diag_agent
}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_AGENT_DATA_IDENTIFIER_H
