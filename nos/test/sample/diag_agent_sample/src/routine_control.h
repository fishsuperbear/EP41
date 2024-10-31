#ifndef ROUTINE_CONTROL_H
#define ROUTINE_CONTROL_H

#include <mutex>
#include <vector>
#include "diag/diag_agent/include/service/diag_agent_routine_control.h"

using namespace hozon::netaos::diag::diag_agent;

class RoutineControl : public DiagAgentRoutineControl {
public:
    RoutineControl();
    virtual ~RoutineControl();

    virtual bool Start(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    virtual bool Stop(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);
    virtual bool Result(const uint16_t rid, const std::vector<uint8_t>& reqData, std::vector<uint8_t>& resData);

private:
    RoutineControl(const RoutineControl &);
    RoutineControl & operator = (const RoutineControl &);
};

#endif  // ROUTINE_CONTROL_H
