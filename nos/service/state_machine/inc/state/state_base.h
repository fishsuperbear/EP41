#ifndef SAMPLE_STATE_BASE_H
#define SAMPLE_STATE_BASE_H

#include <memory>
#include "global_Info.h"
#include "state_mgr.h"
#include "state/common.h"

class StateManager;
class StateBase
{
public:
    StateBase() {}
    virtual ~StateBase() {}
    virtual void Register(StateManager* sm) = 0;
    virtual void Process() = 0;

protected:
    STANDBY_STATE   last_standby_state_ = STANDBY_STATE::STANDBY_DEFAULT;
};

#endif