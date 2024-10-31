#ifndef SAMPLE_STATE_RESET_H
#define SAMPLE_STATE_RESET_H

#include "state/state_base.h"
#include "subfsm/sub_fsm_reset.h"


class Reset : public StateBase
{
public:
    Reset() : StateBase() {
        sub_fsm_reset_ = std::make_unique<FsmReset>();

        if (sub_fsm_reset_ != nullptr)
            sub_fsm_reset_->Initialize();
    }
    virtual ~Reset() {}

    void Register(StateManager* sm) {
        if (sub_fsm_reset_ != nullptr)
            sub_fsm_reset_->Register(sm);
    }

    virtual void Process() {
        if (sub_fsm_reset_ != nullptr)
            sub_fsm_reset_->fsm.command(FsmReset::tick);
    }

private:
    std::unique_ptr<FsmReset> sub_fsm_reset_;
};

#endif