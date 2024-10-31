#ifndef SAMPLE_STATE_CRUISING_H
#define SAMPLE_STATE_CRUISING_H

#include "state/state_base.h"
#include "subfsm/sub_fsm_ntp.h"

class Cruising : public StateBase
{
public:
    Cruising() : StateBase() {
        sub_fsm_cruising_ = std::make_unique<FsmCruising>();

        if (sub_fsm_cruising_ != nullptr)
            sub_fsm_cruising_->Initialize();
    }
    virtual ~Cruising() {
    }

    void Register(StateManager* sm) {
        if (sub_fsm_cruising_ != nullptr)
            sub_fsm_cruising_->Register(sm);
    }

    virtual void Process() {
        if (sub_fsm_cruising_ != nullptr)
            sub_fsm_cruising_->fsm.command(FsmCruising::tick);
    }

private:
    std::unique_ptr<FsmCruising> sub_fsm_cruising_;
};

#endif