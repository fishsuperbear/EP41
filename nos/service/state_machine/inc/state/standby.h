#ifndef _SAMPLE_STATE_STANDBY_H_
#define _SAMPLE_STATE_STANDBY_H_

#include "state/state_base.h"
#include "subfsm/sub_fsm_standby.h"


class StandbyFapaParkingIn : public StateBase
{
public:
    StandbyFapaParkingIn() : StateBase() {
        sub_fsm_standby_ = std::make_unique<FsmStandbyFapaParkingIn>();

        if (sub_fsm_standby_ != nullptr)
            sub_fsm_standby_->Initialize();
    }
    virtual ~StandbyFapaParkingIn() {}

    void Register(StateManager* sm) {
        if (sub_fsm_standby_ != nullptr)
            sub_fsm_standby_->Register(sm);
    }

    virtual void Process() {
        if (sub_fsm_standby_ != nullptr)
            sub_fsm_standby_->fsm.command(FsmStandbyFapaParkingIn::tick);
    }

private:
    std::unique_ptr<FsmStandbyFapaParkingIn> sub_fsm_standby_;
};

#endif