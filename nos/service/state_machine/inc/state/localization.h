#ifndef SAMPLE_STATE_LOCALIZATION_H
#define SAMPLE_STATE_LOCALIZATION_H

#include "state/state_base.h"
#include "subfsm/sub_fsm_ntp.h"

class Localization : public StateBase
{
public:
    Localization() : StateBase() {
        sub_fsm_localization_ = std::make_unique<FsmLocalization>();

        if (sub_fsm_localization_ != nullptr)
            sub_fsm_localization_->Initialize();
    }
    virtual ~Localization() {
    }

    void Register(StateManager* sm) {
        if (sub_fsm_localization_ != nullptr)
            sub_fsm_localization_->Register(sm);
    }

    virtual void Process() {
        if (sub_fsm_localization_ != nullptr)
            sub_fsm_localization_->fsm.command(FsmLocalization::tick);
    }

private:
    std::unique_ptr<FsmLocalization> sub_fsm_localization_;
};

#endif