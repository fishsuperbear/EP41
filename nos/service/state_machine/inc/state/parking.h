#ifndef _SAMPLE_STATE_PARKING_H_
#define _SAMPLE_STATE_PARKING_H_

#include "state/state_base.h"
#include "subfsm/sub_fsm_parking.h"


class FapaParkingIn : public StateBase
{
public:
    FapaParkingIn() : StateBase() {
        fapa_parkingin_ = std::make_unique<FsmFapaParkingIn>();

        if (fapa_parkingin_ != nullptr)
            fapa_parkingin_->Initialize();
    }
    virtual ~FapaParkingIn() {}

    void Register(StateManager* sm) {
        if (fapa_parkingin_ != nullptr)
            fapa_parkingin_->Register(sm);
    }

    virtual void Process() {
        if (fapa_parkingin_ != nullptr)
            fapa_parkingin_->fsm.command(FsmFapaParkingIn::tick);
    }

private:
    std::unique_ptr<FsmFapaParkingIn> fapa_parkingin_;
};

#endif