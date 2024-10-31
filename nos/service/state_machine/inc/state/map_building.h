#ifndef SAMPLE_STATE_MAP_BUILDING_H
#define SAMPLE_STATE_MAP_BUILDING_H

#include "state/state_base.h"
#include "subfsm/sub_fsm_ntp.h"

class MapBuilding : public StateBase
{
public:
    MapBuilding() : StateBase() {
        sub_fsm_building_ = std::make_unique<FsmMapBuilding>();

        if (sub_fsm_building_ != nullptr)
            sub_fsm_building_->Initialize();
    }
    virtual ~MapBuilding() {
    }

    void Register(StateManager* sm) {
        if (sub_fsm_building_ != nullptr)
            sub_fsm_building_->Register(sm);
    }

    virtual void Process() {
        if (sub_fsm_building_ != nullptr)
            sub_fsm_building_->fsm.command(FsmMapBuilding::tick);
    }

private:
    std::unique_ptr<FsmMapBuilding> sub_fsm_building_;
};

#endif