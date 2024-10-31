#ifndef SAMPLE_STATE_MANAGER_H
#define SAMPLE_STATE_MANAGER_H

#include <atomic>
#include <thread>
#include "global_Info.h"
#include "state/state_base.h"


class StateManager
{
public:
    StateManager();
    ~StateManager();

    void Init();
    void ExecState();

    void UpdateParkingState(PARKING_STATE state = PARKING_STATE::FAPA_PARKING_IN);
    void UpdateStandbyState(STANDBY_STATE state = STANDBY_STATE::STANDBY_FAPA_PARKING_IN);
    void UpdateResetState();

    void MapSearch();
    int32_t MapId();

private:
    void StandByStateTransition();
    void ParkingStateTransition();

    template<class T>
    void Create(std::shared_ptr<StateBase>& obj)
    {
        obj.reset(new T());
        obj->Register(this);
    }

private:
    PARKING_STATE   parking_state_;
    STANDBY_STATE   standby_state_;

    std::shared_ptr<StateBase>  reset_;
    std::shared_ptr<StateBase>  localization_;
    std::shared_ptr<StateBase>  map_building_;
    std::shared_ptr<StateBase>  ntp_cruising_;
    std::shared_ptr<StateBase>  standby_;
    std::shared_ptr<StateBase>  parking_;

    std::thread                 th_;
    std::atomic<int32_t>        map_id_;
    std::atomic<bool>           stop_;
};

#endif
