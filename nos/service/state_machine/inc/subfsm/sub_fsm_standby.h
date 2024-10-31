#pragma once

#include "fsm.hpp"
#include "sm_comm.h"
#include "global_Info.h"


class FsmStandbyFapaParkingIn {
public:
    enum {
        STEP_0=0,
        STEP_1,
        STEP_2,
        STEP_3,
        STEP_4,
        STEP_5,
        STEP_6,
        STEP_7,
        RESET,
        tick=100,
        FAIL=999,
    };

    fsm::stack fsm;
    int32_t tick_count = 0;
    int32_t is_initial =0;

    FsmStandbyFapaParkingIn() {}
    ~FsmStandbyFapaParkingIn() {}

    StateManager* state_mgr_;
    void Register(StateManager* sm) {
        state_mgr_ = sm;
    }

    void Initialize() {
        fsm.on(STEP_1, 'init') = [&]( const fsm::args &args ) {
            tick_count = 0;
            NODE_LOG_DEBUG << "fsm-> Standby FapaParkingIn step 1 init tick_count =" << tick_count;
        };
        fsm.on(STEP_1, 'quit') = [&]( const fsm::args &args ) {
            NODE_LOG_DEBUG << "fsm-> Standby FapaParkingIn step1 quit tick_count =" << tick_count ;
        };
        fsm.on(STEP_1, 'push') = [&]( const fsm::args &args ) {
            NODE_LOG_DEBUG << "fsm-> Standby FapaParkingIn step1 pushing current task.";
        };
        fsm.on(STEP_1, 'back') = [&](const fsm::args &args ) {
            NODE_LOG_DEBUG << "fsm-> Standby FapaParkingIn step1 back from another task ";
        };
        fsm.on(STEP_1, tick) = [&]( const fsm::args &args ) {
            NODE_LOG_INFO << "fsm-> Standby FapaParkingIn STEP_1 tick=" << tick_count;

            if (tick_count++ > 500) {
                NODE_LOG_ERROR << "fsm-> Standby FapaParkingIn STEP_1 tick_count > 500";
                tick_count = 0;
            }
        };
        fsm.on(STEP_2, 'init') = [&]( const fsm::args &args ) {
            tick_count = 0;
            NODE_LOG_INFO << "fsm-> Standby FapaParkingIn STEP_2 init tick_count =" << tick_count ;
        };
        fsm.on(STEP_2, tick) = [&]( const fsm::args &args ) {
            NODE_LOG_INFO << "fsm-> Standby FapaParkingIn STEP_2 tick=" << tick_count;

            if (tick_count++ > 500) {
                NODE_LOG_ERROR << "fsm-> Standby FapaParkingIn STEP_2 tick_count > 500";
                tick_count = 0;
            }
        };
        // set initial fsm state
        fsm.set(STEP_1);
        is_initial = 1;
    }
};