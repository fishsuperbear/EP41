#pragma once

#include "fsm.hpp"
#include "sm_comm.h"
#include "global_Info.h"
#include "state_mgr.h"
#include "config_param.h"

class StateManager;
class FsmReset {
public:
    enum {
        STEP_0=0,
        STEP_1,
        PER_RESET,
        FINISH,
        RESET,
        tick=100,
        FAIL=999,
    };

    fsm::stack fsm;
    int32_t tick_count = 0;
    int32_t is_initial =0;

    FsmReset() {}
    ~FsmReset() {}

    StateManager* state_mgr_;
    void Register(StateManager* sm) {
        state_mgr_ = sm;
    }

    void Initialize() {
        fsm.on(STEP_1, 'init') = [&]( const fsm::args &args ) {
            tick_count = 0;
            G_ParkingTask()->mode = ParkingMode::init;  // 记忆泊车
            G_ParkingTask()->APA_funmode = APA_FunMode::NO_DEF;
            NODE_LOG_DEBUG << "gtx-> FsmReset step 1 init tick_count =" << tick_count;
        };
        fsm.on(STEP_1, 'quit') = [&]( const fsm::args &args ) {
            NODE_LOG_DEBUG << "fsm-> FsmReset step1 quit tick_count =" << tick_count ;
        };
        fsm.on(STEP_1, 'push') = [&]( const fsm::args &args ) {
            NODE_LOG_DEBUG << "fsm-> FsmReset step1 pushing current task.";
        };
        fsm.on(STEP_1, 'back') = [&](const fsm::args &args ) {
            NODE_LOG_DEBUG << "fsm-> FsmReset step1 back from another task ";
        };
        fsm.on(STEP_1, tick) = [&]( const fsm::args &args ) {
            NODE_LOG_INFO << "gtx-> FsmReset STEP_1 tick=" << tick_count;
            // G_Out_SM_Pnc()->hpp_command.system_command = 0x0;
            // G_Out_SM_Pnc()->pilot_status.processing_status = 0x00;  // init
            G_Out_SM_Pnc()->hpp_command.system_reset =0x01;
            if (G_In_SM_Pnc()->pnc_control_state.pnc_warninginfo != 0) {
               G_Out_ApaChassis()->ADCS8_PA_warninginfo = G_In_SM_Pnc()->pnc_control_state.pnc_warninginfo;
            }

            if (G_In_SM_Pnc()->pnc_control_state.pnc_run_state == 0x00) {
                NODE_LOG_INFO << "fsm-> FsmReset STEP_1 -> PER_RESET";
                fsm.set(PER_RESET);
            }

            if (tick_count++ > 500) {
                NODE_LOG_ERROR << "fsm-> FsmReset STEP_1 tick_count > 500";
                NODE_LOG_ERROR << "fsm-> FsmReset 10s timeout, STEP_1 -> PER_RESET";
                tick_count = 0;
                fsm.set(PER_RESET);
            }
        };
        fsm.on(PER_RESET, 'init') = [&]( const fsm::args &args ) {
            tick_count = 0;
            NODE_LOG_INFO << "fsm-> FsmReset PER_RESET init tick_count =" << tick_count ;
        };
        fsm.on(PER_RESET, tick) = [&]( const fsm::args &args ) {
            NODE_LOG_INFO << "fsm-> FsmReset PER_RESET tick=" << tick_count;
            G_Out_SM_Per()->hpp_command = {0};
            // G_Out_SM_Per()->hpp_command.system_command = 0x0;
            // G_Out_SM_Per()->pilot_status.processing_status = 0x00;  // init
            G_Out_SM_Per()->hpp_command.enable_parking_slot_detection = 0x0;
            G_Out_SM_Per()->hpp_command.enable_object_detection = 0x0;
            G_Out_SM_Per()->hpp_command.enable_freespace_detection = 0x0;
            G_Out_SM_Per()->hpp_command.enable_uss = 0x0;
            G_Out_SM_Per()->hpp_command.enable_radar = 0x0;
            G_Out_SM_Per()->hpp_command.enable_lidar = 0x0;
            G_Out_SM_Per()->hpp_command.system_reset =0x01;

            if (G_In_SM_Per()->hpp_perception_status.processing_status == 0x02) {
                NODE_LOG_ERROR << "fsm-> FsmReset PER_RESET -> FINISH";
                fsm.set(FINISH);
            }

            if (tick_count++ > 500) {
                NODE_LOG_ERROR << "fsm-> FsmReset PER_RESET tick_count > 500";
                tick_count = 0;
                fsm.set(FINISH);
            }
        };
        fsm.on(FINISH, 'init') = [&]( const fsm::args &args ) {
            tick_count = 0;
            NODE_LOG_INFO << "fsm-> FsmReset FINISH init tick_count =" << tick_count ;
        };
        fsm.on(FINISH, tick) = [&]( const fsm::args &args ) {
            NODE_LOG_INFO << "fsm-> FsmReset FINISH tick=" << tick_count;
            G_Out_SM_Per()->hpp_command.system_reset =0x0;
            G_Out_SM_Pnc()->hpp_command.system_reset =0x0;

            if (G_In_SM_Per()->hpp_perception_status.processing_status == 0x02  // 停止
            && G_In_SM_Pnc()->pnc_control_state.pnc_run_state == 0x00) {    // standby
                NODE_LOG_INFO << "fsm-> FsmReset FINISH to RESET.";
                fsm.set(RESET);
            } else {
                NODE_LOG_ERROR << "fsm-> FsmReset FINISH per status=" << G_In_SM_Per()->hpp_perception_status.processing_status;
                NODE_LOG_ERROR << "fsm-> FsmReset FINISH pnc state=" << G_In_SM_Pnc()->pnc_control_state.pnc_run_state;
                if (tick_count > 150) {
                    fsm.set(RESET);
                }
            }

            if (tick_count++ > 500) {
                NODE_LOG_ERROR << "fsm-> FsmReset FINISH tick_count > 500";
                tick_count = 0;
            }
        };
        fsm.on(RESET, 'init') = [&]( const fsm::args &args ) {
            tick_count = 0;
            NODE_LOG_INFO << "fsm-> FsmReset RESET init tick_count =" << tick_count ;
        };
        fsm.on(RESET, tick) = [&]( const fsm::args &args ) {
            if (tick_count > 50) { // keep 1s, then off
                NODE_LOG_INFO << "fsm-> FsmReset go to off.";
                G_ParkingTask()->now_state = Parking_Work_Status::off;
                G_Out_ApaChassis()->ADCS11_PA_WorkSts = 0x0;  // off
                G_Out_ApaChassis()->ADCS4_HPA_FunctionMode = 0x0;
                G_ParkingTask()->mode = ParkingMode::init;
                G_Out_SM_Per()->hpp_command.system_command = 0x0;
                G_Out_SM_Pnc()->hpp_command.system_command = 0x0;
                G_Out_SM_Per()->pilot_status.processing_status = 0x00;  // init
                G_Out_SM_Pnc()->pilot_status.processing_status = 0x00;  // init
                G_ParkingTask()->APA_funmode = APA_FunMode::NO_DEF;//初始化时清0
                G_ParkingTask()->RPA_funmode = RPA_FunMode::NO_DEF;//初始化时清0 
                G_Out_ApaChassis()->ADCS4_ParkingswithReq = 0;
                G_Out_ApaChassis()->ADCS4_text = 0x0; //初始化时清0
                G_Out_ApaChassis()->ADCS4_PA_failinfo = 0x0;//初始化时清0
                // 功能退出的时候，需要增加发送泊车转行车逻辑
                uint8_t ivalue_mode_req = 1; // 切换至行车
                NODE_LOG_INFO << "fsm-> FsmReset ivalue_mode_req =" << ivalue_mode_req;
                hozon::netaos::cfg::ConfigParam::Instance()->SetParam<uint8_t>("system/mode_req", ivalue_mode_req);

                fsm.set(STEP_1);
            } else {
                NODE_LOG_ERROR << "fsm-> FsmReset ========RESET delay 3S to off=======";
            }

            if (tick_count++ > 500) {
                NODE_LOG_ERROR << "fsm-> FsmReset RESET tick_count > 500";
                tick_count = 0;
            }
        };
        // set initial fsm state
        fsm.set(STEP_1);
        is_initial = 1;
    }
};
