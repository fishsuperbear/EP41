 /*
  * Copyright (c) Hozon auto Co., Ltd. 2021-2023. All rights reserved.
  * Description:  adsfi sub and pub demo base percep fusion
  */

#ifndef STATEMACHINE_INT_STATE_MACHINE_H
#define STATEMACHINE_INT_STATE_MACHINE_H


#include <thread>
#include "adf/include/node_base.h"
#include "adf/include/log.h"
#include "proto/perception/perception_parking_lot.pb.h"
#include "proto/planning/planning.pb.h"
#include "proto/statemachine/state_machine.pb.h"
#include "proto/soc/apa2mcu_chassis.pb.h"
#include "proto/soc/chassis.pb.h"
#include "proto/soc/sensor_imu_ins.pb.h"
#include "proto/soc/uss_rawdata.pb.h"
#include "global_Info.h"
#include "sm_comm.h"
#include "state_mgr.h"
#include "schedule_center.h"


inline double GetRealTimestamp() {
    struct timespec time_now;
    clock_gettime(CLOCK_REALTIME, &time_now);

    return static_cast<double>(time_now.tv_sec)
             + static_cast<double>(time_now.tv_nsec) / 1000 / 1000 / 1000;
}

using namespace hozon::netaos::log;
using namespace hozon::netaos::adf;

class StateMachine : public NodeBase {
public:
    StateMachine()
    {
        //to 规控
        frame_pnc =  std::make_shared<hozon::state::StateMachine>();
        //to 感知
        frame_perception =  std::make_shared<hozon::state::StateMachine>();

        module_pnc = {0};
        module_perception = {0};
        p_newSample = std::make_shared<ChasisSignal>();
        _sm_per = std::make_shared<AlgStateMachineFrame>();
        _sm_pnc = std::make_shared<AlgStateMachineFrame>();
        _parkinglot = std::make_shared<AlgParkingLotOutArray>();
        _avp_hmi = std::make_shared<AlgEgoHmiFrame>();
        _imuins = std::make_shared<AlgImuIns>();
        std::memset(p_newSample.get(), 0, sizeof(ChasisSignal));
        std::memset(_sm_per.get(), 0, sizeof(AlgStateMachineFrame));
        std::memset(_sm_pnc.get(), 0, sizeof(AlgStateMachineFrame));
        std::memset(_avp_hmi.get(), 0, sizeof(AlgEgoHmiFrame));
        std::memset(_imuins.get(), 0, sizeof(AlgImuIns));
        apa2chassis = {0};
    };
    ~StateMachine() {

    }
    int32_t AlgProcess1(NodeBundle* input);

    virtual void AlgRelease() {
        schdule_center_.Release();
    }
    virtual int32_t AlgInit() {
        REGISTER_PROTO_MESSAGE_TYPE("chassis", hozon::soc::Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("sm_from_planning", hozon::state::StateMachine)
        REGISTER_PROTO_MESSAGE_TYPE("sm_from_hpp_perception", hozon::state::StateMachine)
        REGISTER_PROTO_MESSAGE_TYPE("parking_lot", hozon::perception::ParkingLotOutArray)
        REGISTER_PROTO_MESSAGE_TYPE("avp_hmi", hozon::planning::ADCTrajectory)
        REGISTER_PROTO_MESSAGE_TYPE("imu_info", hozon::soc::ImuIns)
        REGISTER_PROTO_MESSAGE_TYPE("apa2chassis", hozon::soc::Apa2Chassis)
        REGISTER_PROTO_MESSAGE_TYPE("sm_to_mcu", hozon::state::StateMachine)
        REGISTER_PROTO_MESSAGE_TYPE("sm_to_hpp_perception", hozon::state::StateMachine)

        // 全局收发事件集合-单例初始化
        state_machine::GlobalMsgSet::Instance().Init();
        state_manager_ = std::make_unique<StateManager>();
        state_manager_->Init();
        schdule_center_.Init();
        return 0;
    }

private:
    // input data
    void ChassisConvert(std::shared_ptr<hozon::soc::Chassis> chassisPtr, std::shared_ptr<ChasisSignal> p_newSample);
    void EgohmiConvert(std::shared_ptr<hozon::planning::ADCTrajectory> in, std::shared_ptr<AlgEgoHmiFrame> out);
    void ParkingLotConvert(std::shared_ptr<hozon::perception::ParkingLotOutArray> in, std::shared_ptr<AlgParkingLotOutArray> out);
    void PerStateMachineFrameConvert(std::shared_ptr<hozon::state::StateMachine> in, std::shared_ptr<AlgStateMachineFrame> out);
    void PncStateMachineFrameConvert(std::shared_ptr<hozon::state::StateMachine> in, std::shared_ptr<AlgStateMachineFrame> out);
    void ImuConvert(std::shared_ptr<hozon::soc::ImuIns> in, std::shared_ptr<AlgImuIns> out);

    // output data
    void PerMsgSend(std::shared_ptr<AlgStateMachineFrame> _persm, NodeBundle& _output, uint32_t tick, double msec);
    void PncMsgSend(std::shared_ptr<AlgStateMachineFrame> _pncsm, NodeBundle& _output, uint32_t tick, double msec);

    void ChassisMsgSend(std::shared_ptr<APA2Chassis> _apa2chassis, NodeBundle& _output);

    std::shared_ptr<hozon::state::StateMachine>   frame_pnc, frame_perception;
    module_status module_pnc, module_perception;
    std::shared_ptr<ChasisSignal> p_newSample;
    std::shared_ptr<AlgStateMachineFrame> _sm_per;
    std::shared_ptr<AlgStateMachineFrame> _sm_pnc;
    std::shared_ptr<AlgParkingLotOutArray> _parkinglot;
    std::shared_ptr<AlgEgoHmiFrame> _avp_hmi;
    std::shared_ptr<AlgImuIns> _imuins;
    APA2Chassis apa2chassis;

    std::unique_ptr<StateManager> state_manager_;
    ScheduleCenter schdule_center_;
};
#endif