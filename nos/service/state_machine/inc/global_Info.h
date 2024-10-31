#pragma once

#include <algorithm>
#include <deque>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "sm_comm.h"
#include "adf/include/log.h"
#include "adf/include/node_base.h"
#include "algdata/imu.h"
#include "algdata/egohmi.h"
#include "algdata/parkinglot.h"
#include "algdata/state_machine_frame.h"

namespace state_machine {


class GlobalMsgSet {

    struct EventSetInput {

        std::shared_ptr<ChasisSignal> p_chassis;
        std::shared_ptr<AlgStateMachineFrame> p_sm_pnc;
        std::shared_ptr<AlgStateMachineFrame> p_sm_per;
        std::shared_ptr<AlgParkingLotOutArray> p_plot;
        std::shared_ptr<AlgEgoHmiFrame> p_ego_hmi;
        std::shared_ptr<AlgImuIns> p_imuins;
    };
    struct EventSetOutput {
        std::shared_ptr<AlgStateMachineFrame> sm_to_pnc;
        std::shared_ptr<AlgStateMachineFrame> sm_to_per;
        std::shared_ptr<APA2Chassis> _apa2chassis;
    };
    struct EventSetAll {
        EventSetInput msg_in;
        EventSetOutput msg_out;
        std::shared_ptr<ParkingTask> pParkingTask;
    };


public:

    ~GlobalMsgSet() = default;
    // 全局单例
    static GlobalMsgSet& Instance(){
        static std::unique_ptr<GlobalMsgSet> instance_ptr(new GlobalMsgSet());
        return *instance_ptr;
    };

    int Init();
    int UpdateEventSetIn(std::shared_ptr<ChasisSignal> chassis,
                          std::shared_ptr<AlgStateMachineFrame> pnc,
                          std::shared_ptr<AlgStateMachineFrame> per,
                          std::shared_ptr<AlgParkingLotOutArray> slot,
                          std::shared_ptr<AlgEgoHmiFrame> ego_hmi,
                          std::shared_ptr<AlgImuIns> imuins){
         this->msg_set_->msg_in.p_chassis = chassis;
         this->msg_set_->msg_in.p_sm_pnc = pnc;
         this->msg_set_->msg_in.p_sm_per = per;
         this->msg_set_->msg_in.p_plot = slot;
         this->msg_set_->msg_in.p_ego_hmi = ego_hmi;
         this->msg_set_->msg_in.p_imuins = imuins;
         return 0;
    }
    int IsReceivedAll() {
        if (this->msg_set_->msg_in.p_chassis == nullptr) {
            return 1;
        } else if (this->msg_set_->msg_in.p_sm_pnc == nullptr) {
            return 2;
        } else if (this->msg_set_->msg_in.p_sm_per == nullptr) {
            return 3;
        } else {
            return 0;
        }
        return 0;
    }
    int MsgSetInitial() {
        msg_set_ = std::make_shared<EventSetAll>();
        this->msg_set_->msg_in.p_chassis = nullptr;
        this->msg_set_->msg_in.p_sm_pnc = nullptr;
        this->msg_set_->msg_in.p_sm_per = nullptr;
        this->msg_set_->msg_in.p_plot = nullptr;
        this->msg_set_->msg_in.p_ego_hmi = nullptr;
        this->msg_set_->msg_in.p_imuins = nullptr;
        this->msg_set_->msg_out.sm_to_per =   std::make_shared<AlgStateMachineFrame>();
        this->msg_set_->msg_out.sm_to_pnc = std::make_shared<AlgStateMachineFrame>();
        this->msg_set_->msg_out._apa2chassis = std::make_shared<APA2Chassis>();
        this->msg_set_->pParkingTask = std::make_shared<ParkingTask>();
        return 0;
    }


    int ClearMsgOut() {
        if (this->msg_set_->msg_out.sm_to_per != nullptr) {
            memset(this->msg_set_->msg_out.sm_to_per.get(),0,sizeof(AlgStateMachineFrame));
        }
        if (this->msg_set_->msg_out.sm_to_pnc != nullptr) {
            memset(this->msg_set_->msg_out.sm_to_pnc.get(),0,sizeof(AlgStateMachineFrame));
        }
        memset(this->msg_set_->msg_out._apa2chassis.get(), 0 ,sizeof(APA2Chassis));
        // memset(this->msg_set_->pParkingTask.get(), 0, sizeof(ParkingTask));
        return 0;
    }

    EventSetInput GetGlobalEventIn(){
        return this->msg_set_->msg_in;
    }
    EventSetOutput GetGlobalEventOut(){
        return this->msg_set_->msg_out;
    }

    // msg in
    std::shared_ptr<ChasisSignal> GetGlobalInChassisPtr() {
        return this->msg_set_->msg_in.p_chassis;
    }
    std::shared_ptr<AlgStateMachineFrame> GetGlobalInPncPtr() {
        return this->msg_set_->msg_in.p_sm_pnc;
    }
    std::shared_ptr<AlgStateMachineFrame> GetGlobalInPerPtr() {
        return this->msg_set_->msg_in.p_sm_per;
    }
    std::shared_ptr<AlgParkingLotOutArray> GetGlobalInSlotPtr() {
        return this->msg_set_->msg_in.p_plot;
    }
    std::shared_ptr<AlgEgoHmiFrame> GetGlobalInEgoHmiPtr() {
        return this->msg_set_->msg_in.p_ego_hmi;
    }
    std::shared_ptr<AlgImuIns> GetGlobalImuInsPtr() {
        return this->msg_set_->msg_in.p_imuins;
    }
    //out msg
    std::shared_ptr<AlgStateMachineFrame> GetGlobalOutPncPtr() {
        return this->msg_set_->msg_out.sm_to_pnc;
    }
    std::shared_ptr<AlgStateMachineFrame> GetGlobalOutPerPtr() {
        return this->msg_set_->msg_out.sm_to_per;
    }
    std::shared_ptr<APA2Chassis> GetGlobalApa2ChassisPtr() {
        return this->msg_set_->msg_out._apa2chassis;
    }
    std::shared_ptr<ParkingTask>  GetGlobalParkingTask() {
        return this->msg_set_->pParkingTask;
    }



private:
    std::shared_ptr<EventSetAll> msg_set_;
    GlobalMsgSet() = default;

};


}


#define G_MsgSetIn() state_machine::GlobalMsgSet::Instance().GetGlobalEventIn()
#define G_MsgSetOut() state_machine::GlobalMsgSet::Instance().GetGlobalEventOut()

#define G_In_Chassis() state_machine::GlobalMsgSet::Instance().GetGlobalInChassisPtr()
#define G_In_SM_Pnc() state_machine::GlobalMsgSet::Instance().GetGlobalInPncPtr()
#define G_In_SM_Per() state_machine::GlobalMsgSet::Instance().GetGlobalInPerPtr()
#define G_In_Slot() state_machine::GlobalMsgSet::Instance().GetGlobalInSlotPtr()
#define G_In_EgoHmi() state_machine::GlobalMsgSet::Instance().GetGlobalInEgoHmiPtr()
#define G_In_Imu() state_machine::GlobalMsgSet::Instance().GetGlobalImuInsPtr()

#define G_Out_SM_Pnc() state_machine::GlobalMsgSet::Instance().GetGlobalOutPncPtr()
#define G_Out_SM_Per() state_machine::GlobalMsgSet::Instance().GetGlobalOutPerPtr()
#define G_Out_ApaChassis() state_machine::GlobalMsgSet::Instance().GetGlobalApa2ChassisPtr()
#define G_ParkingTask() state_machine::GlobalMsgSet::Instance().GetGlobalParkingTask()

#define G_Is_Msg_OK() state_machine::GlobalMsgSet::Instance().IsReceivedAll()
#define G_ClearMSGOut() state_machine::GlobalMsgSet::Instance().ClearMsgOut()

#define G_UpdateAll(chassis, \
                          pnc, \
                          per, \
                          slot, \
                          ego_hmi, \
                          imu)  \
        state_machine::GlobalMsgSet::Instance().UpdateEventSetIn(chassis, \
                          pnc, \
                          per, \
                          slot, \
                          ego_hmi, \
                          imu)





// #define G_Msg_Instance() state_machine::GlobalMsgSet::Instance()

