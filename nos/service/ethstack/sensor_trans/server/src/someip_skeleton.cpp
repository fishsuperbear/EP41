
#include "someip_skeleton.h"
#include <pthread.h>
#include <sys/select.h>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <memory>
#include <mutex>
#include <ratio>
#include <unordered_map>
#include <bits/types/struct_timeval.h>
#include "ara/com/e2e/e2e_types.h"
#include "ara/com/sample_ptr.h"
#include "ara/com/serializer/transformation_props.h"
#include "proto/planning/lanemarkers_lane_line.pb.h"
#include "proto/planning/open_space_task_config.pb.h"
#include "ara/core/initialization.h"
#include "cfg/include/cfg_data_def.h"
#include "logger.h"
#include "param_config.h"
#include "skeleton_ego2mcu_chassis.h"
#include "skeleton_location.h"
#include "common.h"

namespace hozon {
namespace netaos {
namespace sensor {

SomeipSkeleton::SomeipSkeleton():
        _skeleton(nullptr),
        _running_mode(1),
        _need_stop(false),
        _init_ok(false) {
    std::shared_ptr<SkeletonEgo2McuChassis> _skeleton_ego2mcu_chassis 
        = std::make_shared<SkeletonEgo2McuChassis>();
    std::shared_ptr<SkeletonApa2McuChassis> _skeleton_apa2mcu_chassis
        = std::make_shared<SkeletonApa2McuChassis>();
    std::shared_ptr<SkeletonLaneDetection> _skeleton_lane_detection
        = std::make_shared<SkeletonLaneDetection>();
    std::shared_ptr<SkeletonLocation> _skeleton_location
        = std::make_shared<SkeletonLocation>();
    std::shared_ptr<SkeletonFusion> _skeleton_fusion
         = std::make_shared<SkeletonFusion>();
    std::shared_ptr<SkeletonApaStateMachine> _skeleton_apa_statemachime 
        = std::make_shared<SkeletonApaStateMachine>();
    memset(&_apa2mcu_chassis_data, 0, sizeof(_apa2mcu_chassis_data));
    memset(&_ego2mcu_chassis_data, 0, sizeof(_ego2mcu_chassis_data));
}
std::unordered_map<std::string, uint32_t > _someip_skeleton_map = {
    {"apa2mcu_chassis",  50},       // 50ms
    {"ego2mcu_chassis",  100},      // 100ms
};

int SomeipSkeleton::Init(uint32_t is_nnp) {
    _need_stop = false;
    ara::core::Initialize();
    std::string str = "1";
    ara::com::InstanceIdentifier instance(str.c_str());
    _skeleton = std::make_shared<Skeleton>(instance);
    _skeleton->OfferService();
    
    for(auto someip_skeleton : _someip_skeleton_map) {
        _send_thread.emplace_back(std::make_shared<std::thread>(&SomeipSkeleton::SomeipSendThread, 
                    this, someip_skeleton.first, someip_skeleton.second));
         
        pthread_setname_np(_send_thread.back()->native_handle(),
                     someip_skeleton.first.c_str());
    }
    
    SENSOR_LOG_INFO << "Someip skeleton init success.";
    _init_ok = true;
    return 0;
}

int SomeipSkeleton::Write(std::string name, adf::NodeBundle data) {
    if(!_init_ok) {
        return -1;
    }
    // idl data trans to proto
    if(name == "ego2mcu_chassis") {
        std::shared_ptr<hozon::planning::ADCTrajectory> data_proto = _skeleton_ego2mcu_chassis->Trans2Proto(data);
        if(data_proto != nullptr) {  
            {
                std::lock_guard<std::recursive_mutex> lck(_ego2mcu_chassis_mutex);
                _skeleton_ego2mcu_chassis->TransEgo2McuChassis(data_proto, _ego2mcu_chassis_data);
            }
            // SENSOR_LOG_INFO << "Someip send ego to muc chassis successful.";  

            hozon::netaos::AlgEgoToMcuFrame ego2mcu_data = {0};
            _skeleton_ego2mcu_chassis->TransEgo2Mcu(data_proto, ego2mcu_data);
            _skeleton->AlgEgoToMCU.Send(ego2mcu_data);
            // SENSOR_LOG_INFO << "Someip send planning egotomcu successful.";

            hozon::netaos::HafEgoTrajectory traj_data = {0};
            _skeleton_ego2mcu_chassis->Trans2Traj(data_proto, traj_data);
            _skeleton->TrajData.Send(traj_data);
            if(!(traj_data.header.seq % 100)) {
                SENSOR_LOG_INFO << "Someip trans ego to muc chassis / send planning egotomcu(0x800C) / trajData(0x8005) successful.";
            }
        }
    }
    else if(name == "apa2mcu_chassis") {
        std::shared_ptr<hozon::soc::Apa2Chassis> data_proto = _skeleton_apa2mcu_chassis->Trans2Proto(data);
        if(data_proto != nullptr) {  
            std::lock_guard<std::recursive_mutex> lck(_apa2mcu_chassis_mutex);  // 递归锁
            _skeleton_apa2mcu_chassis->TransApa2McuChassis(data_proto, _apa2mcu_chassis_data);
            SENSOR_LOG_INFO << "TransApa2McuChassis ok";
        }
    }
    else if(name == "ihbc") {
        std::lock_guard<std::recursive_mutex> lck(_ego2mcu_chassis_mutex);  // 递归锁
        if(_skeleton_apa2mcu_chassis->TransIhbc(data, _ego2mcu_chassis_data)) {
            SENSOR_LOG_INFO << "Trans Ihbc failed.";
            return -1;
        }
        SENSOR_LOG_INFO << "Trans Ihbc ok";
    }   
    else if(name == "guard_mode") {
        std::lock_guard<std::recursive_mutex> lck(_apa2mcu_chassis_mutex);  // 递归锁
        if(_skeleton_apa2mcu_chassis->TransGuardMode(data, _apa2mcu_chassis_data)) {
            SENSOR_LOG_INFO << "Trans GuardMode failed.";
            return -1;
        }
        SENSOR_LOG_INFO << "Trans GuardMode ok";
    }
    else if(name == "mod") {
        std::lock_guard<std::recursive_mutex> lck(_apa2mcu_chassis_mutex);  // 递归锁
        if(_skeleton_apa2mcu_chassis->TransMod(data, _apa2mcu_chassis_data)) {
            SENSOR_LOG_INFO << "Trans Mod failed.";
            return -1;
        }
        SENSOR_LOG_INFO << "Trans Mod ok";
    }
    else if(name == "tsrtlr") {
        std::lock_guard<std::recursive_mutex> lck(_ego2mcu_chassis_mutex);  // 递归锁
        if(_skeleton_apa2mcu_chassis->TransTsrTlr(data, _ego2mcu_chassis_data)) {
            SENSOR_LOG_INFO << "Trans tsrtlr failed.";
            return -1;
        }
        SENSOR_LOG_INFO << "Trans tsrtlr ok";
    }
    else if(name == "parkinglot2hmi_2") {
        std::lock_guard<std::recursive_mutex> lck(_apa2mcu_chassis_mutex);  // 递归锁
        _skeleton_apa2mcu_chassis->TransParkingLot2Hmi(data, _apa2mcu_chassis_data);
        SENSOR_LOG_INFO << "TransParkingLot2Hmi ok";
    }
    else if ((name == "nnplane") 
            || (name == "hpplane")) {
        hozon::netaos::HafLaneDetectionOutArray trans_data = {0};
        int ret = _skeleton_lane_detection->Trans(name, data, trans_data);
        if(ret == 0) {
            _skeleton->SnsrFsnLaneDate.Send(trans_data);
            if(!(trans_data.header.seq % 100)) {
                SENSOR_LOG_INFO << "Someip send " << name << " FsnLaneDate(0x8007)  successful.";
            }
        }
    }
    else if ((name == "nnplocation") 
            || (name == "hpplocation")) {
        hozon::netaos::HafLocation trans_data = {0};
        int ret = _skeleton_location->Trans(name, data, trans_data);
        if(ret == 0) {
            _skeleton->PoseData.Send(trans_data);
            if(!(trans_data.header.seq % 100)) {
                SENSOR_LOG_INFO << "Someip send " << name << " PoseData(0x8006) successful.";
            }
        }
    } 
    else if ((name == "nnpobject")
            || (name == "hppobject"))  {
        hozon::netaos::HafFusionOutArray trans_data = {0};
        int ret = _skeleton_fusion->Trans(name, data, trans_data);
        if(ret == 0) {
            _skeleton->SnsrFsnObj.Send(trans_data);
            if(!(trans_data.header.seq % 100)) {
                SENSOR_LOG_INFO << "Someip send " << name << " SnsrFsnObj(0x8008) successful.";
            }
        }
    }
    else if (name == "sm2mcu") {
        hozon::netaos::APAStateMachineFrame trans_data = {0};
        int ret = _skeleton_apa_statemachime->Trans(data, trans_data);    
        if(ret == 0) {
            _skeleton->ApaStateMachine.Send(trans_data);
            SENSOR_LOG_INFO << "Someip send ApaStateMachine(0x800B) successful.";
        }
    }
    return 0;
}
#define APA2MCU_CHASSIS_INTERVAL  50  // 50 ms
int SomeipSkeleton::SomeipSendThread(std::string name, uint32_t interval) {
    struct timeval time_stamp = {0};
    time_stamp.tv_sec = interval / 1000;
    time_stamp.tv_usec = (interval % 1000) * 1000;
    uint32_t count = 0;
    uint64_t pub_last_time = 0;
    std::string someip_eventid;
    while(!_need_stop) {
        select(0, NULL, NULL, NULL, &time_stamp);
        std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        
        if(!_need_stop) {
            if (name == "apa2mcu_chassis") {
                hozon::netaos::AlgCanFdMsgFrame trans_data = {0};
                {
                    std::lock_guard<std::recursive_mutex> lck(_apa2mcu_chassis_mutex);
                    memcpy(&trans_data, &_apa2mcu_chassis_data, sizeof(trans_data));
                }
                PRINTSENSORDATA(trans_data.CANFD_Msg190.ADCS5_RPA_slot_ID_1_P0_X)
                PRINTSENSORDATA(trans_data.CANFD_Msg8F.ADCS11_Parking_WorkSts)
                PRINTSENSORDATA(trans_data.CANFD_Msg8F.ADCS11_PA_ParkingFnMd)
                PRINTSENSORDATA(trans_data.CANFD_Msg194.ADCS8_PA_FPAS_SensorFaultStsFRM)
                PRINTSENSORDATA(trans_data.CANFD_Msg8F.ADCS11_PA_StopReq)
                PRINTSENSORDATA(trans_data.CANFD_MsgFE.ADCS4_AVM_DayNightStatus)
                _skeleton->APAToMCUChassis.Send(trans_data);
               someip_eventid = "0x800D";
            }
            else if (name == "ego2mcu_chassis") {
                hozon::netaos::AlgEgoHmiFrame trans_data = {0};
                {
                    std::lock_guard<std::recursive_mutex> lck(_ego2mcu_chassis_mutex);
                    memcpy(&trans_data, &_ego2mcu_chassis_data, sizeof(trans_data));
                }
                PRINTSENSORDATA(trans_data.tsr_ihbc_info.ADCS8_TSR_StrLightColor);
                _skeleton->EgoToMCUChassis.Send(trans_data);
                someip_eventid = "0x800E";
            }
            if(!(count++ % 100) 
                && (!pub_last_time || count)) {
                struct timespec time;
                if(0 != clock_gettime(CLOCK_REALTIME, &time)) {
                    SENSOR_LOG_WARN << "clock_gettime fail ";
                }
                uint64_t courent_time = (uint64_t)time.tv_sec * 1e3 + ((uint64_t)(time.tv_nsec)/1e6);
                uint64_t diff_time = courent_time - pub_last_time;
                if(pub_last_time && (diff_time - interval * 100) > interval * 10) {
                    SENSOR_LOG_WARN << "Someip send " << name << " " << someip_eventid << ""
                            << count << " interval: " << diff_time << "ms"; 
                }
                else {
                    SENSOR_LOG_INFO << "Someip send " << name << " " << someip_eventid << " "
                            << count << " interval: " << diff_time << "ms";
                }
                pub_last_time = courent_time;
            }
        }
        double cost_time = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start_time).count();
        if(cost_time < (interval * 1000)) {
            uint64_t cost_time_micro = (interval * 1000) - cost_time;
            time_stamp.tv_sec = cost_time_micro / 10e6;
            time_stamp.tv_usec = cost_time_micro - time_stamp.tv_sec * 10e6;
        }
        else {
            time_stamp.tv_sec = 0;
            time_stamp.tv_usec = 0;
        }
    }
    return 0;
}
void SomeipSkeleton::Deinit() {
    _need_stop = true;
    for(auto th : _send_thread) {
        if(th->joinable()) {
            th->join(); 
        }
    }

    _skeleton->StopOfferService();
    ara::core::Deinitialize();
    SENSOR_LOG_INFO << "Someip stop offer service successful.";
}

}
} 
}