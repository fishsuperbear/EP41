#pragma once

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <ara/core/initialization.h>
#include "adf/include/log.h"
#include "type.h"
#include "impl_type_idt_hpp_info_struct.h"
#include "impl_type_idt_hpp_location_struct.h"
#include "impl_type_idt_hpp_mapobjectdisplay_struct.h"
#include "impl_type_idt_hpp_path_struct.h"
#include "impl_type_idt_ins_info_struct.h"
#include "impl_type_idt_nns_info_struct.h"

#include "si_adasdataservice_skeleton.h"
#include "si_apadataservice_skeleton.h"

#include "proto/localization/localization.pb.h"
#include "proto/perception/perception_obstacle.pb.h"
#include "proto/perception/transport_element.pb.h"
#include "proto/planning/planning.pb.h"
#include "proto/statemachine/state_machine.pb.h"
#include "proto/perception/transport_element.pb.h"


#define DRIVING_MODE (1)
#define PARKING_MODE (2)

using AdasSkeleton = ap_datatype_package::v0::skeleton::SI_ADASdataServiceSkeleton;
using ApaSkeleton = ap_datatype_package::v0::skeleton::SI_APAdataServiceSkeleton;

namespace hozon {
namespace netaos {
namespace extra {
class Server : public adf::NodeBase {
   public:
    ~Server() = default;

    static Server* Instance(){
        static Server server;
        return &server;
    };

    int32_t AlgInit() override;

    int32_t AlgProcess1(adf::NodeBundle* input, const adf::ProfileToken& token, const std::string& trigger);
    
    void AlgRelease() override;
    
    void SetMode(const uint8_t mode) { running_mode_ =  mode; };
    void SetTest(const bool isTest) { isTest_ = isTest; };

    void NnpLaneProcess(adf::NodeBundle& input, const std::string& topic);
    void HppLaneProcess(adf::NodeBundle& input, const std::string& topic);
    void LocationProcess(adf::NodeBundle& input, const std::string& topic);
    void FstObjectProcess(adf::NodeBundle& input, const std::string& topic);
    void ParkingObjectProcess(adf::NodeBundle& input, const std::string& topic);
    void PlanningProcess(adf::NodeBundle& input, const std::string& topic);

   private:
    Server() = default;
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    std::shared_ptr<ap_datatype_package::datatypes::IDT_ADAS_Dataproperties_Struct> GetAdasData() {
        std::lock_guard<std::mutex> lock(adas_mutex_);
        return send_adas_data_;
    };

    std::shared_ptr<ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct> GetApaData() {
        std::lock_guard<std::mutex> lock(apa_mutex_);
        return send_apa_data_;
    };

    std::shared_ptr<ap_datatype_package::datatypes::IDT_HPP_Path_Struct> GetHppPathData() {
        std::lock_guard<std::mutex> lock(hpp_path_mutex_);
        return send_hpp_path_data_;
    };

    std::shared_ptr<ap_datatype_package::datatypes::IDT_HPP_Location_Struct> GetHppLocation() {
        std::lock_guard<std::mutex> lock(hpp_location_mutex);
        return send_hpp_location_data_;
    };

    std::shared_ptr<ap_datatype_package::datatypes::IDT_HPP_MapObjectDisplay_struct> GetHppMap() {
        std::lock_guard<std::mutex> lock(hpp_map_mutex);
        return send_hpp_map_object_display_data_;
    };

    void SetLaneData(const std::shared_ptr<hozon::perception::TransportElement>);
    void SetLocationData(const std::shared_ptr<hozon::localization::Localization>);
    void SetPerceptionObstaclesData(const std::shared_ptr<hozon::perception::PerceptionObstacles>);
    void SetADCTrajectoryData(const std::shared_ptr<hozon::planning::ADCTrajectory>);
    
    void Pub50Ms();

    // 发给座舱的server端
    std::shared_ptr<AdasSkeleton> adas_skeleton_;
    std::shared_ptr<ApaSkeleton> apa_skeleton_;

    // 高亮障碍物
    std::unordered_map<uint32_t,uint32_t> dynamic_highlight_target; // {id, is_high_light}

    // 发给座舱的数据s
    std::shared_ptr<ap_datatype_package::datatypes::IDT_ADAS_Dataproperties_Struct> send_adas_data_;  // 智能驾驶行车数据，notify
    std::shared_ptr<ap_datatype_package::datatypes::IDT_APA_Dataproperties_Struct> send_apa_data_;    // 智能驾驶泊车数据, notify

    std::shared_ptr<ap_datatype_package::datatypes::IDT_HPP_Path_Struct> send_hpp_path_data_;                             // 记忆泊车路线数据
    std::shared_ptr<ap_datatype_package::datatypes::IDT_HPP_Location_Struct> send_hpp_location_data_;                     // 记忆泊车当前定位信息
    std::shared_ptr<ap_datatype_package::datatypes::IDT_HPP_MapObjectDisplay_struct> send_hpp_map_object_display_data_;   // 记忆泊车实时显示信息

    // 座舱反馈的数据
    std::shared_ptr<ap_datatype_package::datatypes::IDT_HPP_Info_Struct> sub_hpp_info_data_;  // 座舱反馈当前道路信息
    std::shared_ptr<ap_datatype_package::datatypes::IDT_NNS_Info_Struct> sub_nns_info_data_;  // 智能召唤路线数据
    std::shared_ptr<ap_datatype_package::datatypes::IDT_Ins_Info_Struct> sub_ins_info_data_;  // 组合导航与J02偏转坐标数据
    std::vector<std::thread> works_;
    
    std::mutex adas_mutex_;
    std::mutex apa_mutex_;

    std::mutex hpp_path_mutex_;
    std::mutex hpp_location_mutex;
    std::mutex hpp_map_mutex;

    bool serving_ = true;

    bool isTest_;
    uint8_t running_mode_;
};
}  // namespace extra
}  // namespace netaos
}  // namespace hozon