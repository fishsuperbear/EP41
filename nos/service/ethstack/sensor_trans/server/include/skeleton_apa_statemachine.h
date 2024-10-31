#pragma once

#include <memory>
#include "logger.h"
#include "hozon/netaos/impl_type_apastatemachineframe.h"
#include "cm_proxy.h"

namespace hozon {
namespace netaos {
namespace sensor {
class SkeletonApaStateMachine {
public: 
    SkeletonApaStateMachine() = default;
    ~SkeletonApaStateMachine() = default;
    int Trans(adf::NodeBundle data, hozon::netaos::APAStateMachineFrame &send_data) {
        adf::BaseDataTypePtr idl_data = data.GetOne("sm2mcu");
        if (idl_data == nullptr) {
            SENSOR_LOG_WARN << "Fail to get apa statemachine data.";
            return -1;
        }

        std::shared_ptr<hozon::state::StateMachine> Sample 
            = std::static_pointer_cast<hozon::state::StateMachine>(idl_data->proto_msg);

        send_data.pilot_status.processing_status = Sample->mutable_pilot_status()->processing_status();
        send_data.pilot_status.turn_light_status = Sample->mutable_pilot_status()->turn_light_status();
        send_data.pilot_status.localization_status = Sample->mutable_pilot_status()->localization_status();
        send_data.pilot_status.camera_status = Sample->mutable_pilot_status()->camera_status();
        send_data.pilot_status.uss_status = Sample->mutable_pilot_status()->uss_status();
        send_data.pilot_status.radar_status = Sample->mutable_pilot_status()->radar_status();
        send_data.pilot_status.lidar_status = Sample->mutable_pilot_status()->lidar_status();
        send_data.pilot_status.velocity_status = Sample->mutable_pilot_status()->velocity_status();
        send_data.pilot_status.perception_status = Sample->mutable_pilot_status()->perception_status();
        send_data.pilot_status.planning_status = Sample->mutable_pilot_status()->planning_status();
        send_data.pilot_status.controlling_status = Sample->mutable_pilot_status()->controlling_status();
        send_data.hpp_command.enable_parking_slot_detection = Sample->mutable_hpp_command()->enable_parking_slot_detection();
        send_data.hpp_command.reserved1 = Sample->mutable_hpp_command()->reserved1();
        send_data.hpp_command.reserved2 = Sample->mutable_hpp_command()->reserved2();
        send_data.hpp_command.reserved3 = Sample->mutable_hpp_command()->reserved3();
        send_data.hpp_command.enable_object_detection = Sample->mutable_hpp_command()->enable_object_detection();
        send_data.hpp_command.enable_freespace_detection = Sample->mutable_hpp_command()->enable_freespace_detection();
        send_data.hpp_command.enable_uss = Sample->mutable_hpp_command()->enable_uss();
        send_data.hpp_command.enable_radar = Sample->mutable_hpp_command()->enable_radar();
        send_data.hpp_command.enable_lidar = Sample->mutable_hpp_command()->enable_lidar();
        send_data.hpp_command.system_command = Sample->mutable_hpp_command()->system_command();
        send_data.hpp_command.emergencybrake_state = Sample->mutable_hpp_command()->emergencybrake_state();
        send_data.hpp_command.system_reset = Sample->mutable_hpp_command()->system_reset();
        
        return 0;
    }
};

}
}
}