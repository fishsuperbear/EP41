#include <unistd.h>
#include "proto/statemachine/state_machine.pb.h"
#include "proto/soc/chassis.pb.h"
#include "cm/include/proto_cm_writer.h"
#include "log/include/default_logger.h"



void ChassisData(hozon::soc::Chassis &proto_data)
{
    proto_data.set_gear_location(static_cast<hozon::soc::Chassis_GearPosition>(4));

    // FAPA泊入
    proto_data.mutable_avm_pds_info()->set_cdcs11_pasw(1);
    sleep(1);
    proto_data.mutable_avm_pds_info()->set_cdcs11_apa_functionmode(1); // FAPA泊入
    sleep(1);
    proto_data.mutable_avm_pds_info()->set_cdcs11_parkinginreq(1);     // 点击了开始泊入按键
    sleep(1);

    // FAPA泊出
    proto_data.mutable_avm_pds_info()->set_cdcs11_apa_functionmode(2); // FAPA泊出
    sleep(1);
    proto_data.mutable_avm_pds_info()->set_cdcs11_parkingoutreq(1);     // 点击了开始泊出按键
}

void PerSmData(hozon::state::StateMachine &proto_data)
{
    // AutopilotStatus out
    proto_data.mutable_pilot_status()->set_processing_status(static_cast<uint32_t>(1));
    proto_data.mutable_pilot_status()->set_camera_status(static_cast<uint32_t>(2));
    proto_data.mutable_pilot_status()->set_uss_status(static_cast<uint32_t>(3));
    proto_data.mutable_pilot_status()->set_radar_status(static_cast<uint32_t>(4));
    proto_data.mutable_pilot_status()->set_lidar_status(static_cast<uint32_t>(5));
    proto_data.mutable_pilot_status()->set_velocity_status(static_cast<uint32_t>(6));
    proto_data.mutable_pilot_status()->set_perception_status(static_cast<uint32_t>(7));
    proto_data.mutable_pilot_status()->set_planning_status(static_cast<uint32_t>(8));
    proto_data.mutable_pilot_status()->set_controlling_status(static_cast<uint32_t>(9));
    proto_data.mutable_pilot_status()->set_turn_light_status(static_cast<uint32_t>(10));
    proto_data.mutable_pilot_status()->set_localization_status(static_cast<uint32_t>(11));

    // Command out
    proto_data.mutable_hpp_command()->set_enable_parking_slot_detection(static_cast<uint32_t>(11));
    proto_data.mutable_hpp_command()->set_enable_object_detection(static_cast<uint32_t>(12));
    proto_data.mutable_hpp_command()->set_enable_freespace_detection(static_cast<uint32_t>(13));
    proto_data.mutable_hpp_command()->set_enable_uss(static_cast<uint32_t>(14));
    proto_data.mutable_hpp_command()->set_enable_radar(static_cast<uint32_t>(15));
    proto_data.mutable_hpp_command()->set_enable_lidar(static_cast<uint32_t>(16));
    proto_data.mutable_hpp_command()->set_system_command(static_cast<uint32_t>(17));
    proto_data.mutable_hpp_command()->set_emergencybrake_state(static_cast<uint32_t>(18));
    proto_data.mutable_hpp_command()->set_system_reset(static_cast<uint32_t>(19));
    proto_data.mutable_hpp_command()->set_reserved1(static_cast<uint32_t>(20));
    proto_data.mutable_hpp_command()->set_reserved2(static_cast<uint32_t>(21));
    proto_data.mutable_hpp_command()->set_reserved3(static_cast<uint32_t>(22));

    // WorkingStatus in
    proto_data.mutable_hpp_perception_status()->set_processing_status(2);
    proto_data.mutable_hpp_perception_status()->set_error_code(21);
    proto_data.mutable_hpp_perception_status()->set_perception_warninginfo(22);
    proto_data.mutable_hpp_perception_status()->set_perception_adcs4__tex(23);
    proto_data.mutable_hpp_perception_status()->set_perception_adcs4_pa_failinfo(24);
    proto_data.mutable_hpp_perception_status()->set_tba__distance(25);
    proto_data.mutable_hpp_perception_status()->set_tba(true);
    proto_data.mutable_hpp_perception_status()->set_tba_text(26);
}

void PncSmData(hozon::state::StateMachine &proto_data)
{
    // AutopilotStatus out
    proto_data.mutable_pilot_status()->set_processing_status(static_cast<uint32_t>(1));
    proto_data.mutable_pilot_status()->set_camera_status(static_cast<uint32_t>(2));
    proto_data.mutable_pilot_status()->set_uss_status(static_cast<uint32_t>(3));
    proto_data.mutable_pilot_status()->set_radar_status(static_cast<uint32_t>(4));
    proto_data.mutable_pilot_status()->set_lidar_status(static_cast<uint32_t>(5));
    proto_data.mutable_pilot_status()->set_velocity_status(static_cast<uint32_t>(6));
    proto_data.mutable_pilot_status()->set_perception_status(static_cast<uint32_t>(7));
    proto_data.mutable_pilot_status()->set_planning_status(static_cast<uint32_t>(8));
    proto_data.mutable_pilot_status()->set_controlling_status(static_cast<uint32_t>(9));
    proto_data.mutable_pilot_status()->set_turn_light_status(static_cast<uint32_t>(10));
    proto_data.mutable_pilot_status()->set_localization_status(static_cast<uint32_t>(11));

    // Command out
    proto_data.mutable_hpp_command()->set_enable_parking_slot_detection(static_cast<uint32_t>(11));
    proto_data.mutable_hpp_command()->set_enable_object_detection(static_cast<uint32_t>(12));
    proto_data.mutable_hpp_command()->set_enable_freespace_detection(static_cast<uint32_t>(13));
    proto_data.mutable_hpp_command()->set_enable_uss(static_cast<uint32_t>(14));
    proto_data.mutable_hpp_command()->set_enable_radar(static_cast<uint32_t>(15));
    proto_data.mutable_hpp_command()->set_enable_lidar(static_cast<uint32_t>(16));
    proto_data.mutable_hpp_command()->set_system_command(static_cast<uint32_t>(17));
    proto_data.mutable_hpp_command()->set_emergencybrake_state(static_cast<uint32_t>(18));
    proto_data.mutable_hpp_command()->set_system_reset(static_cast<uint32_t>(19));
    proto_data.mutable_hpp_command()->set_reserved1(static_cast<uint32_t>(20));
    proto_data.mutable_hpp_command()->set_reserved2(static_cast<uint32_t>(21));
    proto_data.mutable_hpp_command()->set_reserved3(static_cast<uint32_t>(22));

    // PNCControlState in
    proto_data.mutable_pnc_control_state()->set_fct_state(static_cast<hozon::state::PNCControlState_FctState>(4));
    proto_data.mutable_pnc_control_state()->set_fapa(true);
    proto_data.mutable_pnc_control_state()->set_rpa(true);
    proto_data.mutable_pnc_control_state()->set_tba(true);
    proto_data.mutable_pnc_control_state()->set_lapa_map_building(true);
    proto_data.mutable_pnc_control_state()->set_lapa_cruising(true);
    proto_data.mutable_pnc_control_state()->set_lapa_pick_up(true);
    proto_data.mutable_pnc_control_state()->set_ism(true);
    proto_data.mutable_pnc_control_state()->set_avp(true);
}

int main(int argc, char* argv[]) {
    hozon::netaos::cm::ProtoCMWriter<hozon::soc::Chassis> chassis_writer;
    hozon::netaos::cm::ProtoCMWriter<hozon::state::StateMachine> per_sm_writer;
    hozon::netaos::cm::ProtoCMWriter<hozon::state::StateMachine> pnc_sm_writer;
    DefaultLogger::GetInstance().InitLogger();

    int32_t ret1 = chassis_writer.Init(0, "/soc/chassis");
    int32_t ret2 = per_sm_writer.Init(0, "/perception/parking/state_machine");
    int32_t ret3 = pnc_sm_writer.Init(0, "/soc/statemachine");
    if (ret1 < 0) {
        DF_LOG_ERROR << "Fail to init chasssis writer " << ret1;
        return -1;
    }
    if (ret2 < 0) {
        DF_LOG_ERROR << "Fail to init statemachine per writer " << ret2;
        return -1;
    }
    if (ret3 < 0) {
        DF_LOG_ERROR << "Fail to init statemachine pnc writer " << ret3;
        return -1;
    }

    hozon::soc::Chassis chassis{};
    hozon::state::StateMachine per_sm{};
    hozon::state::StateMachine pnc_sm{};
    ChassisData(chassis);
    PerSmData(per_sm);
    PncSmData(pnc_sm);

    for (int i = 0; i < 150; ++i) {
        DF_LOG_INFO << "Write data seq=" << i;
        // ret1 = chassis_writer.Write(chassis);
        if (ret1 < 0) {
            DF_LOG_ERROR << "Fail to write chassis " << ret1;
        }
        ret2 = per_sm_writer.Write(per_sm);
        if (ret2 < 0) {
            DF_LOG_ERROR << "Fail to write per " << ret2;
        }
        ret3 = pnc_sm_writer.Write(pnc_sm);
        if (ret3 < 0) {
            DF_LOG_ERROR << "Fail to write pnc " << ret3;
        }

        sleep(1);
    }

    chassis_writer.Deinit();
    per_sm_writer.Deinit();
    pnc_sm_writer.Deinit();
    DF_LOG_INFO << "Deinit end." << ret1;
    DF_LOG_INFO << "Deinit end." << ret2;
    DF_LOG_INFO << "Deinit end." << ret3;
}