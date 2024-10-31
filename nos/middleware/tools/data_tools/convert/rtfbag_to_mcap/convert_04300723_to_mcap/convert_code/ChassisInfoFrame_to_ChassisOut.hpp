#pragma once
#include "hozon/chassis/impl_type_chassisinfoframe.h"  //mdc 数据变量
#include "proto/soc/chassis.pb.h"                      //proto 数据变量

hozon::soc::Chassis ChassisInfoFrameToChassisOut(hozon::chassis::ChassisInfoFrame mdc_data) {
    hozon::soc::Chassis proto_data;

    // Chassis
    proto_data.set_parking_brake(mdc_data.esc_driving_info.ESC_BrakePedalSwitchStatus);
    proto_data.set_error_code(static_cast<hozon::soc::Chassis::ErrorCode>(mdc_data.error_code));
    proto_data.set_gear_location(static_cast<hozon::soc::Chassis::GearPosition>(mdc_data.vcu_info.VCU_ActGearPosition));

    // hozon.common.Header
    proto_data.mutable_header()->set_seq(mdc_data.header.seq);
    proto_data.mutable_header()->set_frame_id(mdc_data.header.frameId);
    proto_data.mutable_header()->set_publish_stamp(static_cast<double>(mdc_data.header.stamp.sec) + static_cast<double>(mdc_data.header.stamp.nsec) / 1e9);

    // hozon.common.VehicleSignal
    proto_data.mutable_signal()->set_high_beam(static_cast<bool>(mdc_data.body_state_info.BCM_HighBeamSt));
    proto_data.mutable_signal()->set_low_beam(static_cast<bool>(mdc_data.body_state_info.BCM_LowBeamSt));
    proto_data.mutable_signal()->set_emergency_light(static_cast<bool>(mdc_data.body_state_info.HazardLampSt));
    proto_data.mutable_signal()->set_turn_switch(static_cast<hozon::common::TurnLightSwitchStatus>(mdc_data.body_state_info.BCM_TurnLightSW));
    proto_data.mutable_signal()->set_cs1_high_beam_req_st(static_cast<uint32_t>(mdc_data.body_state_info.CS1_HighBeamReqSt));
    proto_data.mutable_signal()->set_bcm_front_fog_lamp_st(mdc_data.body_state_info.BCM_FrontFogLampSt);
    proto_data.mutable_signal()->set_bcm_rear_fog_lamp_st(mdc_data.body_state_info.BCM_RearFogLampSt);
    proto_data.mutable_signal()->set_bcm_front_lamp_st(static_cast<uint32_t>(mdc_data.body_state_info.BCM_FrontLampSt));

    // ChassisGPS
    proto_data.mutable_chassis_gps()->set_year(static_cast<int32_t>(mdc_data.chassis_time_info.CalendarYear));
    proto_data.mutable_chassis_gps()->set_month(static_cast<int32_t>(mdc_data.chassis_time_info.CalendarMonth));
    proto_data.mutable_chassis_gps()->set_day(static_cast<int32_t>(mdc_data.chassis_time_info.CalendarDay));
    proto_data.mutable_chassis_gps()->set_hours(static_cast<int32_t>(mdc_data.chassis_time_info.HourOfDay));
    proto_data.mutable_chassis_gps()->set_minutes(static_cast<int32_t>(mdc_data.chassis_time_info.MinuteOfHour));

    // WheelSpeed
    proto_data.mutable_wheel_speed()->set_is_wheel_spd_rr_valid(mdc_data.wheel_info.ESC_RRWheelSpeedValid);
    proto_data.mutable_wheel_speed()->set_wheel_direction_rr(static_cast<hozon::soc::WheelSpeed::WheelSpeedType>(mdc_data.wheel_info.ESC_RRWheelDirection));
    proto_data.mutable_wheel_speed()->set_wheel_spd_rr(mdc_data.wheel_info.ESC_RRWheelSpeed);
    proto_data.mutable_wheel_speed()->set_is_wheel_spd_rl_valid(mdc_data.wheel_info.ESC_RLWheelSpeedValid);
    proto_data.mutable_wheel_speed()->set_wheel_direction_rl(static_cast<hozon::soc::WheelSpeed::WheelSpeedType>(mdc_data.wheel_info.ESC_RLWheelDirection));
    proto_data.mutable_wheel_speed()->set_wheel_spd_rl(mdc_data.wheel_info.ESC_RLWheelSpeed);
    proto_data.mutable_wheel_speed()->set_is_wheel_spd_fr_valid(mdc_data.wheel_info.ESC_FRWheelSpeedValid);
    proto_data.mutable_wheel_speed()->set_wheel_direction_fr(static_cast<hozon::soc::WheelSpeed::WheelSpeedType>(mdc_data.wheel_info.ESC_FRWheelDirection));
    proto_data.mutable_wheel_speed()->set_wheel_spd_fr(mdc_data.wheel_info.ESC_FRWheelSpeed);
    proto_data.mutable_wheel_speed()->set_is_wheel_spd_fl_valid(mdc_data.wheel_info.ESC_FLWheelSpeedValid);
    proto_data.mutable_wheel_speed()->set_wheel_direction_fl(static_cast<hozon::soc::WheelSpeed::WheelSpeedType>(mdc_data.wheel_info.ESC_FLWheelDirection));
    proto_data.mutable_wheel_speed()->set_wheel_spd_fl(mdc_data.wheel_info.ESC_FLWheelSpeed);

    proto_data.set_yaw_rate(mdc_data.esc_driving_info.ESC_YawRate);
    proto_data.set_steering_angle(mdc_data.steering_info.SteeringAngle);

    // WheelInfo
    proto_data.mutable_wheel_counter()->set_is_wheel_cnt_rr_valid(mdc_data.wheel_info.ESC_RR_WhlPulCntValid);
    proto_data.mutable_wheel_counter()->set_wheel_counter_rr(mdc_data.wheel_info.ESC_RR_WhlPulCnt);
    proto_data.mutable_wheel_counter()->set_is_wheel_cnt_rl_valid(mdc_data.wheel_info.ESC_RL_WhlPulCntValid);
    proto_data.mutable_wheel_counter()->set_wheel_counter_rl(mdc_data.wheel_info.ESC_RL_WhlPulCnt);
    proto_data.mutable_wheel_counter()->set_is_wheel_cnt_fr_valid(mdc_data.wheel_info.ESC_FR_WhlPulCntValid);
    proto_data.mutable_wheel_counter()->set_wheel_counter_fr(mdc_data.wheel_info.ESC_FR_WhlPulCnt);
    proto_data.mutable_wheel_counter()->set_is_wheel_cnt_fl_valid(mdc_data.wheel_info.ESC_FL_WhlPulCntValid);
    proto_data.mutable_wheel_counter()->set_wheel_counter_fl(mdc_data.wheel_info.ESC_FL_WhlPulCnt);

    proto_data.set_speed_display(mdc_data.fault_did_info.ICU1_VehicleSpdDisplay);

    // SwitchInfo
    proto_data.mutable_switch_info()->set_cruise_speed_add(mdc_data.swswitch_info.SWSM_A_CruiseSpeed_Add);
    proto_data.mutable_switch_info()->set_cruise_speed_minus(mdc_data.swswitch_info.SWSM_A_CruiseSpeed_Minus);
    proto_data.mutable_switch_info()->set_cruise_distance_add(mdc_data.swswitch_info.SWSM_A_CruiseDistance_Add);
    proto_data.mutable_switch_info()->set_cruise_distance_minus(mdc_data.swswitch_info.SWSM_A_CruiseDistance_Minus);

    // ResetSwitch
    proto_data.mutable_reset_switch()->set_factory_reset(static_cast<hozon::soc::ResetSwitch::Status>(mdc_data.center_console_info.FactoryReset));
    proto_data.mutable_reset_switch()->set_reset_all_setup(static_cast<hozon::soc::ResetSwitch::Status>(mdc_data.center_console_info.ResetAllSetup));

    // WarningSwitchMemory: AlgMcuEgoMemMsg not in ChassisInfoFrame ???

    // WarningSwitch: AlgEgoWarningInfo not in ChassisInfoFrame
    proto_data.mutable_warning_switch_from_cdcs()->set_voice_mode(static_cast<int32_t>(mdc_data.warnning_hmi_info.ADCS8_VoiceMode));
    proto_data.mutable_warning_switch_from_cdcs()->set_rcta_on_off_set(static_cast<hozon::soc::WarningSwitch::Status>(mdc_data.warnning_hmi_info.RCTA_OnOffSet));
    proto_data.mutable_warning_switch_from_cdcs()->set_fcta_on_off_set(static_cast<hozon::soc::WarningSwitch::Status>(mdc_data.warnning_hmi_info.FCTA_OnOffSet));
    proto_data.mutable_warning_switch_from_cdcs()->set_dow_on_off_set(static_cast<hozon::soc::WarningSwitch::Status>(mdc_data.warnning_hmi_info.DOW_OnOffSet));
    proto_data.mutable_warning_switch_from_cdcs()->set_rcw_on_off_set(static_cast<hozon::soc::WarningSwitch::Status>(mdc_data.warnning_hmi_info.RCW_OnOffSet));
    proto_data.mutable_warning_switch_from_cdcs()->set_lca_on_off_set(static_cast<hozon::soc::WarningSwitch::Status>(mdc_data.warnning_hmi_info.LCA_OnOffSet));

    // IgState
    proto_data.mutable_ig_state()->set_ig_off(mdc_data.ig_status_info.IG_OFF);
    proto_data.mutable_ig_state()->set_acc(mdc_data.ig_status_info.ACC);
    proto_data.mutable_ig_state()->set_ig_on(mdc_data.ig_status_info.IG_ON);
    proto_data.mutable_ig_state()->set_start(mdc_data.ig_status_info.Start);
    proto_data.mutable_ig_state()->set_remote_ig_on(mdc_data.ig_status_info.Remote_IG_ON);

    // BackDoorStatus
    proto_data.mutable_back_door_status()->set_status(static_cast<hozon::soc::BackDoorStatus::Status>(mdc_data.body_state_info.BCM_TGOpn));

    // HoodAjarStatus
    proto_data.mutable_hood_ajar_status()->set_status(static_cast<hozon::soc::HoodAjarStatus::Status>(mdc_data.body_state_info.BCM_HodOpen));

    // DriverBuckleStatus
    proto_data.mutable_driver_buckle_status()->set_status(static_cast<hozon::soc::DriverBuckleStatus::Status>(mdc_data.body_state_info.BCM_DrvSeatbeltBucklesta));

    // FrontWiperStatus
    proto_data.mutable_front_wiper_status()->set_status(static_cast<hozon::soc::FrontWiperStatus::Status>(mdc_data.body_state_info.BCM_FrontWiperSt));

    proto_data.set_odometer(mdc_data.fault_did_info.ICU2_Odometer);
    proto_data.set_crash_status(mdc_data.fault_did_info.Ignition_status);

    proto_data.set_vcu_act_gear_position_valid(mdc_data.vcu_info.VCU_ActGearPosition_Valid);
    proto_data.set_vcu_real_throttle_pos_valid(mdc_data.vcu_info.VCU_Real_ThrottlePos_Valid);
    proto_data.set_steering_angle_valid(mdc_data.steering_info.SteeringAngleValid);
    proto_data.set_steering_angle_speed_valid(mdc_data.steering_info.SteeringAngleSpeedValid);

    // CenterConsoleInfo
    proto_data.mutable_center_console_info()->set_tsr_on_off_set(static_cast<uint32_t>(mdc_data.center_console_info.TSR_OnOffSet));
    proto_data.mutable_center_console_info()->set_tsr_overspeed_onoff_set(static_cast<uint32_t>(mdc_data.center_console_info.TSR_OverspeedOnoffSet));
    proto_data.mutable_center_console_info()->set_tsr_limit_overspeed_set(static_cast<uint32_t>(mdc_data.center_console_info.TSR_LimitOverspeedSet));
    proto_data.mutable_center_console_info()->set_ihbc_sys_sw_state(static_cast<uint32_t>(mdc_data.center_console_info.IHBC_SysSwState));

    // ParkInfo
    proto_data.mutable_park_info()->set_tcs_active(mdc_data.park_info.TCSActive);
    proto_data.mutable_park_info()->set_abs_active(mdc_data.park_info.ABSActive);
    proto_data.mutable_park_info()->set_arp_active(mdc_data.park_info.ARPActive);
    proto_data.mutable_park_info()->set_esc_active(mdc_data.park_info.ESCActive);
    proto_data.mutable_park_info()->set_epb_status(static_cast<uint32_t>(mdc_data.park_info.EPBStatus));

    // EscDrivingInfo
    proto_data.mutable_esc_driving_info()->set_esc_vehicle_speed(mdc_data.esc_driving_info.ESC_VehicleSpeed);
    proto_data.mutable_esc_driving_info()->set_esc_vehicle_speed_valid(mdc_data.esc_driving_info.ESC_VehicleSpeedValid);
    proto_data.mutable_esc_driving_info()->set_esc_brake_pedal_switch_status(mdc_data.esc_driving_info.ESC_BrakePedalSwitchStatus);
    proto_data.mutable_esc_driving_info()->set_esc_brake_pedal_switch_valid(mdc_data.esc_driving_info.ESC_BrakePedalSwitchValid);
    proto_data.mutable_esc_driving_info()->set_brk_ped_val(mdc_data.esc_driving_info.BrkPedVal);
    proto_data.mutable_esc_driving_info()->set_esc_apa_stand_still(mdc_data.esc_driving_info.ESC_ApaStandStill);
    proto_data.mutable_esc_driving_info()->set_vehicle_spd_display_valid(mdc_data.esc_driving_info.VehicleSpdDisplayValid);
    proto_data.mutable_esc_driving_info()->set_esc_long_acc_value_valid(mdc_data.esc_driving_info.ESC_LongAccValue_Valid);
    proto_data.mutable_esc_driving_info()->set_esc_lat_acc_value_valid(mdc_data.esc_driving_info.ESC_LatAccValue_Valid);
    proto_data.mutable_esc_driving_info()->set_esc_yaw_rate_valid(mdc_data.esc_driving_info.ESC_YawRate_Valid);

    return proto_data;
}