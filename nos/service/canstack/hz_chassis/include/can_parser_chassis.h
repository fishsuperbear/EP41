/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: socket can interface monitor[chassis]
 */

#pragma once

#include <stdint.h>
#include <string.h>

#include <bitset>
#include <functional>
#include <map>
#include <mutex>

#include "can_parser.h"
#include "data_type.h"
#include "hz_common.h"

namespace hozon {
namespace netaos {
namespace canstack {
namespace chassis {

using namespace hozon::netaos::canstack;
/* EP40 */
#define ADCS_MCU_SOC_Tx_CYC_100 (0x100)      // 1 20ms
#define ADCS_MCU_SOC_Tx_CYC_112 (0x112)      // 2 20ms
#define ADCS_MCU_SOC_Tx_CYC_1A2 (0x1A2)      // 3 50ms
#define ADCS_MCU_SOC_TX_CYC_A2 (0x0A2)       // 4 10ms
#define ADCS_MCU_SOC_Tx_CYC_E3 (0x0E3)       // 5 10ms
#define ADCS_MCU_SOC_Tx_CYC_E5 (0x0E5)       // 6 20ms
#define ADCS_MCU_SOC_Tx_Event_0x127 (0x127)  // 7 Event
#define ADCS_MCU_SOC_Tx_Event_0x1E3 (0x1E3)  // 8 Event
#define ADCS_MCU_SOC_Tx_CYC_200 (0x200)      // 9 100ms
#define ADCS_MCU_SOC_Tx_CYC_201 (0x201)      // 10 50ms
#define TSR_LIMIT_OVER_SPEED_INVALID (0xFF)  // 0xFF is invalid
/* EP40*/

#define TIME_DIFF (120U)

/* Parking_NUMBER_FOR TEST */
#define PARKING_NUMBER_0x666 (0x666)

/* Parking_NUMBER_FOR TEST */

typedef struct Timestamp {
    uint32_t sec;
    uint32_t nsec;
} Timestamp_t;

/* EP40 */
typedef struct ADCS_MCU_SOC_Tx_CYC_E5_ {
    uint8_t mcu_high_beam_req_st;
    uint8_t mcu_crash_status;
    uint8_t mcu_driver_buckle;
    uint8_t mcu_front_wiper_req_st;
    uint8_t mcu_retry;
    uint8_t mcu_fixslot;
    uint8_t mcu_parking_req;
    uint8_t mcu_recover;
    uint8_t mcu_parkingout_slot_type;
    uint8_t mcu_btm1_parkingout_slot_dire;
    uint8_t mcu_optional_slot_type;
    uint8_t mcu_btm1_optional_slot_dire;
    uint8_t mcu_function_mode;
    uint8_t mcu_btm1_remote_park_req;
    uint8_t mcu_movectrl;
    uint8_t mcu_rolling_counter;
    uint16_t mcu_btm1_checksum;
    double mcu_btm1_optional_slot_angle;
    uint16_t mcu_btm1_optional_slot_coor_p0_x;
    uint16_t mcu_btm1_optional_slot_coor_p0_y;
    uint16_t mcu_btm1_optional_slot_coor_p1_x;
    uint16_t mcu_btm1_optional_slot_coor_p1_y;
    uint16_t mcu_btm1_optional_slot_coor_p2_x;
    uint16_t mcu_btm1_optional_slot_coor_p2_y;
    uint16_t mcu_btm1_optional_slot_coor_p3_x;
    uint16_t mcu_btm1_optional_slot_coor_p3_y;
    uint8_t mcu_btm1_security_sts;
    uint8_t mcu_btm1_phone_bat_sts;
    uint8_t mcu_btm1_remote_int_mod_sel;
    uint8_t mcu_btm1_select_slot_id;
    uint8_t mcu_esc_active;
    uint8_t mcu_abs_active;
    uint8_t mcu_tcs_active;
    uint8_t mcu_arpactive;
} ADCS_MCU_SOC_Tx_CYC_E5_t;  // ADCS_MCU_SOC_Tx_CYC_E5

typedef struct ADCS_MCU_SOC_Tx_CYC_1A2_ {
    uint8_t mcu_usb_st;
    uint8_t mcu_storage_st;
    uint8_t mcu_user_scene_mode;
    uint8_t mcu_front_wiper_sts;
    double mcu_ac_outside_temp;
    uint8_t mcu_ac_outside_temp_valid;
    double mcu_odometer;
    uint8_t mcu_rr_door_ajar;
    uint8_t mcu_rr_lock_st;
    uint8_t mcu_fr_door_ajar;
    uint8_t mcu_fr_lock_st;
    uint8_t mcu_rl_door_ajar;
    uint8_t mcu_rl_lock_st;
    uint8_t mcu_fl_door_ajar;
    uint8_t mcu_fl_lock_st;
    uint8_t mcu_work_status;
    uint8_t mcu_btm2_connect_sts;
    uint8_t mcu_key_position;
    double mcu_state_of_charge;
    uint8_t mcu_charge_state;
    uint8_t mcu_rls_rq_wiper_spd;
    uint16_t mcu_fl_win_pos;
    uint16_t mcu_rl_win_pos;
    uint8_t mcu_mirror_fold_st;
    uint16_t mcu_fr_win_pos;
    uint16_t mcu_rr_win_pos;
    uint8_t mcu_swsm_a_cruise_distance_add;
    uint8_t mcu_swsm_a_cruise_distance_minus;
    uint8_t mcu_swsm_a_cruise_speed_add;
    uint8_t mcu_swsm_a_cruise_speed_minus;
} ADCS_MCU_SOC_Tx_CYC_1A2_t;  // ADCS_MCU_SOC_Tx_CYC_1A2

typedef struct ADCS_MCU_SOC_Tx_Event_0x127_ {
    uint32_t mcu_xcoordinate;
    uint32_t mcu_ycoordinate;
    uint16_t mcu_xgesturespd;
    uint16_t mcu_ygesturespd;
    uint8_t mcu_touchevttyp;
} ADCS_MCU_SOC_Tx_Event_0x127_t;  // ADCS_MCU_SOC_Tx_CYC_127

typedef struct ADCS_MCU_SOC_Tx_Event_0x1E3_ {
    uint8_t mcu_tsr_limit_overspeed_set;
    uint8_t mcu_dc_wposition;
    uint8_t mcu_learnpath_st;
    uint8_t mcu_find_car_avm_req;
    uint8_t mcu_avm_powertype;
    uint16_t mcu_hpa_select_id;
    uint8_t mcu_factory_reset;
    uint8_t mcu_reset_all_setup;
    uint8_t mcu_avm_diaphaneity;
    uint8_t mcu_low_speed_bsd_sw;
    uint8_t mcu_avm_3_d_rotate;
    uint8_t mcu_avm_turnlightstart;
    uint8_t mcu_avm_steeringstart;
    uint8_t mcu_avm_magnifiedview;
    uint8_t mcu_avm_auxiliaryline;
    uint8_t mcu_avm_raderdisplay;
    uint8_t mcu_apa_function_mode;
    uint8_t mcu_trackreverse_sw;
    uint8_t mcu_avm_sw;
    uint8_t mcu_location_sw;
    uint8_t mcu_pathrefresh;
    uint8_t mcu_hpa_guide_sw;
    uint8_t mcu_pathlearning_sw;
    uint8_t mcu_pa_sw;
    uint8_t mcu_pa_recover;
    uint8_t mcu_select_slot_id;
    uint8_t mcu_parking_in_req;
    uint8_t mcu_parking_out_req;
    uint8_t mcu_avm_view_mode;
    uint8_t mcu_rpa_function_mode;
    uint8_t mcu_avm_initialization;
    uint8_t mcu_avm_body_sync;
    uint8_t mcu_avm_spe_view;
    uint8_t mcu_avm_licenseplateabbr;
    uint8_t mcu_avm_licenseplate_area;
    uint8_t mcu_avm_license_plate_no1;
    uint8_t mcu_avm_license_plate_no2;
    uint8_t mcu_avm_license_plate_no3;
    uint8_t mcu_avm_license_plate_no4;
    uint8_t mcu_avm_license_plate_no5;
    uint8_t mcu_avm_license_plate_no6;
    uint8_t mcu_avm_license_plate_no7;
    uint8_t mcu_avm_license_plate_no8;
    uint8_t mcu_avm_diaphaneity_sw;
    uint8_t mcu_mod_sound_sw;
    uint8_t mcu_mod_bsd_sw;
    uint8_t mcu_pa_frs_on_off_set;
    uint8_t mcu_pa_3_d_on_off_set;
    uint8_t mcu_pa_measure_on_off_set;
    uint8_t mcu_mod_sw;
    uint8_t mcu_hpa_sw;
    uint16_t mcu_delete_map_id;
    uint16_t mcu_upload_map_id;
    uint8_t mcu_cdcs11_parkingout_slot_dire;
    uint8_t mcu_optional_slot_type;
    uint8_t mcu_cdcs11_optional_slot_dire;
    double mcu_optional_slot_angle;
    uint32_t mcu_optional_slot_coor_p0_x;
    uint32_t mcu_optional_slot_coor_p0_y;
    uint32_t mcu_optional_slot_coor_p1_x;
    uint32_t mcu_optional_slot_coor_p1_y;
    uint32_t mcu_optional_slot_coor_p2_x;
    uint32_t mcu_optional_slot_coor_p2_y;
    uint32_t mcu_optional_slot_coor_p3_x;
    uint32_t mcu_optional_slot_coor_p3_y;
    uint8_t mcu_avm_mo_ddetection;
    uint8_t mcu_gm_on_off_set;
    uint8_t mcu_avm_transparentchassis;
    uint8_t mcu_gm_vided_in;
    uint8_t mcu_set_pathway_sw;
    uint8_t mcu_try_hpp;
    uint8_t mcu_rpa_sw;
    uint8_t mcu_set_sw;
    uint8_t mcu_hpa_pathwayto_cloud_sw;
    uint16_t mcu_hpa_on_path_id1;
    uint16_t mcu_hpa_on_path_id2;
    uint16_t mcu_hpa_on_path_id3;
    uint16_t mcu_hpa_on_path_id4;
    uint8_t mcu_hp_apreparking_sw;
    uint8_t mcu_hpa_pathwayto_cloud_work_sts;
    uint8_t mcu_parkingout_slot_type;
    uint8_t mcu_fcta_on_off_set;
    uint8_t mcu_rcta_on_off_set;
    uint8_t mcu_rcw_on_off_set;
    uint8_t mcu_dow_on_off_set;
    uint8_t mcu_lca_on_off_set;
    uint8_t mcu_tsr_on_off_set;
    uint8_t mcu_ihbc_sys_sw_state;
    uint8_t mcu_tsr_overspeed_onoff_set;
} ADCS_MCU_SOC_Tx_Event_0x1E3_t;  // ADCS_MCU_SOC_Tx_CYC_1E3

typedef struct ADCS_MCU_SOC_Tx_CYC_112_ {
    uint16_t mcu_hpa_pick_path_id;
    uint16_t mcu_hpa_on_path_id;
    uint8_t mcu_pa_stand_still;
    uint8_t mcu_epb_status;
    uint8_t mcu_act_gear_valid;
    uint8_t mcu_act_gear;
    uint8_t mcu_accel_position_valid;
    uint16_t mcu_accel_position;
    uint8_t mcu_tbox2_rolling_counter;
    uint8_t mcu_remotepick;
    uint8_t mcu_tbox2_remote_park_req;
    uint8_t mcu_remote_hpp;
    uint8_t mcu_nrp_req;
    uint8_t mcu_nnssw;
    uint8_t mcu_nn_ssuspend;
    uint8_t mcu_vehicle_spd_display_valid;
    double mcu_vehicle_spd_display;
    uint8_t mcu_avm_on_req;
    uint8_t mcu_remote_park_start_press_req_sts;
    uint8_t mcu_av_mview_req;
    uint8_t mcu_security_sts;
    uint8_t mcu_connect_sts;
    uint8_t mcu_phone_in_car;
    uint8_t mcu_remote_park_abort_req;
    uint8_t mcu_remote_park_start_press_req;
    uint8_t mcu_remote_park_out_mod_sel;
    uint8_t mcu_phone_bat_sts;
    uint8_t mcu_remote_int_mod_sel;
    uint8_t mcu_remote_sw;
    uint8_t mcu_remote_ctr_mod_sel;
    uint8_t mcu_remote_ctr_sw;
    uint8_t mcu_remote_vhpower;
    uint8_t mcu_sd_map_req;
    uint8_t mcu_pa_pause_recover_req;
    uint8_t mcu_360view_req;
    uint8_t mcu_360_spec_view;
    uint8_t mcu_gm_onreq;
    uint16_t mcu_checksum;
} ADCS_MCU_SOC_Tx_CYC_112_t;  // ADCS_MCU_SOC_Tx_CYC_112

typedef struct ADCS_MCU_SOC_TX_CYC_200 {
    Timestamp_t data_time;
    Timestamp_t gnss_time;
    uint16_t mcu_fcw_target_id;
    uint16_t mcu_fcw_state;
    uint16_t mcu_aeb_target_id;
    uint16_t mcu_aeb_state;
    uint16_t mcu_ta_pilot_mode;
    uint64_t mcu_fct2_soc_tbd_u32_05;
    uint64_t mcu_fct2_soc_tbd_u32_04;
    uint64_t mcu_fct2_soc_tbd_u32_03;
    uint64_t mcu_fct2_soc_tbd_u32_02;
    uint64_t mcu_fct2_soc_tbd_u32_01;
    uint16_t mcu_drive_mode;
    uint16_t mcu_alc_warnning_state;
    uint16_t mcu_alc_warnning_target_id;
    uint16_t mcu_acc_target_id;
    uint16_t mcu_fct_pnc_warninginfo;
    uint16_t mcu_fct_avp_run_state;
    uint8_t mcu_fct_high_beam_st;
    uint8_t mcu_fct_ads_driving_mode;
    uint8_t mcu_fct_low_beam_st;
    uint16_t mcu_fct_system_command;
    uint16_t mcu_fct_avp_sys_mode;
    uint8_t mcu_fct_horn_st;
    uint8_t mcu_fct_low_high_beam_st;
    uint8_t mcu_fct_pay_mode_confirm;
    uint8_t mcu_fct_spd_adapt_comfirm;
    uint8_t mcu_fct_lcsndconfirm;
    uint8_t mcu_fct_lcsndrequest;
    uint8_t mcu_fct_alc_mode;
    uint8_t mcu_fct_turn_light_req_st;
    uint8_t mcu_fct_hazard_lamp_st;
    uint8_t mcu_fct_drive_offinhibition_obj;
    uint16_t mcu_fct_nnp_sys_state;
    uint16_t mcu_fct_longitud_ctrl_set_distance;
    uint16_t mcu_fct_longitud_ctrl_set_speed;
    uint8_t mcu_fct_adas_drive_offinhibition;
    uint8_t mcu_fct_longitud_ctrl_drive_off;
    uint8_t mcu_fct_longitud_ctrl_dec_to_stop_req;
} ADCS_MCU_SOC_TX_CYC_200_t;  // ADCS_MCU_SOC_TX_CYC_200_t

typedef struct ADCS_MCU_SOC_TX_CYC_201 {
    Timestamp_t data_time;
    Timestamp_t gnss_time;
    uint8_t mcu_avp_available;
    uint8_t mcu_ism_available;
    uint8_t mcu_lapa_pick_up_available;
    uint8_t mcu_lapa_cruising_available;
    uint8_t mcu_lapa_map_building_available;
    uint8_t mcu_tba_available;
    uint8_t mcu_rpa_available;
    uint8_t mcu_fapa_available;
    uint8_t mcu_voice_mode;
    uint16_t mcu_reserved3;
    uint16_t mcu_reserved2;
    uint16_t mcu_tba_text;
    uint16_t mcu_pnc_adcs4_pa_failinfo;
    uint16_t mcu_pnc_adcs4_tex;
    uint16_t mcu_pnc_warninginfo;
    uint16_t mcu_pnc_run_state;
    uint16_t mcu_control_status;
    uint16_t mcu_control_enable;
    uint16_t mcu_plan_trigger;
    uint16_t mcu_need_replan_stop;
    uint16_t mcu_m_iuss_state_obs;
    uint16_t mcu_fct_state;
    uint8_t mcu_pnc_adcs4_tba_failinfo;
    uint8_t mcu_pnc_adcs4_rpa_failinfo;
    uint8_t mcu_pnc_adcs4_lapa_map_building_failinfo;
    uint8_t mcu_pnc_adcs4_lapa_cruising_failinfo;
    uint8_t mcu_pnc_adcs4_lapa_pick_up_failinfo;
    uint8_t mcu_pnc_adcs4_ism_failinfo;
    uint8_t mcu_pnc_adcs4_avp_failinfo;
} ADCS_MCU_SOC_TX_CYC_201_t;  // ADCS_MCU_SOC_TX_CYC_201_t

typedef struct ADCS_MCU_SOC_Tx_CYC_100_ {
    uint8_t mcu_front_lamp_st;
    uint8_t mcu_trunk_lock_sts;
    uint8_t mcu_turn_light_sw;
    uint8_t mcu_power_manage_mode;
    uint8_t mcu_power_mode;
    uint8_t mcu_back_door_ajar;
    uint8_t mcu_low_beam_st;
    uint8_t mcu_high_beam_st;
    uint8_t mcu_left_turn_light_st;
    uint8_t mcu_right_turn_light_st;
    uint8_t mcu_hazard_lamp_st;
    uint8_t mcu_rear_fog_lamp_st;
    uint8_t mcu_front_fog_lamp_st;
    uint8_t mcu_hood_ajar_sts;
    uint8_t mcu_calendar_day;
    uint8_t mcu_calendar_month;
    uint16_t mcu_calendar_year;
    uint8_t mcu_hour_of_day;
    uint8_t mcu_minute_of_hour;
    uint8_t mcu_secs_of_minute;
    uint8_t mcu_time_dsp_fmt;
} ADCS_MCU_SOC_Tx_CYC_100_t;

typedef struct ADCS_MCU_SOC_Tx_CYC_A2_ {
    uint8_t mcu_rkecmd;
    double mcu_steeringangle;
    uint8_t mcu_steeringanglevalid;
    double mcu_steeranglespd;
    uint8_t mcu_steeranglespdvalid;
} ADCS_MCU_SOC_Tx_CYC_A2_t;

typedef struct ADCS_MCU_SOC_Tx_CYC_E3_ {
    double mcu_rcu1_brk_ped_val;
    uint8_t mcu_rr_whl_pul_cnt_valid;
    double mcu_long_acc_sensor_value;
    double mcu_lat_acc_sensor_value;
    double mcu_yaw_rate;
    uint8_t mcu_vehicle_spd_valid;
    double mcu_vehicle_spd;
    uint8_t mcu_fr_whl_pul_cnt_valid;
    double mcu_fr_whl_pul_cnt;
    uint8_t mcu_fl_whl_pul_cnt_valid;
    double mcu_fl_whl_pul_cnt;
    uint8_t mcu_brake_pedal_applied_v;
    uint8_t mcu_brake_pedal_applied;
    double mcu_fl_whl_velocity;
    uint8_t mcu_fl_whl_velocity_valid;
    double mcu_fr_whl_velocity;
    uint8_t mcu_fr_whl_velocity_valid;
    uint8_t mcu_fr_whl_dir;
    uint8_t mcu_fl_whl_dir;
    uint8_t mcu_rl_whl_dir;
    uint8_t mcu_rr_whl_dir;
    double mcu_rl_whl_velocity;
    uint8_t mcu_rl_whl_velocity_valid;
    double mcu_rr_whl_velocity;
    uint8_t mcu_rr_whl_velocity_valid;
    double mcu_rl_whl_pul_cnt;
    double mcu_rr_whl_pul_cnt;
    uint8_t mcu_rl_whl_pul_cnt_valid;
} ADCS_MCU_SOC_Tx_CYC_E3_t;

/* EP40 */

typedef struct ChassisInfo {
    Timestamp_t data_time;
    Timestamp_t gnss_time;

    /* EP40 */
    ADCS_MCU_SOC_Tx_CYC_E5_t adcs_e5_info;       /* ADCS_MCU_SOC_Tx_CYC_E5 */
    ADCS_MCU_SOC_Tx_CYC_1A2_t adcs_1a2_info;     /* ADCS_MCU_SOC_Tx_CYC_1A2 */
    ADCS_MCU_SOC_Tx_Event_0x127_t adcs_127_info; /* ADCS_MCU_SOC_Tx_CYC_127 */
    ADCS_MCU_SOC_Tx_Event_0x1E3_t adcs_1e3_info; /* ADCS_MCU_SOC_Tx_CYC_1e3 */
    ADCS_MCU_SOC_Tx_CYC_112_t adcs_112_info;     /* ADCS_MCU_SOC_Tx_CYC_112 */
    ADCS_MCU_SOC_Tx_CYC_100_t adcs_110_info;     /* ADCS_MCU_SOC_Tx_CYC_110 */
    ADCS_MCU_SOC_Tx_CYC_A2_t adcs_a2_info;       /* ADCS_MCU_SOC_Tx_CYC_A2*/
    ADCS_MCU_SOC_Tx_CYC_E3_t adcs_e3_info;       /* ADCS_MCU_SOC_Tx_CYC_E3*/

    /* EP40 */

} ChassisInfo_t;

class EventFrameProcess {
   public:
    void Init(uint64_t id) {
        this->frame_counter_ = 0;
        this->frame_flag_ = false;
        this->frame_id_ = id;
    }

    void SetFlag() { this->frame_flag_ = true; }

    void CounterIsReach(ChassisInfo_t& chassis_info) {
        //判断此时事件帧报文是否来过
        if (true == this->frame_flag_) {
            /* Use actual value */
            this->frame_counter_ = 0;
        } else {
            this->frame_counter_ += 10;
            //判断事件帧报文距离上一次来过是否超过180ms
            if (this->frame_counter_ > TIME_DIFF) {
                switch (this->frame_id_) {
                    case ADCS_MCU_SOC_Tx_Event_0x127:
                        chassis_info.adcs_127_info.mcu_xcoordinate = 65534;
                        chassis_info.adcs_127_info.mcu_ycoordinate = 65534;
                        chassis_info.adcs_127_info.mcu_xgesturespd = 254;
                        chassis_info.adcs_127_info.mcu_ygesturespd = 254;
                        chassis_info.adcs_127_info.mcu_touchevttyp = 0;
                        break;
                    case ADCS_MCU_SOC_Tx_Event_0x1E3:
                        // chassis_info.adcs_1e3_info.mcu_tsr_limit_overspeed_set = 0;
                        chassis_info.adcs_1e3_info.mcu_dc_wposition = 0;
                        chassis_info.adcs_1e3_info.mcu_learnpath_st = 0;
                        chassis_info.adcs_1e3_info.mcu_find_car_avm_req = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_powertype = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_select_id = 0;
                        chassis_info.adcs_1e3_info.mcu_factory_reset = 0;
                        chassis_info.adcs_1e3_info.mcu_reset_all_setup = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_diaphaneity = 0;
                        chassis_info.adcs_1e3_info.mcu_low_speed_bsd_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_3_d_rotate = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_turnlightstart = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_steeringstart = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_magnifiedview = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_auxiliaryline = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_raderdisplay = 0;
                        chassis_info.adcs_1e3_info.mcu_apa_function_mode = 0;
                        chassis_info.adcs_1e3_info.mcu_trackreverse_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_location_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_pathrefresh = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_guide_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_pathlearning_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_pa_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_pa_recover = 0;
                        chassis_info.adcs_1e3_info.mcu_select_slot_id = 0;
                        chassis_info.adcs_1e3_info.mcu_parking_in_req = 0;
                        chassis_info.adcs_1e3_info.mcu_parking_out_req = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_view_mode = 0;
                        chassis_info.adcs_1e3_info.mcu_rpa_function_mode = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_initialization = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_body_sync = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_spe_view = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_licenseplateabbr = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_licenseplate_area = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no1 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no2 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no3 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no4 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no5 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no6 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no7 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_license_plate_no8 = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_diaphaneity_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_mod_sound_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_mod_bsd_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_pa_frs_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_pa_3_d_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_pa_measure_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_mod_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_delete_map_id = 0;
                        chassis_info.adcs_1e3_info.mcu_upload_map_id = 0;
                        chassis_info.adcs_1e3_info.mcu_cdcs11_parkingout_slot_dire = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_type = 0;
                        chassis_info.adcs_1e3_info.mcu_cdcs11_optional_slot_dire = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_angle = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p0_x = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p0_y = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p1_x = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p1_y = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p2_x = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p2_y = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p3_x = 0;
                        chassis_info.adcs_1e3_info.mcu_optional_slot_coor_p3_y = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_mo_ddetection = 0;
                        chassis_info.adcs_1e3_info.mcu_gm_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_avm_transparentchassis = 0;
                        chassis_info.adcs_1e3_info.mcu_gm_vided_in = 0;
                        chassis_info.adcs_1e3_info.mcu_set_pathway_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_try_hpp = 0;
                        chassis_info.adcs_1e3_info.mcu_rpa_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_set_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_pathwayto_cloud_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_on_path_id1 = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_on_path_id2 = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_on_path_id3 = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_on_path_id4 = 0;
                        chassis_info.adcs_1e3_info.mcu_hp_apreparking_sw = 0;
                        chassis_info.adcs_1e3_info.mcu_hpa_pathwayto_cloud_work_sts = 0;
                        chassis_info.adcs_1e3_info.mcu_parkingout_slot_type = 0;
                        chassis_info.adcs_1e3_info.mcu_fcta_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_rcta_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_rcw_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_dow_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_lca_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_tsr_on_off_set = 0;
                        chassis_info.adcs_1e3_info.mcu_ihbc_sys_sw_state = 0;
                        chassis_info.adcs_1e3_info.mcu_tsr_overspeed_onoff_set = 0;
                        break;
                    default:
                        break;
                }
            } else {
                /* Use last time value */
            }
        }
    }

    void Clear() { this->frame_flag_ = false; }

   private:
    bool frame_flag_;
    uint8_t frame_counter_;
    uint64_t frame_id_;
};

class CanParserChassis : public CanParser {
   public:
    static CanParserChassis* Instance();
    virtual ~CanParserChassis();

    virtual void Init();
    virtual void ParseCan(can_frame& receiveFrame);
    virtual void ParseCanfd(canfd_frame& receiveFrame);
    virtual void GetCanFilters(std::vector<can_filter>& filters);

   private:
    CanParserChassis();
    void ParseChassisInfo(unsigned int can_id, unsigned char* data);

    /* E3 Upstream*/
    bool Parse100(unsigned char* msg);  // 1
    bool Parse112(unsigned char* msg);  // 2
    bool Parse1A2(unsigned char* msg);  // 3
    bool ParseA2(unsigned char* msg);   // 4
    bool ParseE3(unsigned char* msg);   // 5
    bool ParseE5(unsigned char* msg);   // 6
    bool Parse127(unsigned char* msg);  // 7
    bool Parse1E3(unsigned char* msg);  // 8
    bool Parse200(unsigned char* msg);  // 9
    bool Parse201(unsigned char* msg);  // 10
    /* E3 Upstream*/

    void SetChassisInfoToBuffer();
    void SetMcu2EgoInfoToBuffer();
    void SetMcu2StateMachineInfoToBuffer();

    int CheckTimeLog();

    void LogData(unsigned int can_id, unsigned char* data);
    void CanId2Flag(unsigned int can_id);

    using CanFrameParseFunc = std::function<bool(unsigned char*)>;

    struct CanFrameParseInfo {
        int32_t data_type;
        uint32_t buf_index;
        CanFrameParseFunc Parse;
    };

    static CanParserChassis* instance_ptr_;
    static std::mutex mtx_;
    ChassisInfo_t chassis_info_;
    ADCS_MCU_SOC_TX_CYC_200_t mcu_to_ego_200_;
    ADCS_MCU_SOC_TX_CYC_201_t mcu_to_statemachine_201_;
    bool e3_chassis_info_flag_;
    uint64_t time_diff_;
    EventFrameProcess _127_frame_;
    EventFrameProcess _1E3_frame_;

    std::map<unsigned int, CanFrameParseInfo> can_parse_info_map_;
};

}  // namespace chassis
}  // namespace canstack
}  // namespace netaos
}  // namespace hozon
