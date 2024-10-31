#ifndef SM_ALG_STRUCT_STATE_MACHINE_FRAME_INFO_H__
#define SM_ALG_STRUCT_STATE_MACHINE_FRAME_INFO_H__

#include <stdint.h>

struct AlgAutopilotStatus {
    uint8_t processing_status;
    uint8_t camera_status;
    uint8_t uss_status;
    uint8_t radar_status;
    uint8_t lidar_status;
    uint8_t velocity_status;
    uint8_t perception_status;
    uint8_t planning_status;
    uint8_t controlling_status;
    uint8_t turn_light_status;
    uint8_t localization_status;
};

struct AlgCommand {
    uint8_t enable_parking_slot_detection;
    uint8_t enable_object_detection;
    uint8_t enable_freespace_detection;
    uint8_t enable_uss;
    uint8_t enable_radar;
    uint8_t enable_lidar;
    uint8_t system_command;
    uint8_t system_reset;
    uint8_t emergencybrake_state;
    uint8_t reserved1;
    uint8_t reserved2;
    uint8_t reserved3;
};

struct AlgWorkingStatus {
    uint8_t processing_status;
    uint8_t error_code;
    uint8_t perception_warninginfo;
    uint8_t perception_ADCS4_PA_failinfo;
    uint8_t perception_ADCS4_Tex;
    uint8_t TBA_Distance;
    bool TBA;
    uint8_t TBA_text;
    uint8_t HPA;
    uint8_t HPA_PathOnParkArea;
    uint8_t HPA_PathStoreSts;
    uint8_t HPA_learnpathStSw;
    uint8_t HPA_PathlearnSts;
    uint8_t HPA_PathlearningWorkSts;
    uint8_t HPA_PointInParkslot;
    uint8_t HPA_PathwaytoCloudWorkSts;
    uint8_t HPA_GuideSts;
    uint8_t HPA_ReturnButton;
    uint8_t HPA_PathexistSts;
    uint8_t HPA_distance;
    uint8_t HPA_Pathavailable_ID;
    uint8_t HPA_CrossingNumber;
    uint8_t perception_ADCS4_HPA_failinfo;
    uint8_t HPA_LocalizationSts;
};

struct AlgPNCControlState {
    uint8_t fct_state;
    uint8_t m_iuss_state_obs;
    uint8_t need_replan_stop;
    uint8_t plan_trigger;
    uint8_t control_enable;
    uint8_t control_status;
    uint8_t pnc_run_state;
    uint8_t pnc_warninginfo;
    uint8_t pnc_ADCS4_Tex;
    uint8_t pnc_ADCS4_PA_failinfo;
    bool FAPA;
    bool RPA;
    bool TBA;
    bool LAPA_MapBuilding;
    bool LAPA_Cruising;
    bool LAPA_PickUp;
    bool ISM;
    bool AVP;
    uint8_t pnc_ADCS4_TBA_failinfo;
    uint8_t pnc_ADCS4_RPA_failinfo;
    uint8_t pnc_ADCS4_LAPA_MapBuilding_failinfo;
    uint8_t pnc_ADCS4_LAPA_Cruising_failinfo;
    uint8_t pnc_ADCS4_LAPA_PickUp_failinfo;
    uint8_t pnc_ADCS4_ISM_failinfo;
    uint8_t pnc_ADCS4_AVP_failinfo;
    uint16_t TBA_text;
    uint8_t reserved2;
    uint8_t reserved3;
};

struct AlgStateMachineFrame {
    uint64_t timestamp;
    uint8_t counter;
    AlgAutopilotStatus pilot_status;
    AlgCommand hpp_command;
    AlgWorkingStatus hpp_perception_status;
    AlgPNCControlState pnc_control_state;
    bool isValid;
};

#endif