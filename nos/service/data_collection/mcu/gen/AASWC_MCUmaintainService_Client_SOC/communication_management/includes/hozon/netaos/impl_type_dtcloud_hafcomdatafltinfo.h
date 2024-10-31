/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file impl_type_dtcloud_hafcomdatafltinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFCOMDATAFLTINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFCOMDATAFLTINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_HafcomDataFltInfo {
    std::uint8_t MCU_Fr01_0AA_Data_Error;
    std::uint8_t FMCU_Fr01_0AB_Data_Error;
    std::uint8_t EPS_Fr01_0B1_Data_Error;
    std::uint8_t EPS_Fr02_0B2_Data_Error;
    std::uint8_t IDB_Fr02_0C0_Data_Error;
    std::uint8_t ACU_Fr02_0C4_Data_Error;
    std::uint8_t IDB_Fr03_0C5_Data_Error;
    std::uint8_t IDB_Fr04_0C7_Data_Error;
    std::uint8_t IDB_Fr10_0C9_Data_Error;
    std::uint8_t VCU_Fr05_0E3_Data_Error;
    std::uint8_t IDB_Fr01_0E5_Data_Error;
    std::uint8_t RCU_Fr01_0E6_Data_Error;
    std::uint8_t BTM_Fr01_0F3_Data_Error;
    std::uint8_t VCU_Fr06_0FB_Data_Error;
    std::uint8_t ACU_Fr01_102_Data_Error;
    std::uint8_t VCU_Fr0E_108_Data_Error;
    std::uint8_t BDCS_Fr01_110_Data1_Error;
    std::uint8_t BDCS_Fr03_112_Data_Error;
    std::uint8_t BDCS_Fr16_114_Data_Error;
    std::uint8_t IDB_Fr09_121_Data_Error;
    std::uint8_t IDB_Fr05_129_Data_Error;
    std::uint8_t IDB_Fr06_12B_Data_Error;
    std::uint8_t IDB_Fr07_12D_Data_Error;
    std::uint8_t ICU_Fr01_131_Data_Error;
    std::uint8_t RCU_Fr02_137_Data_Error;
    std::uint8_t RCU_Fr03_138_Data_Error;
    std::uint8_t RCU_Fr04_139_Data_Error;
    std::uint8_t RCU_Fr06_13A_Data_Error;
    std::uint8_t RCU_Fr07_13B_Data_Error;
    std::uint8_t RCU_Fr08_13C_Data_Error;
    std::uint8_t RCU_Fr05_13D_Data_Error;
    std::uint8_t RCU_Fr09_13E_Data_Error;
    std::uint8_t GW_Fr01_15B_Data_Error;
    std::uint8_t DDCU_Fr01_1A2_Data_Error;
    std::uint8_t PDCU_Fr01_1A3_Data_Error;
    std::uint8_t TBOX_Fr02_1B6_Data_Error;
    std::uint8_t CDCS_Fr10_1BA_Data_Error;
    std::uint8_t CDCS_Fr13_1BB_Data_Error;
    std::uint8_t BDCS_Fr04_1E2_Data_Error;
    std::uint8_t BDCS_Fr05_1E4_Data_Error;
    std::uint8_t BDCS_Fr06_1E6_Data_Error;
    std::uint8_t BDCS_Fr10_29E_Data_Error;
    std::uint8_t EDU_Fr03_2C2_Data_Error;
    std::uint8_t EDU_Fr04_2C3_Data_Error;
    std::uint8_t BTM_Fr02_2F4_Data_Error;
    std::uint8_t HNS1_Fr02_2FD_Data_Error;
    std::uint8_t ICU_Fr02_325_Data_Error;
    std::uint8_t TBOX_Fr01_3E0_Data_Error;
    std::uint8_t EDU_Fr06_3E5_Data_Error;
    std::uint8_t CDCS_Fr15_203_Data_Error;
    std::uint8_t AICS_Fr2_243_Data_Error;
    std::uint8_t BMS_Fr03_258_Data_Error;
    std::uint8_t BDCS_Fr13_LIN2_285_Data_Error;
    std::uint8_t VCU_Fr04_2D4_Data_Error;
    std::uint8_t VCU_Fr03_2E3_Data_Error;
    std::uint8_t VCU_Fr07_0FC_Data_Error;
    std::uint8_t EDU_Fr05_1D5_Data_Error;
    std::uint8_t BDCS_Fr01_110_Data2_Error;
    std::uint8_t HNS2_Fr02_2FE_Data_Error;
    std::uint8_t ACU_Node_Lost;
    std::uint8_t BDCS_Node_Lost;
    std::uint8_t BTM_Node_Lost;
    std::uint8_t CDCS_Node_Lost;
    std::uint8_t DDCU_Node_Lost;
    std::uint8_t EDU_Node_Lost;
    std::uint8_t EPS_Node_Lost;
    std::uint8_t FMCU_Node_Lost;
    std::uint8_t GW_Node_Lost;
    std::uint8_t HNS_Node_Lost;
    std::uint8_t ICU_Node_Lost;
    std::uint8_t IDB_Node_Lost;
    std::uint8_t MCU_Node_Lost;
    std::uint8_t PDCU_Node_Lost;
    std::uint8_t RCU_Node_Lost;
    std::uint8_t TBOX_Node_Lost;
    std::uint8_t PDCS_Node_Lost;
    std::uint8_t BMS_Node_Lost;
    std::uint8_t FD3_Bus_Error;
    std::uint8_t FD6_Bus_Error;
    std::uint8_t FD8_Bus_Error;
    std::uint8_t Innr_Bus_Error;
    std::uint8_t Innr_MsgLost_Error;
    std::uint8_t Innr_MsgData_Error;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_HafcomDataFltInfo,MCU_Fr01_0AA_Data_Error,FMCU_Fr01_0AB_Data_Error,EPS_Fr01_0B1_Data_Error,EPS_Fr02_0B2_Data_Error,IDB_Fr02_0C0_Data_Error,ACU_Fr02_0C4_Data_Error,IDB_Fr03_0C5_Data_Error,IDB_Fr04_0C7_Data_Error,IDB_Fr10_0C9_Data_Error,VCU_Fr05_0E3_Data_Error,IDB_Fr01_0E5_Data_Error,RCU_Fr01_0E6_Data_Error,BTM_Fr01_0F3_Data_Error,VCU_Fr06_0FB_Data_Error,ACU_Fr01_102_Data_Error,VCU_Fr0E_108_Data_Error,BDCS_Fr01_110_Data1_Error,BDCS_Fr03_112_Data_Error,BDCS_Fr16_114_Data_Error,IDB_Fr09_121_Data_Error,IDB_Fr05_129_Data_Error,IDB_Fr06_12B_Data_Error,IDB_Fr07_12D_Data_Error,ICU_Fr01_131_Data_Error,RCU_Fr02_137_Data_Error,RCU_Fr03_138_Data_Error,RCU_Fr04_139_Data_Error,RCU_Fr06_13A_Data_Error,RCU_Fr07_13B_Data_Error,RCU_Fr08_13C_Data_Error,RCU_Fr05_13D_Data_Error,RCU_Fr09_13E_Data_Error,GW_Fr01_15B_Data_Error,DDCU_Fr01_1A2_Data_Error,PDCU_Fr01_1A3_Data_Error,TBOX_Fr02_1B6_Data_Error,CDCS_Fr10_1BA_Data_Error,CDCS_Fr13_1BB_Data_Error,BDCS_Fr04_1E2_Data_Error,BDCS_Fr05_1E4_Data_Error,BDCS_Fr06_1E6_Data_Error,BDCS_Fr10_29E_Data_Error,EDU_Fr03_2C2_Data_Error,EDU_Fr04_2C3_Data_Error,BTM_Fr02_2F4_Data_Error,HNS1_Fr02_2FD_Data_Error,ICU_Fr02_325_Data_Error,TBOX_Fr01_3E0_Data_Error,EDU_Fr06_3E5_Data_Error,CDCS_Fr15_203_Data_Error,AICS_Fr2_243_Data_Error,BMS_Fr03_258_Data_Error,BDCS_Fr13_LIN2_285_Data_Error,VCU_Fr04_2D4_Data_Error,VCU_Fr03_2E3_Data_Error,VCU_Fr07_0FC_Data_Error,EDU_Fr05_1D5_Data_Error,BDCS_Fr01_110_Data2_Error,HNS2_Fr02_2FE_Data_Error,ACU_Node_Lost,BDCS_Node_Lost,BTM_Node_Lost,CDCS_Node_Lost,DDCU_Node_Lost,EDU_Node_Lost,EPS_Node_Lost,FMCU_Node_Lost,GW_Node_Lost,HNS_Node_Lost,ICU_Node_Lost,IDB_Node_Lost,MCU_Node_Lost,PDCU_Node_Lost,RCU_Node_Lost,TBOX_Node_Lost,PDCS_Node_Lost,BMS_Node_Lost,FD3_Bus_Error,FD6_Bus_Error,FD8_Bus_Error,Innr_Bus_Error,Innr_MsgLost_Error,Innr_MsgData_Error);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFCOMDATAFLTINFO_H_
/* EOF */