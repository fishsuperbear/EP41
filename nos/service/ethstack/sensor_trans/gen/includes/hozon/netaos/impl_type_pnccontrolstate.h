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
 * @file impl_type_pnccontrolstate.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_PNCCONTROLSTATE_H_
#define HOZON_NETAOS_IMPL_TYPE_PNCCONTROLSTATE_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct PNCControlState {
    std::uint8_t fct_state;
    std::uint8_t m_iuss_state_obs;
    std::uint8_t need_replan_stop;
    std::uint8_t plan_trigger;
    std::uint8_t control_enable;
    std::uint8_t control_status;
    std::uint8_t pnc_run_state;
    std::uint8_t pnc_warninginfo;
    std::uint8_t pnc_ADCS4_Tex;
    std::uint8_t pnc_ADCS4_PA_failinfo;
    std::uint8_t FAPA;
    bool RPA;
    bool TBA;
    bool LAPA_MapBuilding;
    bool LAPA_Cruising;
    bool LAPA_PickUp;
    bool ISM;
    bool AVP;
    bool pnc_ADCS4_TBA_failinfo;
    std::uint8_t pnc_ADCS4_RPA_failinfo;
    std::uint8_t pnc_ADCS4_LAPA_MapBuilding_failinfo;
    std::uint8_t pnc_ADCS4_LAPA_Cruising_failinfo;
    std::uint8_t pnc_ADCS4_LAPA_PickUp_failinfo;
    std::uint8_t pnc_ADCS4_ISM_failinfo;
    std::uint8_t pnc_ADCS4_AVP_failinfo;
    std::uint8_t TBA_text;
    std::uint8_t reserved3;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::PNCControlState,fct_state,m_iuss_state_obs,need_replan_stop,plan_trigger,control_enable,control_status,pnc_run_state,pnc_warninginfo,pnc_ADCS4_Tex,pnc_ADCS4_PA_failinfo,FAPA,RPA,TBA,LAPA_MapBuilding,LAPA_Cruising,LAPA_PickUp,ISM,AVP,pnc_ADCS4_TBA_failinfo,pnc_ADCS4_RPA_failinfo,pnc_ADCS4_LAPA_MapBuilding_failinfo,pnc_ADCS4_LAPA_Cruising_failinfo,pnc_ADCS4_LAPA_PickUp_failinfo,pnc_ADCS4_ISM_failinfo,pnc_ADCS4_AVP_failinfo,TBA_text,reserved3);

#endif // HOZON_NETAOS_IMPL_TYPE_PNCCONTROLSTATE_H_
/* EOF */