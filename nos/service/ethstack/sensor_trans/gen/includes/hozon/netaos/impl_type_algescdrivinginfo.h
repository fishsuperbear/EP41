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
 * @file impl_type_algescdrivinginfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGESCDRIVINGINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGESCDRIVINGINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
namespace hozon {
namespace netaos {
struct AlgEscDrivingInfo {
    float ESC_VehicleSpeed;
    bool ESC_VehicleSpeedValid;
    bool ESC_BrakePedalSwitchStatus;
    bool ESC_BrakePedalSwitchValid;
    float BrkPedVal;
    float VehicleSpdDisplay;
    bool VehicleSpdDisplayValid;
    bool ESC_ApaStandStill;
    float ESC_LongAccValue;
    bool ESC_LongAccValue_Valid;
    float ESC_LatAccValue;
    bool ESC_LatAccValue_Valid;
    float ESC_YawRate;
    bool ESC_YawRate_Valid;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgEscDrivingInfo,ESC_VehicleSpeed,ESC_VehicleSpeedValid,ESC_BrakePedalSwitchStatus,ESC_BrakePedalSwitchValid,BrkPedVal,VehicleSpdDisplay,VehicleSpdDisplayValid,ESC_ApaStandStill,ESC_LongAccValue,ESC_LongAccValue_Valid,ESC_LatAccValue,ESC_LatAccValue_Valid,ESC_YawRate,ESC_YawRate_Valid);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGESCDRIVINGINFO_H_
/* EOF */