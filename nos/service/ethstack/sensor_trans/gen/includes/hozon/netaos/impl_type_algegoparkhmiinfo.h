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
 * @file impl_type_algegoparkhmiinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_ALGEGOPARKHMIINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_ALGEGOPARKHMIINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AlgEgoParkHmiInfo {
    std::uint8_t ADCS4_PA_ParkBarPercent;
    float ADCS4_PA_GuideLineE_a;
    float ADCS4_PA_GuideLineE_b;
    float ADCS4_PA_GuideLineE_c;
    float ADCS4_PA_GuideLineE_d;
    float ADCS4_PA_GuideLineE_Xmin;
    float ADCS4_PA_GuideLineE_Xmax;
    std::uint8_t ADCS4_HourOfDay;
    std::uint8_t ADCS4_MinuteOfHour;
    std::uint8_t ADCS4_SecondOfMinute;
    std::uint16_t ADCS11_NNS_distance;
    std::uint16_t ADCS11_HPA_distance;
    std::uint16_t ADCS11_Parkingtimeremaining;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AlgEgoParkHmiInfo,ADCS4_PA_ParkBarPercent,ADCS4_PA_GuideLineE_a,ADCS4_PA_GuideLineE_b,ADCS4_PA_GuideLineE_c,ADCS4_PA_GuideLineE_d,ADCS4_PA_GuideLineE_Xmin,ADCS4_PA_GuideLineE_Xmax,ADCS4_HourOfDay,ADCS4_MinuteOfHour,ADCS4_SecondOfMinute,ADCS11_NNS_distance,ADCS11_HPA_distance,ADCS11_Parkingtimeremaining);

#endif // HOZON_NETAOS_IMPL_TYPE_ALGEGOPARKHMIINFO_H_
/* EOF */