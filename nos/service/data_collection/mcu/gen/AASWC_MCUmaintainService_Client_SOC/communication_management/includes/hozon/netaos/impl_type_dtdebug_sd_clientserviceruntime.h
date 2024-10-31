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
 * @file impl_type_dtdebug_sd_clientserviceruntime.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTDEBUG_SD_CLIENTSERVICERUNTIME_H_
#define HOZON_NETAOS_IMPL_TYPE_DTDEBUG_SD_CLIENTSERVICERUNTIME_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtDebug_Sd_ClientServiceRuntime {
    std::uint8_t SoTcpConOpend;
    std::uint16_t SoConOpend;
    std::uint8_t RetryEnabled;
    std::uint16_t ServicePhase;
    std::uint16_t LastServicePhase;
    std::uint16_t State;
    std::uint8_t CurRepetition;
    std::uint16_t TxFindCount;
    std::uint16_t RxOfferCount;
    std::uint16_t RxStopOfferCount;
    std::uint16_t TxSubscribeCount;
    std::uint16_t TxStopSubscribeCount;
    std::uint16_t RxSubscribeAckCount;
    std::uint16_t RxSubscribeNackCount;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtDebug_Sd_ClientServiceRuntime,SoTcpConOpend,SoConOpend,RetryEnabled,ServicePhase,LastServicePhase,State,CurRepetition,TxFindCount,RxOfferCount,RxStopOfferCount,TxSubscribeCount,TxStopSubscribeCount,RxSubscribeAckCount,RxSubscribeNackCount);

#endif // HOZON_NETAOS_IMPL_TYPE_DTDEBUG_SD_CLIENTSERVICERUNTIME_H_
/* EOF */