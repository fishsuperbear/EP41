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
 * @file impl_type_dtcloud_egomemmsg.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_EGOMEMMSG_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_EGOMEMMSG_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_EgoMemMsg {
    std::uint8_t CDCS11_VoiceMode;
    std::uint8_t RCTA_OnOffSet_mem;
    std::uint8_t FCTA_OnOffSet_mem;
    std::uint8_t DOW_OnOffSet_mem;
    std::uint8_t RCW_OnOffSet_mem;
    std::uint8_t LCA_OnOffSet_mem;
    std::uint8_t TSR_OnOffSet_mem;
    std::uint8_t RCW_OverspeedOnOffSet_mem;
    std::uint8_t IHBC_OnOffSet_mem;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_EgoMemMsg,CDCS11_VoiceMode,RCTA_OnOffSet_mem,FCTA_OnOffSet_mem,DOW_OnOffSet_mem,RCW_OnOffSet_mem,LCA_OnOffSet_mem,TSR_OnOffSet_mem,RCW_OverspeedOnOffSet_mem,IHBC_OnOffSet_mem);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_EGOMEMMSG_H_
/* EOF */