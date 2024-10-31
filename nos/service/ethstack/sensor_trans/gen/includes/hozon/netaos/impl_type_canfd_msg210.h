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
 * @file impl_type_canfd_msg210.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_CANFD_MSG210_H_
#define HOZON_NETAOS_IMPL_TYPE_CANFD_MSG210_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct CANFD_Msg210 {
    std::uint8_t ADCS9_AVM_MODdetectionSts;
    std::uint8_t ADCS9_HPA_OnOffSetSts;
    std::uint8_t ADCS9_HPA_CrossingNumber;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::CANFD_Msg210,ADCS9_AVM_MODdetectionSts,ADCS9_HPA_OnOffSetSts,ADCS9_HPA_CrossingNumber);

#endif // HOZON_NETAOS_IMPL_TYPE_CANFD_MSG210_H_
/* EOF */