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
 * @file impl_type_haflanedetectionout_a.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_HAFLANEDETECTIONOUT_A_H_
#define HOZON_NETAOS_IMPL_TYPE_HAFLANEDETECTIONOUT_A_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "hozon/netaos/impl_type_haflanedetectionout.h"
#include "ara/core/array.h"

namespace hozon {
namespace netaos {
using HafLaneDetectionOut_A = ara::core::Array<::hozon::netaos::HafLaneDetectionOut, 2>;

} // namespace netaos
} // namespace hozon


#endif // HOZON_NETAOS_IMPL_TYPE_HAFLANEDETECTIONOUT_A_H_
/* EOF */