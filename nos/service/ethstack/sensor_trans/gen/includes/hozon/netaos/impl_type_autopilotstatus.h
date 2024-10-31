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
 * @file impl_type_autopilotstatus.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_AUTOPILOTSTATUS_H_
#define HOZON_NETAOS_IMPL_TYPE_AUTOPILOTSTATUS_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct AutopilotStatus {
    std::uint8_t processing_status;
    std::uint8_t camera_status;
    std::uint8_t uss_status;
    std::uint8_t radar_status;
    std::uint8_t lidar_status;
    std::uint8_t velocity_status;
    std::uint8_t perception_status;
    std::uint8_t planning_status;
    std::uint8_t controlling_status;
    std::uint8_t turn_light_status;
    std::uint8_t localization_status;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::AutopilotStatus,processing_status,camera_status,uss_status,radar_status,lidar_status,velocity_status,perception_status,planning_status,controlling_status,turn_light_status,localization_status);

#endif // HOZON_NETAOS_IMPL_TYPE_AUTOPILOTSTATUS_H_
/* EOF */