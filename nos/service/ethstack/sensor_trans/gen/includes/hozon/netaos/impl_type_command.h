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
 * @file impl_type_command.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_COMMAND_H_
#define HOZON_NETAOS_IMPL_TYPE_COMMAND_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct Command {
    std::uint8_t enable_parking_slot_detection;
    std::uint8_t enable_object_detection;
    std::uint8_t enable_freespace_detection;
    std::uint8_t enable_uss;
    std::uint8_t enable_radar;
    std::uint8_t enable_lidar;
    std::uint8_t system_command;
    std::uint8_t emergencybrake_state;
    std::uint8_t system_reset;
    std::uint8_t reserved1;
    std::uint8_t reserved2;
    std::uint8_t reserved3;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::Command,enable_parking_slot_detection,enable_object_detection,enable_freespace_detection,enable_uss,enable_radar,enable_lidar,system_command,emergencybrake_state,system_reset,reserved1,reserved2,reserved3);

#endif // HOZON_NETAOS_IMPL_TYPE_COMMAND_H_
/* EOF */