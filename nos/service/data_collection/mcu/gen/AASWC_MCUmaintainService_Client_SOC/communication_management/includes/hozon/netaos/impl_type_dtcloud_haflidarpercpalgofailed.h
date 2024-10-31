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
 * @file impl_type_dtcloud_haflidarpercpalgofailed.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOFAILED_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOFAILED_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_HafLidarPercpAlgoFailed {
    std::uint8_t algo_configload_error;
    std::uint8_t algo_pointcloud_preproc_error;
    std::uint8_t algo_target_detn_error;
    std::uint8_t algo_target_track_error;
    std::uint8_t alog_grd_detn_error;
    std::uint8_t algo_lane_detn_error;
    std::uint8_t algo_freespace_detn_error;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_HafLidarPercpAlgoFailed,algo_configload_error,algo_pointcloud_preproc_error,algo_target_detn_error,algo_target_track_error,alog_grd_detn_error,algo_lane_detn_error,algo_freespace_detn_error);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFLIDARPERCPALGOFAILED_H_
/* EOF */