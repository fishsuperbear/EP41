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
 * @file impl_type_dtcloud_hafsocclustfltvalinfo.h
 * @brief 
 * @date  
 *
 */
#ifndef HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFSOCCLUSTFLTVALINFO_H_
#define HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFSOCCLUSTFLTVALINFO_H_

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif
#include "ara/com/serializer/transformation_reflection.h"
#include <cstdint>
namespace hozon {
namespace netaos {
struct DtCloud_HafsocClustFltValInfo {
    std::uint8_t SOC_platform;
    std::uint8_t f_7v_fusion;
    std::uint8_t multi_sensor_fusion;
    std::uint8_t f_30_obj;
    std::uint8_t f_30_freespace;
    std::uint8_t f_120_obj;
    std::uint8_t f_120_freespace;
    std::uint8_t fv_lane;
    std::uint8_t fv_roadmark;
    std::uint8_t flv_obj;
    std::uint8_t flv_freespace;
    std::uint8_t frv_obj;
    std::uint8_t frv_freespace;
    std::uint8_t rlv_obj;
    std::uint8_t rlv_freespace;
    std::uint8_t rrv_obj;
    std::uint8_t rrv_freespace;
    std::uint8_t rv_obj;
    std::uint8_t rv_lane;
    std::uint8_t rv_freespace;
    std::uint8_t llidar_obj;
    std::uint8_t llidar_lane;
    std::uint8_t llidar_freespace;
    std::uint8_t rlidar_obj;
    std::uint8_t rlidar_lane;
    std::uint8_t rlidar_freespace;
    std::uint8_t fradar_obj;
    std::uint8_t frr_obj;
    std::uint8_t flr_obj;
    std::uint8_t rrr_obj;
    std::uint8_t rlr_obj;
    std::uint8_t nnp_location;
    std::uint8_t hd_map;
    std::uint8_t fusion_obj;
    std::uint8_t fusion_parkinglot;
    std::uint8_t uss_parkinglot;
    std::uint8_t uss_obstacle;
    std::uint8_t avm_freespace;
    std::uint8_t hpp_location;
    std::uint8_t slam_map;
    std::uint8_t front_avm_image;
    std::uint8_t left_avm_image;
    std::uint8_t right_avm_image;
    std::uint8_t rear_avm_image;
    std::uint8_t fl_pdc_uss;
    std::uint8_t fml_pdc_uss;
    std::uint8_t fmr_pdc_uss;
    std::uint8_t fr_pdc_uss;
    std::uint8_t rl_pdc_uss;
    std::uint8_t rml_pdc_uss;
    std::uint8_t rmr_pdc_uss;
    std::uint8_t rr_pdc_uss;
    std::uint8_t fls_apa_uss;
    std::uint8_t frs_apa_uss;
    std::uint8_t rls_apa_uss;
    std::uint8_t rrs_apa_uss;
    std::uint8_t reserved1;
    std::uint8_t reserved2;
    std::uint8_t reserved3;
    std::uint8_t reserved4;
};
} // namespace netaos
} // namespace hozon


STRUCTURE_REFLECTION_DEF(::hozon::netaos::DtCloud_HafsocClustFltValInfo,SOC_platform,f_7v_fusion,multi_sensor_fusion,f_30_obj,f_30_freespace,f_120_obj,f_120_freespace,fv_lane,fv_roadmark,flv_obj,flv_freespace,frv_obj,frv_freespace,rlv_obj,rlv_freespace,rrv_obj,rrv_freespace,rv_obj,rv_lane,rv_freespace,llidar_obj,llidar_lane,llidar_freespace,rlidar_obj,rlidar_lane,rlidar_freespace,fradar_obj,frr_obj,flr_obj,rrr_obj,rlr_obj,nnp_location,hd_map,fusion_obj,fusion_parkinglot,uss_parkinglot,uss_obstacle,avm_freespace,hpp_location,slam_map,front_avm_image,left_avm_image,right_avm_image,rear_avm_image,fl_pdc_uss,fml_pdc_uss,fmr_pdc_uss,fr_pdc_uss,rl_pdc_uss,rml_pdc_uss,rmr_pdc_uss,rr_pdc_uss,fls_apa_uss,frs_apa_uss,rls_apa_uss,rrs_apa_uss,reserved1,reserved2,reserved3,reserved4);

#endif // HOZON_NETAOS_IMPL_TYPE_DTCLOUD_HAFSOCCLUSTFLTVALINFO_H_
/* EOF */