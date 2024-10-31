/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFSOCCLUSTFLTVALINFO_H
#define HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFSOCCLUSTFLTVALINFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"

namespace hozon {
namespace soc_mcu {
struct DtCloud_HafsocClustFltValInfo {
    ::UInt8 SOC_platform;
    ::UInt8 f_7v_fusion;
    ::UInt8 multi_sensor_fusion;
    ::UInt8 f_30_obj;
    ::UInt8 f_30_freespace;
    ::UInt8 f_120_obj;
    ::UInt8 f_120_freespace;
    ::UInt8 fv_lane;
    ::UInt8 fv_roadmark;
    ::UInt8 flv_obj;
    ::UInt8 flv_freespace;
    ::UInt8 frv_obj;
    ::UInt8 frv_freespace;
    ::UInt8 rlv_obj;
    ::UInt8 rlv_freespace;
    ::UInt8 rrv_obj;
    ::UInt8 rrv_freespace;
    ::UInt8 rv_obj;
    ::UInt8 rv_lane;
    ::UInt8 rv_freespace;
    ::UInt8 llidar_obj;
    ::UInt8 llidar_lane;
    ::UInt8 llidar_freespace;
    ::UInt8 rlidar_obj;
    ::UInt8 rlidar_lane;
    ::UInt8 rlidar_freespace;
    ::UInt8 fradar_obj;
    ::UInt8 frr_obj;
    ::UInt8 flr_obj;
    ::UInt8 rrr_obj;
    ::UInt8 rlr_obj;
    ::UInt8 nnp_location;
    ::UInt8 hd_map;
    ::UInt8 fusion_obj;
    ::UInt8 fusion_parkinglot;
    ::UInt8 uss_parkinglot;
    ::UInt8 uss_obstacle;
    ::UInt8 avm_freespace;
    ::UInt8 hpp_location;
    ::UInt8 slam_map;
    ::UInt8 front_avm_image;
    ::UInt8 left_avm_image;
    ::UInt8 right_avm_image;
    ::UInt8 rear_avm_image;
    ::UInt8 fl_pdc_uss;
    ::UInt8 fml_pdc_uss;
    ::UInt8 fmr_pdc_uss;
    ::UInt8 fr_pdc_uss;
    ::UInt8 rl_pdc_uss;
    ::UInt8 rml_pdc_uss;
    ::UInt8 rmr_pdc_uss;
    ::UInt8 rr_pdc_uss;
    ::UInt8 fls_apa_uss;
    ::UInt8 frs_apa_uss;
    ::UInt8 rls_apa_uss;
    ::UInt8 rrs_apa_uss;
    ::UInt8 reserved1;
    ::UInt8 reserved2;
    ::UInt8 reserved3;
    ::UInt8 reserved4;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(SOC_platform);
        fun(f_7v_fusion);
        fun(multi_sensor_fusion);
        fun(f_30_obj);
        fun(f_30_freespace);
        fun(f_120_obj);
        fun(f_120_freespace);
        fun(fv_lane);
        fun(fv_roadmark);
        fun(flv_obj);
        fun(flv_freespace);
        fun(frv_obj);
        fun(frv_freespace);
        fun(rlv_obj);
        fun(rlv_freespace);
        fun(rrv_obj);
        fun(rrv_freespace);
        fun(rv_obj);
        fun(rv_lane);
        fun(rv_freespace);
        fun(llidar_obj);
        fun(llidar_lane);
        fun(llidar_freespace);
        fun(rlidar_obj);
        fun(rlidar_lane);
        fun(rlidar_freespace);
        fun(fradar_obj);
        fun(frr_obj);
        fun(flr_obj);
        fun(rrr_obj);
        fun(rlr_obj);
        fun(nnp_location);
        fun(hd_map);
        fun(fusion_obj);
        fun(fusion_parkinglot);
        fun(uss_parkinglot);
        fun(uss_obstacle);
        fun(avm_freespace);
        fun(hpp_location);
        fun(slam_map);
        fun(front_avm_image);
        fun(left_avm_image);
        fun(right_avm_image);
        fun(rear_avm_image);
        fun(fl_pdc_uss);
        fun(fml_pdc_uss);
        fun(fmr_pdc_uss);
        fun(fr_pdc_uss);
        fun(rl_pdc_uss);
        fun(rml_pdc_uss);
        fun(rmr_pdc_uss);
        fun(rr_pdc_uss);
        fun(fls_apa_uss);
        fun(frs_apa_uss);
        fun(rls_apa_uss);
        fun(rrs_apa_uss);
        fun(reserved1);
        fun(reserved2);
        fun(reserved3);
        fun(reserved4);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(SOC_platform);
        fun(f_7v_fusion);
        fun(multi_sensor_fusion);
        fun(f_30_obj);
        fun(f_30_freespace);
        fun(f_120_obj);
        fun(f_120_freespace);
        fun(fv_lane);
        fun(fv_roadmark);
        fun(flv_obj);
        fun(flv_freespace);
        fun(frv_obj);
        fun(frv_freespace);
        fun(rlv_obj);
        fun(rlv_freespace);
        fun(rrv_obj);
        fun(rrv_freespace);
        fun(rv_obj);
        fun(rv_lane);
        fun(rv_freespace);
        fun(llidar_obj);
        fun(llidar_lane);
        fun(llidar_freespace);
        fun(rlidar_obj);
        fun(rlidar_lane);
        fun(rlidar_freespace);
        fun(fradar_obj);
        fun(frr_obj);
        fun(flr_obj);
        fun(rrr_obj);
        fun(rlr_obj);
        fun(nnp_location);
        fun(hd_map);
        fun(fusion_obj);
        fun(fusion_parkinglot);
        fun(uss_parkinglot);
        fun(uss_obstacle);
        fun(avm_freespace);
        fun(hpp_location);
        fun(slam_map);
        fun(front_avm_image);
        fun(left_avm_image);
        fun(right_avm_image);
        fun(rear_avm_image);
        fun(fl_pdc_uss);
        fun(fml_pdc_uss);
        fun(fmr_pdc_uss);
        fun(fr_pdc_uss);
        fun(rl_pdc_uss);
        fun(rml_pdc_uss);
        fun(rmr_pdc_uss);
        fun(rr_pdc_uss);
        fun(fls_apa_uss);
        fun(frs_apa_uss);
        fun(rls_apa_uss);
        fun(rrs_apa_uss);
        fun(reserved1);
        fun(reserved2);
        fun(reserved3);
        fun(reserved4);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("SOC_platform", SOC_platform);
        fun("f_7v_fusion", f_7v_fusion);
        fun("multi_sensor_fusion", multi_sensor_fusion);
        fun("f_30_obj", f_30_obj);
        fun("f_30_freespace", f_30_freespace);
        fun("f_120_obj", f_120_obj);
        fun("f_120_freespace", f_120_freespace);
        fun("fv_lane", fv_lane);
        fun("fv_roadmark", fv_roadmark);
        fun("flv_obj", flv_obj);
        fun("flv_freespace", flv_freespace);
        fun("frv_obj", frv_obj);
        fun("frv_freespace", frv_freespace);
        fun("rlv_obj", rlv_obj);
        fun("rlv_freespace", rlv_freespace);
        fun("rrv_obj", rrv_obj);
        fun("rrv_freespace", rrv_freespace);
        fun("rv_obj", rv_obj);
        fun("rv_lane", rv_lane);
        fun("rv_freespace", rv_freespace);
        fun("llidar_obj", llidar_obj);
        fun("llidar_lane", llidar_lane);
        fun("llidar_freespace", llidar_freespace);
        fun("rlidar_obj", rlidar_obj);
        fun("rlidar_lane", rlidar_lane);
        fun("rlidar_freespace", rlidar_freespace);
        fun("fradar_obj", fradar_obj);
        fun("frr_obj", frr_obj);
        fun("flr_obj", flr_obj);
        fun("rrr_obj", rrr_obj);
        fun("rlr_obj", rlr_obj);
        fun("nnp_location", nnp_location);
        fun("hd_map", hd_map);
        fun("fusion_obj", fusion_obj);
        fun("fusion_parkinglot", fusion_parkinglot);
        fun("uss_parkinglot", uss_parkinglot);
        fun("uss_obstacle", uss_obstacle);
        fun("avm_freespace", avm_freespace);
        fun("hpp_location", hpp_location);
        fun("slam_map", slam_map);
        fun("front_avm_image", front_avm_image);
        fun("left_avm_image", left_avm_image);
        fun("right_avm_image", right_avm_image);
        fun("rear_avm_image", rear_avm_image);
        fun("fl_pdc_uss", fl_pdc_uss);
        fun("fml_pdc_uss", fml_pdc_uss);
        fun("fmr_pdc_uss", fmr_pdc_uss);
        fun("fr_pdc_uss", fr_pdc_uss);
        fun("rl_pdc_uss", rl_pdc_uss);
        fun("rml_pdc_uss", rml_pdc_uss);
        fun("rmr_pdc_uss", rmr_pdc_uss);
        fun("rr_pdc_uss", rr_pdc_uss);
        fun("fls_apa_uss", fls_apa_uss);
        fun("frs_apa_uss", frs_apa_uss);
        fun("rls_apa_uss", rls_apa_uss);
        fun("rrs_apa_uss", rrs_apa_uss);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
        fun("reserved4", reserved4);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("SOC_platform", SOC_platform);
        fun("f_7v_fusion", f_7v_fusion);
        fun("multi_sensor_fusion", multi_sensor_fusion);
        fun("f_30_obj", f_30_obj);
        fun("f_30_freespace", f_30_freespace);
        fun("f_120_obj", f_120_obj);
        fun("f_120_freespace", f_120_freespace);
        fun("fv_lane", fv_lane);
        fun("fv_roadmark", fv_roadmark);
        fun("flv_obj", flv_obj);
        fun("flv_freespace", flv_freespace);
        fun("frv_obj", frv_obj);
        fun("frv_freespace", frv_freespace);
        fun("rlv_obj", rlv_obj);
        fun("rlv_freespace", rlv_freespace);
        fun("rrv_obj", rrv_obj);
        fun("rrv_freespace", rrv_freespace);
        fun("rv_obj", rv_obj);
        fun("rv_lane", rv_lane);
        fun("rv_freespace", rv_freespace);
        fun("llidar_obj", llidar_obj);
        fun("llidar_lane", llidar_lane);
        fun("llidar_freespace", llidar_freespace);
        fun("rlidar_obj", rlidar_obj);
        fun("rlidar_lane", rlidar_lane);
        fun("rlidar_freespace", rlidar_freespace);
        fun("fradar_obj", fradar_obj);
        fun("frr_obj", frr_obj);
        fun("flr_obj", flr_obj);
        fun("rrr_obj", rrr_obj);
        fun("rlr_obj", rlr_obj);
        fun("nnp_location", nnp_location);
        fun("hd_map", hd_map);
        fun("fusion_obj", fusion_obj);
        fun("fusion_parkinglot", fusion_parkinglot);
        fun("uss_parkinglot", uss_parkinglot);
        fun("uss_obstacle", uss_obstacle);
        fun("avm_freespace", avm_freespace);
        fun("hpp_location", hpp_location);
        fun("slam_map", slam_map);
        fun("front_avm_image", front_avm_image);
        fun("left_avm_image", left_avm_image);
        fun("right_avm_image", right_avm_image);
        fun("rear_avm_image", rear_avm_image);
        fun("fl_pdc_uss", fl_pdc_uss);
        fun("fml_pdc_uss", fml_pdc_uss);
        fun("fmr_pdc_uss", fmr_pdc_uss);
        fun("fr_pdc_uss", fr_pdc_uss);
        fun("rl_pdc_uss", rl_pdc_uss);
        fun("rml_pdc_uss", rml_pdc_uss);
        fun("rmr_pdc_uss", rmr_pdc_uss);
        fun("rr_pdc_uss", rr_pdc_uss);
        fun("fls_apa_uss", fls_apa_uss);
        fun("frs_apa_uss", frs_apa_uss);
        fun("rls_apa_uss", rls_apa_uss);
        fun("rrs_apa_uss", rrs_apa_uss);
        fun("reserved1", reserved1);
        fun("reserved2", reserved2);
        fun("reserved3", reserved3);
        fun("reserved4", reserved4);
    }

    bool operator==(const ::hozon::soc_mcu::DtCloud_HafsocClustFltValInfo& t) const
    {
        return (SOC_platform == t.SOC_platform) && (f_7v_fusion == t.f_7v_fusion) && (multi_sensor_fusion == t.multi_sensor_fusion) && (f_30_obj == t.f_30_obj) && (f_30_freespace == t.f_30_freespace) && (f_120_obj == t.f_120_obj) && (f_120_freespace == t.f_120_freespace) && (fv_lane == t.fv_lane) && (fv_roadmark == t.fv_roadmark) && (flv_obj == t.flv_obj) && (flv_freespace == t.flv_freespace) && (frv_obj == t.frv_obj) && (frv_freespace == t.frv_freespace) && (rlv_obj == t.rlv_obj) && (rlv_freespace == t.rlv_freespace) && (rrv_obj == t.rrv_obj) && (rrv_freespace == t.rrv_freespace) && (rv_obj == t.rv_obj) && (rv_lane == t.rv_lane) && (rv_freespace == t.rv_freespace) && (llidar_obj == t.llidar_obj) && (llidar_lane == t.llidar_lane) && (llidar_freespace == t.llidar_freespace) && (rlidar_obj == t.rlidar_obj) && (rlidar_lane == t.rlidar_lane) && (rlidar_freespace == t.rlidar_freespace) && (fradar_obj == t.fradar_obj) && (frr_obj == t.frr_obj) && (flr_obj == t.flr_obj) && (rrr_obj == t.rrr_obj) && (rlr_obj == t.rlr_obj) && (nnp_location == t.nnp_location) && (hd_map == t.hd_map) && (fusion_obj == t.fusion_obj) && (fusion_parkinglot == t.fusion_parkinglot) && (uss_parkinglot == t.uss_parkinglot) && (uss_obstacle == t.uss_obstacle) && (avm_freespace == t.avm_freespace) && (hpp_location == t.hpp_location) && (slam_map == t.slam_map) && (front_avm_image == t.front_avm_image) && (left_avm_image == t.left_avm_image) && (right_avm_image == t.right_avm_image) && (rear_avm_image == t.rear_avm_image) && (fl_pdc_uss == t.fl_pdc_uss) && (fml_pdc_uss == t.fml_pdc_uss) && (fmr_pdc_uss == t.fmr_pdc_uss) && (fr_pdc_uss == t.fr_pdc_uss) && (rl_pdc_uss == t.rl_pdc_uss) && (rml_pdc_uss == t.rml_pdc_uss) && (rmr_pdc_uss == t.rmr_pdc_uss) && (rr_pdc_uss == t.rr_pdc_uss) && (fls_apa_uss == t.fls_apa_uss) && (frs_apa_uss == t.frs_apa_uss) && (rls_apa_uss == t.rls_apa_uss) && (rrs_apa_uss == t.rrs_apa_uss) && (reserved1 == t.reserved1) && (reserved2 == t.reserved2) && (reserved3 == t.reserved3) && (reserved4 == t.reserved4);
    }
};
} // namespace soc_mcu
} // namespace hozon


#endif // HOZON_SOC_MCU_IMPL_TYPE_DTCLOUD_HAFSOCCLUSTFLTVALINFO_H
