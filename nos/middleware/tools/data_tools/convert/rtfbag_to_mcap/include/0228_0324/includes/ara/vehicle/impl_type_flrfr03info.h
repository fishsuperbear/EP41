/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_VEHICLE_IMPL_TYPE_FLRFR03INFO_H
#define ARA_VEHICLE_IMPL_TYPE_FLRFR03INFO_H
#include <cfloat>
#include <cmath>
#include "impl_type_uint8.h"
#include "impl_type_float.h"

namespace ara {
namespace vehicle {
struct FLRFr03Info {
    ::UInt8 flr_aeb_prefill;
    ::UInt8 flr_aeb_dbs_level;
    ::UInt8 flr_aeb_warn;
    ::UInt8 flr_aeb_on_off_sta;
    ::Float flr_aeb_dec_cmd;
    ::UInt8 flr_aeb_partial_act;
    ::UInt8 flr_aeb_warn_set_sta;
    ::UInt8 flr_aeb_stop_req;
    ::UInt8 flr_aeb_full_act;
    ::UInt8 flr_aeb_fail_info;
    ::Float scc_obj_dist;
    ::Float scc_obj_lat_pos;
    ::UInt8 flr_fr03_msg_counter;
    ::UInt8 flr_fr03_checksum;

    static bool IsPlane()
    {
        return true;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(flr_aeb_prefill);
        fun(flr_aeb_dbs_level);
        fun(flr_aeb_warn);
        fun(flr_aeb_on_off_sta);
        fun(flr_aeb_dec_cmd);
        fun(flr_aeb_partial_act);
        fun(flr_aeb_warn_set_sta);
        fun(flr_aeb_stop_req);
        fun(flr_aeb_full_act);
        fun(flr_aeb_fail_info);
        fun(scc_obj_dist);
        fun(scc_obj_lat_pos);
        fun(flr_fr03_msg_counter);
        fun(flr_fr03_checksum);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(flr_aeb_prefill);
        fun(flr_aeb_dbs_level);
        fun(flr_aeb_warn);
        fun(flr_aeb_on_off_sta);
        fun(flr_aeb_dec_cmd);
        fun(flr_aeb_partial_act);
        fun(flr_aeb_warn_set_sta);
        fun(flr_aeb_stop_req);
        fun(flr_aeb_full_act);
        fun(flr_aeb_fail_info);
        fun(scc_obj_dist);
        fun(scc_obj_lat_pos);
        fun(flr_fr03_msg_counter);
        fun(flr_fr03_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("flr_aeb_prefill", flr_aeb_prefill);
        fun("flr_aeb_dbs_level", flr_aeb_dbs_level);
        fun("flr_aeb_warn", flr_aeb_warn);
        fun("flr_aeb_on_off_sta", flr_aeb_on_off_sta);
        fun("flr_aeb_dec_cmd", flr_aeb_dec_cmd);
        fun("flr_aeb_partial_act", flr_aeb_partial_act);
        fun("flr_aeb_warn_set_sta", flr_aeb_warn_set_sta);
        fun("flr_aeb_stop_req", flr_aeb_stop_req);
        fun("flr_aeb_full_act", flr_aeb_full_act);
        fun("flr_aeb_fail_info", flr_aeb_fail_info);
        fun("scc_obj_dist", scc_obj_dist);
        fun("scc_obj_lat_pos", scc_obj_lat_pos);
        fun("flr_fr03_msg_counter", flr_fr03_msg_counter);
        fun("flr_fr03_checksum", flr_fr03_checksum);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("flr_aeb_prefill", flr_aeb_prefill);
        fun("flr_aeb_dbs_level", flr_aeb_dbs_level);
        fun("flr_aeb_warn", flr_aeb_warn);
        fun("flr_aeb_on_off_sta", flr_aeb_on_off_sta);
        fun("flr_aeb_dec_cmd", flr_aeb_dec_cmd);
        fun("flr_aeb_partial_act", flr_aeb_partial_act);
        fun("flr_aeb_warn_set_sta", flr_aeb_warn_set_sta);
        fun("flr_aeb_stop_req", flr_aeb_stop_req);
        fun("flr_aeb_full_act", flr_aeb_full_act);
        fun("flr_aeb_fail_info", flr_aeb_fail_info);
        fun("scc_obj_dist", scc_obj_dist);
        fun("scc_obj_lat_pos", scc_obj_lat_pos);
        fun("flr_fr03_msg_counter", flr_fr03_msg_counter);
        fun("flr_fr03_checksum", flr_fr03_checksum);
    }

    bool operator==(const ::ara::vehicle::FLRFr03Info& t) const
    {
        return (flr_aeb_prefill == t.flr_aeb_prefill) && (flr_aeb_dbs_level == t.flr_aeb_dbs_level) && (flr_aeb_warn == t.flr_aeb_warn) && (flr_aeb_on_off_sta == t.flr_aeb_on_off_sta) && (fabs(static_cast<double>(flr_aeb_dec_cmd - t.flr_aeb_dec_cmd)) < DBL_EPSILON) && (flr_aeb_partial_act == t.flr_aeb_partial_act) && (flr_aeb_warn_set_sta == t.flr_aeb_warn_set_sta) && (flr_aeb_stop_req == t.flr_aeb_stop_req) && (flr_aeb_full_act == t.flr_aeb_full_act) && (flr_aeb_fail_info == t.flr_aeb_fail_info) && (fabs(static_cast<double>(scc_obj_dist - t.scc_obj_dist)) < DBL_EPSILON) && (fabs(static_cast<double>(scc_obj_lat_pos - t.scc_obj_lat_pos)) < DBL_EPSILON) && (flr_fr03_msg_counter == t.flr_fr03_msg_counter) && (flr_fr03_checksum == t.flr_fr03_checksum);
    }
};
} // namespace vehicle
} // namespace ara


#endif // ARA_VEHICLE_IMPL_TYPE_FLRFR03INFO_H
