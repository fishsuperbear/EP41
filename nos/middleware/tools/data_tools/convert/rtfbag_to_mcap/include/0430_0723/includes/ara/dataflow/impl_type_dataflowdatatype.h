/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef ARA_DATAFLOW_IMPL_TYPE_DATAFLOWDATATYPE_H
#define ARA_DATAFLOW_IMPL_TYPE_DATAFLOWDATATYPE_H
#include <cfloat>
#include <cmath>
#include "ara/dataflow/impl_type_header.h"
#include "impl_type_uint8.h"
#include "impl_type_string.h"
#include "impl_type_uint32.h"

namespace ara {
namespace dataflow {
struct DataFlowDataType {
    ::ara::dataflow::Header header;
    ::UInt8 channel_type;
    ::String channel_id;
    ::UInt32 recv_flow_stats_pps;
    ::UInt32 recv_flow_stats_bps;
    ::UInt32 send_flow_stats_pps;
    ::UInt32 send_flow_stats_bps;

    static bool IsPlane()
    {
        return false;
    }


    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(header);
        fun(channel_type);
        fun(channel_id);
        fun(recv_flow_stats_pps);
        fun(recv_flow_stats_bps);
        fun(send_flow_stats_pps);
        fun(send_flow_stats_bps);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(header);
        fun(channel_type);
        fun(channel_id);
        fun(recv_flow_stats_pps);
        fun(recv_flow_stats_bps);
        fun(send_flow_stats_pps);
        fun(send_flow_stats_bps);
    }

    template<typename F>
    void enumerate_internal(F& fun)
    {
        fun("header", header);
        fun("channel_type", channel_type);
        fun("channel_id", channel_id);
        fun("recv_flow_stats_pps", recv_flow_stats_pps);
        fun("recv_flow_stats_bps", recv_flow_stats_bps);
        fun("send_flow_stats_pps", send_flow_stats_pps);
        fun("send_flow_stats_bps", send_flow_stats_bps);
    }

    template<typename F>
    void enumerate_internal(F& fun) const
    {
        fun("header", header);
        fun("channel_type", channel_type);
        fun("channel_id", channel_id);
        fun("recv_flow_stats_pps", recv_flow_stats_pps);
        fun("recv_flow_stats_bps", recv_flow_stats_bps);
        fun("send_flow_stats_pps", send_flow_stats_pps);
        fun("send_flow_stats_bps", send_flow_stats_bps);
    }

    bool operator==(const ::ara::dataflow::DataFlowDataType& t) const
    {
        return (header == t.header) && (channel_type == t.channel_type) && (channel_id == t.channel_id) && (recv_flow_stats_pps == t.recv_flow_stats_pps) && (recv_flow_stats_bps == t.recv_flow_stats_bps) && (send_flow_stats_pps == t.send_flow_stats_pps) && (send_flow_stats_bps == t.send_flow_stats_bps);
    }
};
} // namespace dataflow
} // namespace ara


#endif // ARA_DATAFLOW_IMPL_TYPE_DATAFLOWDATATYPE_H
