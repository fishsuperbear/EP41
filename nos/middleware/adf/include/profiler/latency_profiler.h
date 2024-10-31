#pragma once

#include "adf/include/data_types/common/types.h"
#include "adf/include/profiler/latency_profiler.h"
#include "adf/include/profiler/profiler_client.h"
#include "cm/include/proto_cm_writer.h"
#include "proto/test/soc/dbg_msg.pb.h"

namespace hozon {
namespace netaos {
namespace adf {

class LatencyProfiler {
public:
    LatencyProfiler(const std::string& instance_name);

    int32_t Init(std::vector<std::string> header_names);
    void Show(const std::vector<double> latencies, bool after_process = false);

private:
    hozon::netaos::cm::ProtoCMWriter<hozon::adf::lite::dbg::LatencyInfo> writer;
    hozon::adf::lite::dbg::LatencyInfo _msg;
    std::string _instance_name;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon