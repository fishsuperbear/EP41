#include "adf/include/profiler/latency_profiler.h"

namespace hozon {
namespace netaos {
namespace adf {

LatencyProfiler::LatencyProfiler(const std::string& instance_name) :
    _instance_name(instance_name) {
}

int32_t LatencyProfiler::Init(std::vector<std::string> header_names) {
    int32_t ret = writer.Init(0, "latency_information");
    if (ret < 0) {
        std::cout << "Fail to init writer " << ret << std::endl;
        return -1;
    }
    _msg.set_instance_name(_instance_name);
    for (auto link_name: header_names) {
        _msg.add_link_names(link_name);
    }
    return 0;
}

void LatencyProfiler::Show(const std::vector<double> latencies, bool after_process) {
    _msg.clear_latencies();
    for (double latency: latencies) {
        _msg.add_latencies(latency);
    }
    _msg.set_after_process(after_process);

    int32_t ret = writer.Write(_msg);
    if (ret < 0) {
        std::cout  << "Fail to write " << ret << std::endl;
    }

    std::cout  << "send topic workresult_pb" << ret << std::endl;
}
}  // namespace adf
}  // namespace netaos
}  // namespace hozon