#pragma once

#include <fastdds/dds/topic/TopicDataType.hpp>
#include "adf/include/data_types/image/orin_image.h"
#include "adf/include/node_proto_register.h"
#include "adf/include/node_proxy.h"
#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyIdlCuda : public NodeProxyBase {
   public:
    explicit NodeProxyIdlCuda(const NodeConfig::CommInstanceConfig& config);
    ~NodeProxyIdlCuda();

    void OnDataReceive(void) override;
    void PauseReceive() override;
    void ResumeReceive() override;
    void Deinit() override;

   protected:
    std::shared_ptr<eprosima::fastdds::dds::TopicDataType> _pub_sub_type;
    std::shared_ptr<hozon::netaos::cm::Proxy> _proxy;
    uint32_t _domain;
    std::string _topic;

    cudaStream_t _cuda_stream;
    bool _cuda_memory_init = false;
    std::atomic_bool _initialized;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
