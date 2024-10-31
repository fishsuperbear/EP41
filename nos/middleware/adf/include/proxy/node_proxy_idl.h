#pragma once

#include <fastdds/dds/topic/TopicDataType.hpp>
#include "adf/include/node_proto_register.h"
#include "adf/include/node_proxy.h"
#include "cm/include/proxy.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyIdl : public NodeProxyBase {
   public:
    explicit NodeProxyIdl(const NodeConfig::CommInstanceConfig& config);
    ~NodeProxyIdl();

    void OnDataReceive(void) override;
    void PauseReceive() override;
    void ResumeReceive() override;
    void Deinit() override;

   protected:
    std::shared_ptr<eprosima::fastdds::dds::TopicDataType> _pub_sub_type;
    std::shared_ptr<hozon::netaos::cm::Proxy> _proxy;
    uint32_t _domain;
    std::string _topic;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
