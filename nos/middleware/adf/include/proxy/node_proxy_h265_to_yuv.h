#pragma once

#include "adf/include/internal_log.h"
#include "adf/include/proxy/node_proxy_proto.h"
#include "codec/include/decoder_factory.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyH265ToYUV : public NodeProxyProto {
   public:
    explicit NodeProxyH265ToYUV(const NodeConfig::CommInstanceConfig& config,
                                std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type);
    ~NodeProxyH265ToYUV();

    void OnDataReceive(void) override;

   private:
    std::unique_ptr<codec::Decoder> _decoder;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
