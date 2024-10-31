#pragma once

#include "adf/include/node_proto_register.h"
#include "adf/include/proxy/node_proxy_cm.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeProxyProto : public NodeProxyCM {
   public:
    explicit NodeProxyProto(const NodeConfig::CommInstanceConfig& config,
                            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type);
    ~NodeProxyProto();

    void OnDataReceive(void) override;

   protected:
    BaseDataTypePtr CreateBaseDataFromProto(std::shared_ptr<google::protobuf::Message> msg);
    void ParseProtoHeader(std::shared_ptr<google::protobuf::Message> proto_msg, Header& header);
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
