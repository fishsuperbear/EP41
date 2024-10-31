#pragma once

#include "adf/include/skeleton/node_skeleton_cm.h"

namespace hozon {
namespace netaos {
namespace adf {

class NodeSkeletonProto : public NodeSkeletonCM {
   public:
    explicit NodeSkeletonProto(const NodeConfig::CommInstanceConfig& config,
                               std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type);
    virtual ~NodeSkeletonProto();

    virtual void OnDataNeedSend() override;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon