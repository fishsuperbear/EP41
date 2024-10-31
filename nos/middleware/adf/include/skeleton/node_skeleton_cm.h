#pragma once
#include "adf/include/node_skeleton.h"

namespace hozon {
namespace netaos {
namespace adf {
class NodeSkeletonCM : public NodeSkeletonBase {
   public:
    explicit NodeSkeletonCM(const NodeConfig::CommInstanceConfig& config,
                            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type);
    virtual ~NodeSkeletonCM();
    virtual void OnDataNeedSend() override;
    void Deinit() override;

   protected:
    std::shared_ptr<IDLBaseType> CreateIDLPtrFromBaseData(BaseDataTypePtr base_ptr);

    std::shared_ptr<hozon::netaos::cm::Skeleton> _skeleton;
    std::string _topic;
};
}  // namespace adf
}  // namespace netaos
}  // namespace hozon