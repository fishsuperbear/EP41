#include "adf/include/skeleton/node_skeleton_proto.h"
#include "adf/include/internal_log.h"
#include "idl/generated/cm_protobuf.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeSkeletonProto::NodeSkeletonProto(const NodeConfig::CommInstanceConfig& config,
                                     std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type)
    : NodeSkeletonCM(config, pub_sub_type) {}

NodeSkeletonProto::~NodeSkeletonProto() {}

void NodeSkeletonProto::OnDataNeedSend() {
    BaseDataTypePtr base_ptr = std::static_pointer_cast<BaseData>(*(_container.PopFront()));

    std::shared_ptr<CmProtoBuf> idl_data(new CmProtoBuf);
    idl_data->name(base_ptr->proto_msg->GetTypeName());
    std::string serialized_data;
    base_ptr->proto_msg->SerializeToString(&serialized_data);
    idl_data->str().assign(serialized_data.begin(), serialized_data.end());

    if (_skeleton->Write(idl_data)) {
        ADF_LOG_ERROR << "Write " << _topic << " data fail.";
    } else {
        ADF_LOG_TRACE << "Write " << _topic << " data success.";
    }
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon