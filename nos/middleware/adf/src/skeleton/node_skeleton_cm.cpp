#include "adf/include/skeleton/node_skeleton_cm.h"
#include "adf/include/internal_log.h"

namespace hozon {
namespace netaos {
namespace adf {

NodeSkeletonCM::NodeSkeletonCM(const NodeConfig::CommInstanceConfig& config,
                               std::shared_ptr<eprosima::fastdds::dds::TopicDataType> pub_sub_type)
    : NodeSkeletonBase(config) {
    _topic = config.topic;
    _skeleton = std::make_shared<cm::Skeleton>(pub_sub_type);
    _skeleton->Init(_config.domain, _config.topic);
}

NodeSkeletonCM::~NodeSkeletonCM() {}

std::shared_ptr<IDLBaseType> NodeSkeletonCM::CreateIDLPtrFromBaseData(BaseDataTypePtr base_ptr) {
    std::shared_ptr<IDLBaseType> idl_data = std::static_pointer_cast<IDLBaseType>(base_ptr->idl_msg);

    for (auto data : base_ptr->__header.latency_info.data) {
        LinkInfo link_info;
        link_info.link_name(data.first);
        link_info.timestamp_real().sec(data.second.sec);
        link_info.timestamp_real().nsec(data.second.nsec);
        idl_data->header().latency_info().link_infos().emplace_back(link_info);
    }

    return idl_data;
}

void NodeSkeletonCM::OnDataNeedSend() {
    BaseDataTypePtr base_ptr = std::static_pointer_cast<BaseData>(*(_container.PopFront()));

    std::shared_ptr<IDLBaseType> idl_data = CreateIDLPtrFromBaseData(base_ptr);

    if (_skeleton->Write(idl_data)) {
        ADF_LOG_ERROR << "Write " << _topic << " data fail.";
    } else {
        ADF_LOG_TRACE << "Write " << _topic << " data success.";
    }
}

void NodeSkeletonCM::Deinit() {
    ADF_LOG_DEBUG << "CM Skeleton deinit.";
    _skeleton->Deinit();
}

}  // namespace adf
}  // namespace netaos
}  // namespace hozon
