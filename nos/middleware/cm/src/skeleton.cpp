#include "cm/include/skeleton.h"
#include "cm/include/skeleton_impl.h"

namespace hozon {
namespace netaos {
namespace cm {

Skeleton::Skeleton(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type)
    : _pimpl(std::make_shared<SkeletonImpl>(topic_type)) {
}

Skeleton::Skeleton(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type, uint8_t qos_mode)
    : _pimpl(std::make_shared<SkeletonImpl>(topic_type, qos_mode)) {
}

Skeleton::~Skeleton() {

}

int32_t Skeleton::Init(const uint32_t domain, const std::string& topic) {
    return _pimpl->Init(domain, topic);
}

void Skeleton::Deinit() {
    _pimpl->Deinit();
}

int32_t Skeleton::Write(std::shared_ptr<void> data) {
    return _pimpl->Write(data);
}

void Skeleton::RegisterServiceListen(OnServiceFindCallback callback) {
    _pimpl->RegisterServiceListen(callback);
}

bool Skeleton::IsMatched() {
    return _pimpl->IsMatched();
}

}
}
}