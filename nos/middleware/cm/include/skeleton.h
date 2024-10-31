#pragma once

#include <fastdds/dds/topic/TopicDataType.hpp>

namespace hozon {
namespace netaos {
namespace cm {

class SkeletonImpl;
class Skeleton {
public:
    Skeleton(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type);
    Skeleton(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type, uint8_t qos_mode);
    ~Skeleton();

    int32_t Init(const uint32_t domain, const std::string& topic);
    void Deinit();

    int32_t Write(std::shared_ptr<void> data);
    bool IsMatched();

    using OnServiceFindCallback = std::function<void(void)>;
    void RegisterServiceListen(OnServiceFindCallback callback);

private:
    std::shared_ptr<SkeletonImpl> _pimpl;

};

}
}    
}