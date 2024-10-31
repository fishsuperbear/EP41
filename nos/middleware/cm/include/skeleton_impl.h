#pragma once

#include "cm/include/participant_factory.h"

namespace hozon {
namespace netaos {
namespace cm {

class SkeletonImpl {
public:
    SkeletonImpl(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type);
    SkeletonImpl(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type, uint8_t qos_mode);
    ~SkeletonImpl();

    int32_t Init(const uint32_t domain, const std::string& topic);
    void Deinit();

    int32_t Write(std::shared_ptr<void> data);
    bool IsMatched();

    using OnServiceFindCallback = std::function<void(void)>;
    void RegisterServiceListen(OnServiceFindCallback callback);

private:
    void LogWriterQosInfo(eprosima::fastdds::dds::DataWriter* writer);
    void LogStatisticInfo(void);

    eprosima::fastdds::dds::DataWriter* _writer;
    eprosima::fastdds::dds::Topic* _topic_desc;
    eprosima::fastdds::dds::TypeSupport _type;

    class PubListener : public eprosima::fastdds::dds::DataWriterListener
    {
    public:

        PubListener() = default;

        ~PubListener() override = default;

        void on_publication_matched(
                eprosima::fastdds::dds::DataWriter* writer,
                const eprosima::fastdds::dds::PublicationMatchedStatus& info) override;

        int matched = 0;

        OnServiceFindCallback service_find_callback;

        uint32_t _domain;
        std::string _topic;
    }
    _listener;

    uint32_t _domain;
    std::string _topic;
    QosMode _qos_mode = NO_MODE;

    struct SkeletonStatisticInfo {
        uint64_t from_user;
        uint64_t to_protocol;
        uint64_t to_protocol_bytes;
    };

    SkeletonStatisticInfo _statistic_info = {0};
};

}
}    
}