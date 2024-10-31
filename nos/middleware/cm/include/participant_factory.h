#pragma once

#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/DataReaderListener.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/rtps/common/Guid.h>

#include <iostream>
#include <unordered_map>

#include "cm/include/cm_config.h"
#include "cm/include/cm_logger.h"

namespace hozon {
namespace netaos {
namespace cm {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps::rtps;

inline std::string convertGuidToString(const GUID_t &guid) {
    std::stringstream ss;
    ss << guid;
    return ss.str();
}

class DiscoveryDomainParticipantListener : public DomainParticipantListener {
public:
    void on_participant_discovery(
            DomainParticipant* participant,
            eprosima::fastrtps::rtps::ParticipantDiscoveryInfo&& info) override;

    void on_subscriber_discovery(
            DomainParticipant* participant,
            eprosima::fastrtps::rtps::ReaderDiscoveryInfo&& info) override;

    void on_publisher_discovery(
            DomainParticipant* participant,
            eprosima::fastrtps::rtps::WriterDiscoveryInfo&& info) override;

    void on_type_discovery(
            DomainParticipant* participant,
            const eprosima::fastrtps::rtps::SampleIdentity& request_sample_id,
            const eprosima::fastrtps::string_255& topic,
            const eprosima::fastrtps::types::TypeIdentifier* identifier,
            const eprosima::fastrtps::types::TypeObject* object,
            eprosima::fastrtps::types::DynamicType_ptr dyn_type) override;
};

class DdsPubSubInstance {
public:
    static DdsPubSubInstance& GetInstance();
    ~DdsPubSubInstance();

    DomainParticipant* Getparticipant(const uint32_t domain);
    Publisher* GetPublisher(const uint32_t domain);
    Subscriber* GetSubscriber(const uint32_t domain);
    DataReader* GetDataReader(const uint32_t domain, Topic* topic_desc, DataReaderListener* listener, QosMode mode);
    DataWriter* GetDataWriter(const uint32_t domain, Topic* topic_desc, DataWriterListener* listener);
    Topic* GetTopicDescription(const uint32_t domain, const std::string& topic, const std::string& type_name);

    int32_t RegisterTopicType(const uint32_t domain, TypeSupport& type);

    int32_t DeleteWriter(const uint32_t domain, DataWriter* writer);
    int32_t DeleteReader(const uint32_t domain, DataReader* reader);
    int32_t DeleteTopicDesc(const uint32_t domain, Topic* topic_desc);
    int32_t DeleteSublisher(const uint32_t domain);
    int32_t DeletePublisher(const uint32_t domain);
    int32_t DeleteParticpant(const uint32_t domain);

private:
    DdsPubSubInstance(){};

    struct DdsPubSub_t {
        eprosima::fastdds::dds::DomainParticipant* _participant;
        eprosima::fastdds::dds::Publisher* _publisher;
        eprosima::fastdds::dds::Subscriber* _subscriber;
        DiscoveryDomainParticipantListener _listener;
    };

    std::mutex _mutex;
    std::unordered_map<uint32_t, DdsPubSub_t> _pubsub_map;

    CmQosConfig _cm_qos_cfg;
};

}
}
}