#include "cm/include/participant_factory.h"
#include "cm/include/cm_logger.h"
#include "idl/generated/cm_protobufTypeObject.h"

namespace hozon {
namespace netaos {
namespace cm {

DdsPubSubInstance& DdsPubSubInstance::GetInstance() {
    static DdsPubSubInstance instance;
    return instance;
}

DdsPubSubInstance::~DdsPubSubInstance() {}

DomainParticipant* DdsPubSubInstance::Getparticipant(const uint32_t domain) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._participant == nullptr) {
        int ret = _cm_qos_cfg.GetQosFromEnv(QOS_CONFIG_PATH, domain);
        if (ret < 0) {
            CM_LOG_ERROR << "[domain: " << domain << "] Init qos config failed.";
        }
        DomainParticipantQos pqos = _cm_qos_cfg.GetParticipantQos(domain);
        StatusMask sub_mask = StatusMask::all() >> StatusMask::data_on_readers();
        _pubsub_map[domain]._participant = DomainParticipantFactory::get_instance()->create_participant(domain, pqos, &_pubsub_map[domain]._listener, sub_mask);
        if (_pubsub_map[domain]._participant == nullptr) {
            return nullptr;
        }
    }

    registercm_protobufTypes();
    return _pubsub_map[domain]._participant;
}

Publisher* DdsPubSubInstance::GetPublisher(const uint32_t domain) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._publisher == nullptr) {
        _pubsub_map[domain]._publisher = _pubsub_map[domain]._participant->create_publisher(PUBLISHER_QOS_DEFAULT, nullptr);
        if (_pubsub_map[domain]._publisher == nullptr) {
            return nullptr;
        }
    }

    return _pubsub_map[domain]._publisher;
}

Subscriber* DdsPubSubInstance::GetSubscriber(const uint32_t domain) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._subscriber == nullptr) {
        _pubsub_map[domain]._subscriber = _pubsub_map[domain]._participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT, nullptr);
        if (_pubsub_map[domain]._subscriber == nullptr) {
            return nullptr;
        }
    }
    return _pubsub_map[domain]._subscriber;
}

int32_t DdsPubSubInstance::RegisterTopicType(const uint32_t domain, TypeSupport& type_) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._participant == nullptr) {
        return -1;
    }

    type_.get()->auto_fill_type_information(true);
    type_.get()->auto_fill_type_object(false);
    return (type_.register_type(_pubsub_map[domain]._participant)());
}

Topic* DdsPubSubInstance::GetTopicDescription(const uint32_t domain, const std::string& topic, const std::string& type_name) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._participant == nullptr) {
        return nullptr;
    }

    Topic* topic_desc = static_cast<Topic*>(_pubsub_map[domain]._participant->lookup_topicdescription(topic));
    if (topic_desc != nullptr) {
        return topic_desc;
    }

    return _pubsub_map[domain]._participant->create_topic(topic, type_name, TOPIC_QOS_DEFAULT);
}

DataWriter* DdsPubSubInstance::GetDataWriter(const uint32_t domain, Topic* topic_, DataWriterListener* listener) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._publisher == nullptr) {
        return nullptr;
    }
    DataWriterQos wqos = _cm_qos_cfg.GetWriterQos(domain, topic_->get_name());
    return _pubsub_map[domain]._publisher->create_datawriter(topic_, wqos, listener);
}

DataReader* DdsPubSubInstance::GetDataReader(const uint32_t domain, Topic* topic_, DataReaderListener* listener, QosMode mode) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._subscriber == nullptr) {
        return nullptr;
    }
    DataReaderQos rqos = _cm_qos_cfg.GetReaderQos(domain, topic_->get_name(), mode);
    return _pubsub_map[domain]._subscriber->create_datareader(topic_, rqos, listener);
}

int32_t DdsPubSubInstance::DeleteSublisher(const uint32_t domain) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._subscriber == nullptr) {
        return -1;
    }

    if (_pubsub_map[domain]._subscriber->has_datareaders() == false) {
        _pubsub_map[domain]._participant->delete_subscriber(_pubsub_map[domain]._subscriber);
        _pubsub_map[domain]._subscriber = nullptr;
    }

    return 0;
}

int32_t DdsPubSubInstance::DeletePublisher(const uint32_t domain) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._publisher == nullptr) {
        return -1;
    }

    if (_pubsub_map[domain]._publisher->has_datawriters() == false) {
        _pubsub_map[domain]._participant->delete_publisher(_pubsub_map[domain]._publisher);
        _pubsub_map[domain]._publisher = nullptr;
    }

    return 0;
}

int32_t DdsPubSubInstance::DeleteParticpant(const uint32_t domain) {
    std::lock_guard<std::mutex> lk(_mutex);
    if (_pubsub_map[domain]._participant == nullptr) {
        return -1;
    }

    if (_pubsub_map[domain]._participant->has_active_entities() == false) {
        DomainParticipantFactory::get_instance()->delete_participant(_pubsub_map[domain]._participant);
        _pubsub_map[domain]._participant = nullptr;
    }

    return 0;
}

int32_t DdsPubSubInstance::DeleteWriter(const uint32_t domain, DataWriter* writer) {
    std::lock_guard<std::mutex> lk(_mutex);
    if ((_pubsub_map[domain]._publisher == nullptr) || (writer == nullptr)) {
        return -1;
    }

    return _pubsub_map[domain]._publisher->delete_datawriter(writer)();
}

int32_t DdsPubSubInstance::DeleteReader(const uint32_t domain, DataReader* reader) {
    std::lock_guard<std::mutex> lk(_mutex);
    if ((_pubsub_map[domain]._subscriber == nullptr) || (reader == nullptr)) {
        return -1;
    }

    return _pubsub_map[domain]._subscriber->delete_datareader(reader)();
}

int32_t DdsPubSubInstance::DeleteTopicDesc(const uint32_t domain, Topic* topic_desc) {
    std::lock_guard<std::mutex> lk(_mutex);
    if ((_pubsub_map[domain]._participant == nullptr) || (topic_desc == nullptr)) {
        return -1;
    }

    return _pubsub_map[domain]._participant->delete_topic(topic_desc)();
}

void DiscoveryDomainParticipantListener::on_participant_discovery(DomainParticipant* participant, eprosima::fastrtps::rtps::ParticipantDiscoveryInfo&& info) {
    static_cast<void>(participant);
    switch (info.status) {
        case eprosima::fastrtps::rtps::ParticipantDiscoveryInfo::DISCOVERED_PARTICIPANT:
            /* Process the case when a new DomainParticipant was found in the domain */
            CM_LOG_INFO << "New DomainParticipant '" << convertGuidToString(info.info.m_guid) << "' discovered.";
            break;
        case eprosima::fastrtps::rtps::ParticipantDiscoveryInfo::CHANGED_QOS_PARTICIPANT:
            /* Process the case when a DomainParticipant changed its QOS */
            break;
        case eprosima::fastrtps::rtps::ParticipantDiscoveryInfo::REMOVED_PARTICIPANT:
            /* Process the case when a DomainParticipant was removed from the domain */
            CM_LOG_INFO << "New DomainParticipant '" << convertGuidToString(info.info.m_guid) << "' left the domain.";
            break;
        default:
            break;
    }
}

void DiscoveryDomainParticipantListener::on_subscriber_discovery(DomainParticipant* participant, eprosima::fastrtps::rtps::ReaderDiscoveryInfo&& info) {
    static_cast<void>(participant);
    switch (info.status) {
        case eprosima::fastrtps::rtps::ReaderDiscoveryInfo::DISCOVERED_READER:
            /* Process the case when a new subscriber was found in the domain */
            CM_LOG_TRACE << "New DataReader subscribed to topic '" << info.info.topicName() << "' of type '" << info.info.typeName() << "' discovered";
            break;
        case eprosima::fastrtps::rtps::ReaderDiscoveryInfo::CHANGED_QOS_READER:
            /* Process the case when a subscriber changed its QOS */
            break;
        case eprosima::fastrtps::rtps::ReaderDiscoveryInfo::REMOVED_READER:
            /* Process the case when a subscriber was removed from the domain */
            CM_LOG_TRACE << "New DataReader subscribed to topic '" << info.info.topicName() << "' of type '" << info.info.typeName() << "' left the domain.";
            break;
        default:
            break;
    }
}

void DiscoveryDomainParticipantListener::on_publisher_discovery(DomainParticipant* participant, eprosima::fastrtps::rtps::WriterDiscoveryInfo&& info) {
    static_cast<void>(participant);
    switch (info.status) {
        case eprosima::fastrtps::rtps::WriterDiscoveryInfo::DISCOVERED_WRITER:
            /* Process the case when a new publisher was found in the domain */
            CM_LOG_TRACE << "New DataWriter publishing under topic '" << info.info.topicName() << "' of type '" << info.info.typeName() << "' discovered";
            break;
        case eprosima::fastrtps::rtps::WriterDiscoveryInfo::CHANGED_QOS_WRITER:
            /* Process the case when a publisher changed its QOS */
            break;
        case eprosima::fastrtps::rtps::WriterDiscoveryInfo::REMOVED_WRITER:
            /* Process the case when a publisher was removed from the domain */
            CM_LOG_TRACE << "New DataWriter publishing under topic '" << info.info.topicName() << "' of type '" << info.info.typeName() << "' left the domain.";
            break;
        default:
            break;
    }
}

void DiscoveryDomainParticipantListener::on_type_discovery(DomainParticipant* participant, const eprosima::fastrtps::rtps::SampleIdentity& request_sample_id,
                                                           const eprosima::fastrtps::string_255& topic, const eprosima::fastrtps::types::TypeIdentifier* identifier,
                                                           const eprosima::fastrtps::types::TypeObject* object, eprosima::fastrtps::types::DynamicType_ptr dyn_type) {
    static_cast<void>(participant);
    static_cast<void>(request_sample_id);
    static_cast<void>(topic);
    static_cast<void>(identifier);
    static_cast<void>(object);
    static_cast<void>(dyn_type);
    CM_LOG_INFO << "New data type of topic '" << topic << "' discovered.";
}

}  // namespace cm
}  // namespace netaos
}  // namespace hozon