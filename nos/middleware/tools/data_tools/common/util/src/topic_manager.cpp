
#include "topic_manager.hpp"
#include <unordered_map>
#include <fastdds/rtps/attributes/RTPSParticipantAttributes.h>
#include "data_tools_logger.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"

namespace hozon {
namespace netaos {
namespace data_tool_common {

TopicManager::~TopicManager() {}

void TopicManager::DeInit() {
    std::lock_guard<std::recursive_mutex> lock(_participant_map_mutex);

    for (auto iter : _participant_subscriber_map) {
        iter.second = nullptr;
    }

    _participant_subscriber_map.clear();

    for (auto iter : _participant_publisher_map) {
        iter.second = nullptr;
    }

    _participant_publisher_map.clear();

    for (auto iter : _participant_map) {
        iter.second = nullptr;
    }

    _participant_map.clear();
    _callbackFunctionList.clear();
    _topicInfoMap.clear();
}

bool TopicManager::Init(bool sd) {

    if (sd) {
        // Create participant for service discovery.
        auto participant_sd = GetParticipant(kDdsDataType_SD, false);
        auto sbuscriber_sd = GetSubscriber(kDdsDataType_SD);

        auto participant_sd_1 = GetParticipant(kDdsDataType_SD_1, false);
        auto sbuscriber_sd_1 = GetSubscriber(kDdsDataType_SD_1);

        if (participant_sd) {
            COMMON_LOG_DEBUG << "Create sd participant success.";
        } else {
            COMMON_LOG_ERROR << "Create sd participant success.";
        }

        if (sbuscriber_sd) {
            COMMON_LOG_DEBUG << "Create sd subscribe success.";
        } else {
            COMMON_LOG_ERROR << "Create sd subscribe success.";
        }

        if (participant_sd_1) {
            COMMON_LOG_DEBUG << "Create sd participant success.";
        } else {
            COMMON_LOG_ERROR << "Create sd participant success.";
        }

        if (sbuscriber_sd_1) {
            COMMON_LOG_DEBUG << "Create sd subscribe success.";
        } else {
            COMMON_LOG_ERROR << "Create sd subscribe success.";
        }
    }
    return true;
}

std::shared_ptr<DomainParticipant> TopicManager::GetParticipant(int32_t dds_data_type, bool direction) {
    std::lock_guard<std::recursive_mutex> lock(_participant_map_mutex);
    if (_participant_map.find(dds_data_type) == _participant_map.end()) {
        auto participant = CreateParticipant(dds_data_type, direction);
        if (participant) {
            _participant_map[dds_data_type] = participant;
        } else {
            COMMON_LOG_ERROR << "Creat participant failed !";
        }
    }
    if (_participant_map.find(dds_data_type) != _participant_map.end()) {
        return _participant_map[dds_data_type];
    }
    return nullptr;
}

std::shared_ptr<eprosima::fastdds::dds::Subscriber> TopicManager::GetSubscriber(int32_t dds_data_type) {
    std::lock_guard<std::recursive_mutex> lock(_participant_map_mutex);
    if (_participant_subscriber_map.find(dds_data_type) != _participant_subscriber_map.end()) {
        return _participant_subscriber_map[dds_data_type];
    }

    if (_participant_map.find(dds_data_type) == _participant_map.end()) {
        return nullptr;
    }

    auto participant = _participant_map[dds_data_type];

    eprosima::fastdds::dds::Subscriber* subscriber = participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT, nullptr);
    if (subscriber) {
        _participant_subscriber_map[dds_data_type] = std::shared_ptr<eprosima::fastdds::dds::Subscriber>(subscriber, [this, dds_data_type](eprosima::fastdds::dds::Subscriber* subscriber) {
            if (subscriber) {
                auto participant = GetParticipant(dds_data_type, false);
                if (participant) {
                    participant->delete_subscriber(subscriber);
                }
            }
        });
        return _participant_subscriber_map[dds_data_type];
    } else {
        COMMON_LOG_ERROR << "Creat publisher failed !";
        return nullptr;
    }
}

std::shared_ptr<eprosima::fastdds::dds::Publisher> TopicManager::GetPublisher(int32_t dds_data_type) {
    std::lock_guard<std::recursive_mutex> lock(_participant_map_mutex);
    if (_participant_publisher_map.find(dds_data_type) != _participant_publisher_map.end()) {
        return _participant_publisher_map[dds_data_type];
    }

    if (_participant_map.find(dds_data_type) == _participant_map.end()) {
        return nullptr;
    }

    auto participant = _participant_map[dds_data_type];

    eprosima::fastdds::dds::Publisher* publisher = participant->create_publisher(PUBLISHER_QOS_DEFAULT, nullptr);
    if (publisher) {
        _participant_publisher_map[dds_data_type] = std::shared_ptr<eprosima::fastdds::dds::Publisher>(publisher, [this, dds_data_type](eprosima::fastdds::dds::Publisher* publisher) {
            if (publisher) {
                auto participant = GetParticipant(dds_data_type, true);
                if (participant) {
                    participant->delete_publisher(publisher);
                }
            }
        });
        return _participant_publisher_map[dds_data_type];
    } else {
        COMMON_LOG_ERROR << "Creat publisher failed !";
        return nullptr;
    }
}

eprosima::fastdds::dds::DataReaderQos TopicManager::GetReaderQos(int32_t dds_data_type, const std::string &topic) {
    eprosima::fastdds::dds::DataReaderQos qos = _cm_qos_config.GetReaderQos(dds_data_type, topic);
    return qos;
}

eprosima::fastdds::dds::DataWriterQos TopicManager::GetWriterQos(int32_t dds_data_type, const std::string &topic) {
    eprosima::fastdds::dds::DataWriterQos qos = _cm_qos_config.GetWriterQos(dds_data_type, topic);
    return qos;
}

int32_t TopicManager::GetTopicDataType(std::string topic_name) {
    if (topic_name == "/soc/rawpointcloud") {
        return kDdsDataType_LidarRaw;
    }
    if (topic_name == "/soc/pointcloud") {
        return kDdsDataType_Lidar;
    }
    if (topic_name.find("/soc/encoded_camera") != std::string::npos) {
        return kDdsDataType_CameraH265;
    }
    if (topic_name.find("/soc/camera") != std::string::npos) {
        return kDdsDataType_CameraYuv;
    }
    if (topic_name.find("/soc/zerocopy/camera") != std::string::npos) {
        return kDdsDataType_CameraYuv;
    }
    return kDdsDataType_Normal;
}

uint32_t TopicManager::GetDomainId(int32_t dds_data_type) {
    uint32_t domain_id = 0;
    switch (dds_data_type) {
        case kDdsDataType_SD:
            break;
        case kDdsDataType_SD_1: {
            domain_id = 1;
        } break;
        case kDdsDataType_Normal:
            break;
        case kDdsDataType_Lidar: {
            domain_id = 1;
        } break;
        case kDdsDataType_LidarRaw: {
            domain_id = 1;
        } break;
        case kDdsDataType_CameraYuv:
        case kDdsDataType_CameraH265:
            break;
        default:
            break;
    }

    return domain_id;
}

std::shared_ptr<DomainParticipant> TopicManager::CreateParticipant(int32_t dds_data_type, bool direction) {

    std::shared_ptr<DomainParticipant> participant_sptr;

    uint32_t domain_id = GetDomainId(dds_data_type);
    DomainParticipantQos pqos = GetParticipantQos(dds_data_type, direction);
    DomainParticipantListener* listener = ((dds_data_type == kDdsDataType_SD) || (dds_data_type == kDdsDataType_SD_1)) ? &_partListener : nullptr;

    StatusMask par_mask = StatusMask::subscription_matched() << StatusMask::data_available();
    DomainParticipant* participant = DomainParticipantFactory::get_instance()->create_participant(domain_id, pqos, listener, par_mask);
    if (participant == nullptr) {
        COMMON_LOG_ERROR << " creat RTPSParticipant failed !";
    } else {
        participant_sptr = std::shared_ptr<DomainParticipant>(participant, [](DomainParticipant* participant) {
            if (participant) {
                eprosima::fastrtps::types::ReturnCode_t ret = DomainParticipantFactory::get_instance()->delete_participant(participant);
                if (ret != eprosima::fastrtps::types::ReturnCode_t::RETCODE_OK) {
                    COMMON_LOG_ERROR << "TopicManager delete_participant failed.";
                }
                participant = nullptr;
            }
        });
    }

    return participant_sptr;
}

DomainParticipantQos TopicManager::GetParticipantQos(int32_t dds_data_type, bool direction) {
    int ret = _cm_qos_config.GetQosFromEnv("CONF_PREFIX_PATH", dds_data_type, "topic_monitor_qos.json");
    if (ret < 0) {
        COMMON_LOG_ERROR << "read qos config failed";
    }

    // common qos config
    DomainParticipantQos pqos = _cm_qos_config.GetParticipantQos(dds_data_type);

    // unique qos config
    switch (dds_data_type) {
        case kDdsDataType_SD: {
            //服务发现
            pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
            // pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
            // pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = false;
            pqos.wire_protocol().builtin.typelookup_config.use_client = true;
            pqos.wire_protocol().builtin.typelookup_config.use_server = false;
        } break;
        case kDdsDataType_SD_1: {
            //服务发现
            pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
            // pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
            // pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = false;
            pqos.wire_protocol().builtin.typelookup_config.use_client = true;
            pqos.wire_protocol().builtin.typelookup_config.use_server = false;
        } break;
        case kDdsDataType_Normal: {
            //小数据direction=true， 发送
            pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
            pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = direction ? false : true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = direction ? true : false;
            pqos.wire_protocol().builtin.typelookup_config.use_client = direction ? false : true;
            pqos.wire_protocol().builtin.typelookup_config.use_server = direction ? true : false;
        } break;
        case kDdsDataType_Lidar: {
            //小数据direction=true， 发送，雷达，size不一样
            pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
            pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = direction ? false : true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = direction ? true : false;
            pqos.wire_protocol().builtin.typelookup_config.use_client = direction ? false : true;
            pqos.wire_protocol().builtin.typelookup_config.use_server = direction ? true : false;
        } break;
        case kDdsDataType_LidarRaw: {
            pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
            pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = direction ? false : true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = direction ? true : false;
            pqos.wire_protocol().builtin.typelookup_config.use_client = direction ? false : true;
            pqos.wire_protocol().builtin.typelookup_config.use_server = direction ? true : false;
        } break;
        case kDdsDataType_CameraYuv: {
            pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
            pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = direction ? false : true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = direction ? true : false;
            pqos.wire_protocol().builtin.typelookup_config.use_client = direction ? false : true;
            pqos.wire_protocol().builtin.typelookup_config.use_server = direction ? true : false;
        } break;
        case kDdsDataType_CameraH265: {
            pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
            pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = direction ? false : true;
            pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = direction ? true : false;
            pqos.wire_protocol().builtin.typelookup_config.use_client = direction ? false : true;
            pqos.wire_protocol().builtin.typelookup_config.use_server = direction ? true : false;
        } break;
        default:
            COMMON_LOG_CRITICAL << "Unknow participant";
            break;
    }

    return pqos;
}

std::map<std::string, TopicInfo> TopicManager::GetTopicInfo() {
    std::lock_guard<std::recursive_mutex> lk(_topicInfoMutex);
    return _topicInfoMap;
}

void TopicManager::AddTopicInfo(TopicInfo topicInfo) {
    std::lock_guard<std::recursive_mutex> lk(_topicInfoMutex);
    _topicInfoMap[topicInfo.topicName] = topicInfo;
}

void TopicManager::RegistNewTopicCallback(std::function<void(TopicInfo topicInfo)> callback) {
    _callbackFunctionList.push_back(callback);
};

void TopicManager::TopicManagerSubListener::on_type_information_received(DomainParticipant* participant, const string_255 topic_name, const string_255 type_name,
                                                                         const types::TypeInformation& type_information) {
    COMMON_LOG_DEBUG << "on_type_information_received find: topic_name= " << topic_name << ", type_name=" << type_name;
    TopicInfo topicInfo;
    topicInfo.topicName = std::string(topic_name);
    topicInfo.typeName = std::string(type_name);
    topicInfo.change_type = CHANGE_TOPIC;
    topicInfo.operate_type = OPT_JOIN;
    topicInfo.role_type = ROLE_READER;
    _topic_manager->AddTopicInfo(topicInfo);
    for (size_t i = 0; i < _topic_manager->_callbackFunctionList.size(); i++) {
        _topic_manager->_callbackFunctionList[i](topicInfo);
    }
}

}  // namespace data_tool_common
}  //namespace netaos
}  //namespace hozon