

#include "dds_recorder_subscriber.h"
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include "adf_lite_utile.h"
#include "bag_data_pubsubtype.h"
#include "cm/include/cm_config.h"
#include "data_tools_logger.hpp"

namespace hozon {
namespace netaos {
namespace bag {

DDSRecorderSubscriber::DDSRecorderSubscriber() {}

DDSRecorderSubscriber::~DDSRecorderSubscriber() {}

bool DDSRecorderSubscriber::subscrib(std::string topicName, std::string dataType) {
    if (!_method) {
        //不录制method topic
        if (topicName.find("/request/") == 0 || topicName.find("/reply/") == 0) {
            return true;
        }
    }

    std::lock_guard<std::mutex> lk(_subMutex);
    if (IsLiteMethodCMDTopic(topicName)) {
        BAG_LOG_DEBUG << "adf-lite info topic: " << topicName << ", will not be subscrib.";
        return true;
    }
    auto it = _topic_state_map.find(topicName);
    if (it != _topic_state_map.end()) {
        if (TopicState::DISCONNECTED == _topic_state_map[topicName]) {
            _topic_state_map[topicName] = TopicState::CONNECTED;
            SendSubscriptionMatchedEvent(topicName, TopicState::CONNECTED);
        }
        BAG_LOG_DEBUG << topicName << " alrerady be subcribed.";
        return true;
    }

    int32_t dds_data_type = topic_manager_->GetTopicDataType(topicName);
    auto participant = topic_manager_->GetParticipant(dds_data_type, false);
    auto subscriber = topic_manager_->GetSubscriber(dds_data_type);

    //REGISTER THE TYPE
    eprosima::fastdds::dds::TypeSupport type_support = participant->find_type(dataType);
    if (type_support && (type_support.get_type_name() == dataType)) {
        BAG_LOG_WARN << dataType << " is already registered.";
    } else {
        std::shared_ptr<HelloWorldPubSubType> ptr = std::make_shared<HelloWorldPubSubType>();
        ptr->setName(dataType.c_str());
        eprosima::fastdds::dds::TypeSupport type(ptr);
        type.get()->auto_fill_type_information(true);
        type.get()->auto_fill_type_object(false);
        type.register_type(participant.get());
    }

    TopicInfo topic_info;
    topic_info.topicName = topicName;
    topic_info.typeName = dataType;
    SubBase::Subscribe(topic_info);
    _topic_state_map[topicName] = TopicState::CONNECTED;
    // topics_[reader] = topic_tr;

    SendSubscriptionMatchedEvent(topicName, TopicState::CONNECTED);
    CONCLE_BAG_LOG_INFO << "subscrib topic:" << topicName << " type:" << dataType;
    BAG_LOG_INFO << "subscrib topic:" << topicName << " type:" << dataType;
    return true;
}

bool DDSRecorderSubscriber::subscrib(RecordOptions options) {
    // _targetTopics = topics;

    // if (_participant == nullptr) {
    //     //CREATE PARTICIPANT
    //     DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
    //     pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::SIMPLE;
    //     pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
    //     pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = true;
    //     pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = false;
    //     pqos.wire_protocol().builtin.typelookup_config.use_client = true;
    //     // pqos.wire_protocol().builtin.use_WriterLivelinessProtocol = false;
    //     pqos.wire_protocol().builtin.discovery_config.leaseDuration = {5, 0};

    //     pqos.transport().use_builtin_transports = false;
    //     auto udp_transport = std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();
    //     std::vector<std::string> conf_network_List;
    //     if (0 == hozon::netaos::cm::GetNetworkList(conf_network_List)) {
    //         udp_transport->interfaceWhiteList = conf_network_List;
    //     } else {
    //         BAG_LOG_WARN << "Can't get network list from default_network_list.json, use default list.";
    //         udp_transport->interfaceWhiteList = hozon::netaos::cm::default_network_List;
    //     }
    //     pqos.transport().user_transports.push_back(udp_transport);

    //     auto shm_transport = std::make_shared<eprosima::fastdds::rtps::SharedMemTransportDescriptor>();
    //     shm_transport->segment_size(10 * 1024 * 1024);
    //     pqos.transport().user_transports.push_back(shm_transport);

    //     pqos.transport().send_socket_buffer_size = 10 * 1024 * 1024;
    //     pqos.transport().listen_socket_buffer_size = 10 * 1024 * 1024;

    //     pqos.name("Participant_sub");
    //     auto factory = DomainParticipantFactory::get_instance();
    //     _participant = factory->create_participant(0, pqos);
    //     if (_participant == nullptr) {
    //         BAG_LOG_ERROR << " creat RTPSParticipant failed !";
    //         return false;
    //     }

    //     if (_subscriber == nullptr) {
    //         _subscriber = _participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT, nullptr);
    //         if (_subscriber == nullptr) {
    //             BAG_LOG_ERROR << " creat subscriber failed !";
    //             return false;
    //         }
    //     }
    // }

    _monitor_all = options.record_all;
    _method = options.method;
    _auto_subscribe = false;

    SubBase::Start(options.topics, false);

    std::map<std::string, TopicInfo> topicInfos = topic_manager_->GetTopicInfo();
    if (topicInfos.size() == 0) {
        BAG_LOG_WARN << "no topic exist to be recorded!";
    }

    if (options.topics.size() > 0) {
        //只订阅指定的topic
        for (size_t i = 0; i < options.topics.size(); i++) {
            if (topicInfos.find(options.topics[i]) == topicInfos.end()) {
                CONCLE_BAG_LOG_WARN << "topic:" << options.topics[i] << " does not exist!";
                BAG_LOG_WARN << "topic:" << options.topics[i] << " does not exist!";
            } else {
                subscrib(options.topics[i], topicInfos[options.topics[i]].typeName);
            }
        }
    } else {
        //订阅全部的topic
        //过滤指定的topic
        for (auto& topicInfo : topicInfos) {
            bool is_exclude = false;
            for (auto temp : options.exclude_topics) {
                if (temp == topicInfo.second.topicName) {
                    BAG_LOG_DEBUG << topicInfo.second.topicName << " is exclude.";
                    // CONCLE_BAG_LOG_INFO << topicInfo.second.topicName << " is exclude.";
                    is_exclude = true;
                    break;
                }
            }
            if (!is_exclude) {
                subscrib(topicInfo.second.topicName, topicInfo.second.typeName);
            }
        }
    }
    return true;
}

void DDSRecorderSubscriber::registDataAvailableCallback(std::function<void(BagMessage*)> callback) {
    _callbackFunctionList.push_back(callback);
}

void DDSRecorderSubscriber::registSubscriptionMatchedCallback(std::function<void(const std::string& topic_name, const TopicState& state)> callback) {
    _subscriptionMatchedCallbackList.push_back(callback);
}

void DDSRecorderSubscriber::RegisterNewTopicCallback(std::function<void(TopicInfo topic_info)> callback) {
    _callbacks_new_topic.push_back(callback);
}

void DDSRecorderSubscriber::reset() {
    _isStop = true;
    SubBase::Stop();

    for (const auto& it : topics_) {
        std::string topic_name = it.second->get_name();
        int32_t dds_data_type = topic_manager_->GetTopicDataType(topic_name);
        auto participant = topic_manager_->GetParticipant(dds_data_type, false);
        auto subscriber = topic_manager_->GetSubscriber(dds_data_type);
        subscriber->delete_datareader(it.first);
        participant->delete_topic(it.second);
    }
    // if (_subscriber != nullptr) {
    //     _participant->delete_subscriber(_subscriber);
    //     _subscriber = nullptr;
    // }
    // if (nullptr != _participant) {
    //     eprosima::fastrtps::types::ReturnCode_t ret = DomainParticipantFactory::get_instance()->delete_participant(_participant);
    //     if (ret != eprosima::fastrtps::types::ReturnCode_t::RETCODE_OK) {
    //         BAG_LOG_ERROR << "DDSRecorderSubscriber delete_participant failed.";
    //     }
    // }

    topics_.clear();
    // _reader_topic.clear();
    // TopicManager::getInstance().reset();
}

void DDSRecorderSubscriber::OnDataAvailable(DataReader* reader) {

    // if (_readerRegistered->_reader_topic.find(reader) != _readerRegistered->_reader_topic.end()) {
    if (topics_.find(reader) != topics_.end()) {
        BagMessage* bagMessage = new BagMessage();
        SampleInfo info;
        if (reader->take_next_sample(&bagMessage->data, &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                // 获取当前时间点,转换为纳秒
                auto nanoseconds = std::chrono::time_point_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now());

                bagMessage->topic = topics_[reader]->get_name();
                bagMessage->type = topics_[reader]->get_type_name();
                bagMessage->time = nanoseconds.time_since_epoch().count();
                for (size_t i = 0; i < _callbackFunctionList.size(); i++) {
                    //开始记录数据
                    _callbackFunctionList[i](bagMessage);
                }
            }
        }
    }
}

void DDSRecorderSubscriber::OnSubscribed(TopicInfo topic_info) {}

void DDSRecorderSubscriber::OnSubscriptionMatched(DataReader* reader, const SubscriptionMatchedStatus& info) {

    if (info.current_count == 0) {  //topic退出
        std::string topic_name = topics_[reader]->get_name();
        _topic_state_map[topic_name] = TopicState::DISCONNECTED;
        SendSubscriptionMatchedEvent(topic_name, TopicState::DISCONNECTED);
    }
}

void DDSRecorderSubscriber::SendSubscriptionMatchedEvent(const std::string& topic_name, const TopicState state) {
    for (auto func : _subscriptionMatchedCallbackList) {
        func(topic_name, state);
    }
}

void DDSRecorderSubscriber::OnNewTopic(TopicInfo topic_info) {
    for (auto func : _callbacks_new_topic) {
        func(topic_info);
    }
}

}  // namespace bag
}  //namespace netaos
}  //namespace hozon
