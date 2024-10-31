
#include "sub_base.h"
#include <fcntl.h>
#include <fstream>
#include <unordered_map>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/attributes/SubscriberAttributes.h>
#include <fastrtps/rtps/common/Types.h>
#include <fastrtps/subscriber/SampleInfo.h>
#include <fastrtps/types/DynamicDataFactory.h>
#include <fastrtps/types/DynamicDataHelper.hpp>
#include <fastrtps/types/DynamicPubSubType.h>
#include <fastrtps/types/DynamicTypePtr.h>
#include <fastrtps/types/TypeObjectFactory.h>
#include <google/protobuf/compiler/parser.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/dynamic_message.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include "adf_lite_utile.h"
#include "cm/include/cm_config.h"
#include "data_tools_logger.hpp"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "idl/generated/cm_someipbufPubSubTypes.h"
#include "idl/generated/cm_someipbufTypeObject.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "idl/generated/proto_methodTypeObject.h"
#include "idl/generated/zerocopy_imagePubSubTypes.h"
#include "idl/generated/zerocopy_imageTypeObject.h"
#include "proto_factory.h"
#include "topic_manager.hpp"
#include "proto/test/soc/dbg_msg.pb.h"

#define UNUSED_VAR(var) (void)(var)

namespace hozon {

namespace netaos {
namespace topic {

SubBase::SubBase() : reader_listener_(this) {

    topic_manager_ = std::make_shared<hozon::netaos::data_tool_common::TopicManager>();
}

SubBase::~SubBase() {}

void SubBase::Stop() {
    _isStop = true;

    {
        std::unique_lock<std::mutex> lck(_condition_mtx);
        _cv.notify_all();
    }
    
    if (_auto_subscribe && check_topic_thread_.joinable()) {
        check_topic_thread_.join();
    }

    for (const auto& it : topics_) {
        int32_t dds_data_type = topic_manager_->GetTopicDataType(it.second->get_name());
        auto participant = topic_manager_->GetParticipant(dds_data_type, false);
        auto subscriber = topic_manager_->GetSubscriber(dds_data_type);
        TOPIC_LOG_DEBUG << "delete reader for topic: " << it.second->get_name();
        subscriber->delete_datareader(it.first);
        TOPIC_LOG_DEBUG << "delete topic topic: " << it.second->get_name();
        participant->delete_topic(it.second);
    }
    topics_.clear();

    topic_manager_->DeInit();
}

void SubBase::Start(std::vector<std::string> topics, bool register_normal_type) {
    _targetTopics = topics;

    if (_auto_subscribe) {
        check_topic_thread_ = std::thread(&SubBase::CheckNewTopic, this);
    }

    bool has_lidar = false;
    bool has_lidar_raw = false;
    bool has_camera_h265 = false;
    bool has_camera_yuv = false;

    if (_monitor_all) {
        has_lidar = true;
        has_lidar_raw = true;
        has_camera_h265 = true;
        has_camera_yuv = true;
    } else {
        for (auto& topic : topics) {
            if (topic.find("pointcloud") != std::string::npos) {
                if (topic.find("raw") != std::string::npos) {
                    has_lidar_raw = true;
                } else {
                    has_lidar = true;
                }
                continue;
            }

            if (topic.find("camera") != std::string::npos) {
                if (topic.find("encoded") != std::string::npos) {
                    has_camera_h265 = true;
                } else {
                    has_camera_yuv = true;
                }
                continue;
            }
        }
    }

    TOPIC_LOG_DEBUG << "has_lidar: " << has_lidar << ", has_lidar_raw: " << has_lidar_raw << ", has_camera_h265: " << has_camera_h265 << ", has_camera_yuv: " << has_camera_yuv;
    // Register topic callback in topic manager.
    topic_manager_->RegistNewTopicCallback([this](TopicInfo topic_info) { OnNewTopic(topic_info); });

    // Craete participant for subscribing normal topics.
    TOPIC_LOG_DEBUG << "create default particpant for lidar raw";
    auto default_sub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_Normal, false);
    auto subscriber = topic_manager_->GetSubscriber(hozon::netaos::data_tool_common::kDdsDataType_Normal);

    // Create participant for subscribing lidar subscription.
    std::shared_ptr<eprosima::fastdds::dds::DomainParticipant> lidar_sub_participant;
    std::shared_ptr<eprosima::fastdds::dds::DomainParticipant> lidar_raw_sub_participant;
    if (has_lidar_raw) {
        TOPIC_LOG_DEBUG << "create particpant for lidar raw";
        lidar_raw_sub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_LidarRaw, false);
        auto subscriber = topic_manager_->GetSubscriber(hozon::netaos::data_tool_common::kDdsDataType_LidarRaw);
        UNUSED_VAR(subscriber);
    }
    if (has_lidar) {
        TOPIC_LOG_DEBUG << "create particpant for lidar";
        lidar_sub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_Lidar, false);
        auto subscriber = topic_manager_->GetSubscriber(hozon::netaos::data_tool_common::kDdsDataType_Lidar);
        UNUSED_VAR(subscriber);
    }

    // Create participant for subscribing camera subscription.
    std::shared_ptr<eprosima::fastdds::dds::DomainParticipant> camera_yuv_sub_participant;
    std::shared_ptr<eprosima::fastdds::dds::DomainParticipant> camera_h265_sub_participant;
#ifdef BUILD_FOR_X86
    if (has_camera_yuv) {
        TOPIC_LOG_DEBUG << "create particpant for lidar yuv";
        camera_yuv_sub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_CameraYuv, false);
        auto subscriber = topic_manager_->GetSubscriber(hozon::netaos::data_tool_common::kDdsDataType_CameraYuv);
        UNUSED_VAR(subscriber);
    }
#endif
    if (has_camera_h265) {
        TOPIC_LOG_DEBUG << "create particpant for lidar h265";
        camera_h265_sub_participant = topic_manager_->GetParticipant(hozon::netaos::data_tool_common::kDdsDataType_CameraH265, false);
        auto subscriber = topic_manager_->GetSubscriber(hozon::netaos::data_tool_common::kDdsDataType_CameraH265);
        UNUSED_VAR(subscriber);
    }

    if (register_normal_type) {
        // Register typesupport of cm proto buf on default sub / lidar sub / camera sub participant.
        TOPIC_LOG_DEBUG << "register type for CmProtoBuf.";
        eprosima::fastdds::dds::TypeSupport event_type(std::make_shared<CmProtoBufPubSubType>());
        registercm_protobufTypes();
        event_type.get()->auto_fill_type_information(true);
        event_type.get()->auto_fill_type_object(false);
        if (default_sub_participant) {
            event_type.register_type(default_sub_participant.get());
        }
        if (lidar_sub_participant) {
            event_type.register_type(lidar_sub_participant.get());
        }
        if (lidar_raw_sub_participant) {
            event_type.register_type(lidar_raw_sub_participant.get());
        }
        if (camera_h265_sub_participant) {
            event_type.register_type(camera_h265_sub_participant.get());
        }
        if (camera_yuv_sub_participant) {
            event_type.register_type(camera_yuv_sub_participant.get());
        }

        // Register typesupport of method type
        TOPIC_LOG_DEBUG << "register type for ProtoMethodBase.";
        registerproto_methodTypes();
        eprosima::fastdds::dds::TypeSupport method_type(std::make_shared<ProtoMethodBasePubSubType>());
        method_type.get()->auto_fill_type_information(true);
        method_type.get()->auto_fill_type_object(false);
        method_type.register_type(default_sub_participant.get());

        // Register zerocopy image types
        TOPIC_LOG_DEBUG << "register type for ZeroCopyImgXXX.";
        registerzerocopy_imageTypes();

        eprosima::fastdds::dds::TypeSupport zerocopy_type_8m420(std::make_shared<ZeroCopyImg8M420PubSubType>());
        zerocopy_type_8m420.get()->auto_fill_type_information(true);
        zerocopy_type_8m420.get()->auto_fill_type_object(false);
        if (camera_yuv_sub_participant) {
            zerocopy_type_8m420.register_type(camera_yuv_sub_participant.get());
        }

        eprosima::fastdds::dds::TypeSupport zerocopy_type_2m422(std::make_shared<ZeroCopyImg2M422PubSubType>());
        zerocopy_type_2m422.get()->auto_fill_type_information(true);
        zerocopy_type_2m422.get()->auto_fill_type_object(false);
        if (camera_yuv_sub_participant) {
            zerocopy_type_2m422.register_type(camera_yuv_sub_participant.get());
        }

        eprosima::fastdds::dds::TypeSupport zerocopy_type_3m422(std::make_shared<ZeroCopyImg3M422PubSubType>());
        zerocopy_type_3m422.get()->auto_fill_type_information(true);
        zerocopy_type_3m422.get()->auto_fill_type_object(false);
        if (camera_yuv_sub_participant) {
            zerocopy_type_3m422.register_type(camera_yuv_sub_participant.get());
        }

        //regist someip type
        eprosima::fastdds::dds::TypeSupport someip_type(std::make_shared<CmSomeipBufPubSubType>());
        registercm_someipbufTypes();
        someip_type.get()->auto_fill_type_information(true);
        someip_type.get()->auto_fill_type_object(false);
        if (default_sub_participant) {
            someip_type.register_type(default_sub_participant.get());
        }
    }

    topic_manager_->Init();

    return;
}

void SubBase::CheckNewTopic() {
    //当有新的topic出现时，订阅新的topic
    while (!_isStop) {
        TopicInfo new_topic_info;
        {
            std::unique_lock<std::mutex> lck(_condition_mtx);
            if (!_cv.wait_for(lck, std::chrono::milliseconds(50), [&]() { return ((_newValibleTopic.size() > 0) || (_isStop == true)); })) {
                continue;
            }
            if (_isStop == true) {
                break;
            }
            new_topic_info = _newValibleTopic.front();
            _newValibleTopic.pop_front();
        }
        //订阅
        if ("" != new_topic_info.topicName) {
            if ("CmProtoBuf" != new_topic_info.typeName && "CmSomeipBuf" != new_topic_info.typeName && "ProtoMethodBase" != new_topic_info.typeName && new_topic_info.typeName.find("ZeroCopy") == std::string::npos) {
                TOPIC_LOG_DEBUG << "type: " << new_topic_info.typeName << " is unsupported!";
                continue;
            }

            if (!_method) {
                //不显示method
                if (new_topic_info.topicName.find("/request/") == 0 || new_topic_info.topicName.find("/reply/") == 0) {
                    continue;
                }
            }

            if (hozon::netaos::data_tool_common::IsAdfTopic(new_topic_info.topicName)) {
                //adf topic 不订阅，通过adf-lite info 来获取相关频率信息
                TOPIC_LOG_DEBUG << "adf topic: " << new_topic_info.topicName << "will not be subscribe.";
                continue;
            }

            if (_monitor_all) {
                if (std::find(_subTopics.begin(), _subTopics.end(), new_topic_info.topicName) == _subTopics.end()) {
                    //订阅
                    Subscribe(new_topic_info);
                } else {
                    TOPIC_LOG_DEBUG << "topic: " << new_topic_info.topicName << " type: " << new_topic_info.typeName << " already be subscribed1.";
                }

            } else if (std::find(_targetTopics.begin(), _targetTopics.end(), new_topic_info.topicName) != _targetTopics.end()) {
                if (std::find(_subTopics.begin(), _subTopics.end(), new_topic_info.topicName) == _subTopics.end()) {
                    Subscribe(new_topic_info);
                } else {
                    TOPIC_LOG_DEBUG << "topic: " << new_topic_info.topicName << " type: " << new_topic_info.typeName << " already be subscribed";
                }
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void SubBase::Subscribe(const TopicInfo& new_topic_info) {
    //订阅

    TOPIC_LOG_DEBUG << "start process to scribe topic: " << new_topic_info.topicName;
    int32_t dds_data_type = topic_manager_->GetTopicDataType(new_topic_info.topicName);
    auto participant = topic_manager_->GetParticipant(dds_data_type, false);
    auto subscriber = topic_manager_->GetSubscriber(dds_data_type);

    // TODO: confirm
    // eprosima::fastdds::dds::Topic* topic = participant->find_topic(new_topic_info.topicName, eprosima::fastrtps::Duration_t(0, 5));
    eprosima::fastdds::dds::Topic* topic = nullptr;
    if (nullptr == topic) {
        TOPIC_LOG_DEBUG << "create topic: " << new_topic_info.topicName << ", type: " << new_topic_info.typeName;

        topic = participant->create_topic(new_topic_info.topicName, new_topic_info.typeName, eprosima::fastdds::dds::TOPIC_QOS_DEFAULT);
    }
    if (topic == nullptr) {
        TOPIC_LOG_ERROR << "create_topic errer! topic_name=" << new_topic_info.topicName << "; type_name=" << new_topic_info.typeName;
        return;
    }

    _subTopics.insert(new_topic_info.topicName);
    eprosima::fastdds::dds::DataReaderQos qos = topic_manager_->GetReaderQos(dds_data_type, new_topic_info.topicName);
    eprosima::fastdds::dds::StatusMask sub_mask = eprosima::fastdds::dds::StatusMask::subscription_matched() << eprosima::fastdds::dds::StatusMask::data_available();
    TOPIC_LOG_DEBUG << "create reader: " << new_topic_info.topicName;
    eprosima::fastdds::dds::DataReader* reader = subscriber->create_datareader(topic, qos, &reader_listener_, sub_mask);

    if (nullptr == reader) {
        participant->delete_topic(topic);
        TOPIC_LOG_ERROR << "create datareader failed! topic_name:" << new_topic_info.topicName << "; type_name:" << new_topic_info.typeName;
        return;
    }

    topics_[reader] = topic;
    OnSubscribed(new_topic_info);

    TOPIC_LOG_DEBUG << "topic: " << new_topic_info.topicName << " type: " << new_topic_info.typeName << " is subscribed";
}

void SubBase::OnNewTopic(TopicInfo topic_info) {
    TOPIC_LOG_DEBUG << "found topic: " << topic_info.topicName << " , type: " << topic_info.typeName;
    std::unique_lock<std::mutex> lck(_condition_mtx);
    _newValibleTopic.push_back(topic_info);
    _cv.notify_all();
}

void SubBase::MyListener::on_data_available(eprosima::fastdds::dds::DataReader* reader) {
    if (parent_) {
        parent_->OnDataAvailable(reader);
    }
}

void SubBase::MyListener::on_subscription_matched(eprosima::fastdds::dds::DataReader* reader, const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) {
    if (parent_) {
        TOPIC_LOG_DEBUG << "subscription matched: " << reader->get_topicdescription()->get_name();
        parent_->OnSubscriptionMatched(reader, info);
    }
}

}  // namespace topic
}  //namespace netaos
}  //namespace hozon
