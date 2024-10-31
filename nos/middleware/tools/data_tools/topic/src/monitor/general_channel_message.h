#ifndef TOOLS_CVT_MONITOR_GENERAL_CHANNEL_MESSAGE_H_
#define TOOLS_CVT_MONITOR_GENERAL_CHANNEL_MESSAGE_H_

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/topic/Topic.hpp>
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/cm_protobufTypeObject.h"
#include "monitor/cyber_topology_message.h"
#include "monitor/general_message_base.h"
#include "topic_manager.hpp"

namespace hozon {
namespace netaos {
namespace topic {

using namespace hozon::netaos::data_tool_common;

class CyberTopologyMessage;
class GeneralMessage;

class GeneralChannelMessage : public GeneralMessageBase {
   public:
    enum class ErrorCode { TopicManagerFailed = -1, CreateReaderFailed = -2, TopicAlreadyExist = -3, CreatTopicFailed = -4, NewSubClassFailed = -5 };

    static const char* ErrCode2Str(ErrorCode errCode);
    static bool IsErrorCode(void* ptr);

    static ErrorCode CastPtr2ErrorCode(void* ptr) {
        assert(IsErrorCode(ptr));
        return static_cast<ErrorCode>(reinterpret_cast<intptr_t>(ptr));
    }

    static GeneralChannelMessage* CastErrorCode2Ptr(ErrorCode errCode) { return reinterpret_cast<GeneralChannelMessage*>(static_cast<intptr_t>(errCode)); }

    ~GeneralChannelMessage() {
        if (nullptr != reader_) {
            // if (TopicManager::getInstance().getSubsciber()->is_enabled()) {
            //     std::cout << "getSubsciber is enable" << std::endl;
            // }
            int32_t dds_data_type = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetTopicDataType(topic_name_);
            dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetSubscriber(dds_data_type)->delete_datareader(reader_);
            reader_ = nullptr;
        }
        if (nullptr != topic_) {
            int32_t dds_data_type = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetTopicDataType(topic_name_);
            dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetParticipant(dds_data_type, false)->delete_topic(topic_);
            topic_ = nullptr;
        }
        if (raw_msg_class_) {
            delete raw_msg_class_;
            raw_msg_class_ = nullptr;
        }
    }

    void set_topic_name(const std::string& topicName) { topic_name_ = topicName; }

    const std::string& topic_name(void) const { return topic_name_; }

    void set_message_type(const std::string& msgTypeName) { message_type_ = msgTypeName; }

    const std::string& proto_type(void) const { return proto_name_; }

    bool is_enabled(void) const { return nullptr == reader_ ? false : true; }

    bool has_message_come(void) const { return has_message_come_; }

    double frame_ratio(void) override;

    void add_reader(eprosima::fastdds::dds::DataReader* reader) { reader_ = reader; }

    void del_reader() {
        if (nullptr != reader_) {
            int32_t dds_data_type = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetTopicDataType(topic_name_);
            dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetSubscriber(dds_data_type)->delete_datareader(reader_);
            reader_ = nullptr;
        }
        if (nullptr != topic_) {
            int32_t dds_data_type = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetTopicDataType(topic_name_);
            dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetParticipant(dds_data_type, false)->delete_topic(topic_);
            topic_ = nullptr;
        }
    }

    int Render(const Screen* s, int key) override;

    void CloseChannel(void) {
        int32_t dds_data_type = dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetTopicDataType(topic_name_);
        dynamic_cast<CyberTopologyMessage*>(parent())->topic_manager_->GetSubscriber(dds_data_type)->delete_datareader(reader_);
        reader_ = nullptr;
    }

   private:
    explicit GeneralChannelMessage(RenderableMessage* parent = nullptr)
        : GeneralMessageBase(parent),
          current_state_(State::ShowDebugString),
          has_message_come_(false),
          frame_counter_(0),
          last_time_(std::chrono::steady_clock::now()),
          msg_time_(last_time_ + std::chrono::nanoseconds(1)),
          inner_lock_(),
          raw_msg_class_(nullptr),
          topic_name_(),
          message_type_(),
          proto_name_(),
          topic_(nullptr),
          reader_(nullptr),
          reader_listener_(this) {}

    GeneralChannelMessage(const GeneralChannelMessage&) = delete;
    GeneralChannelMessage& operator=(const GeneralChannelMessage&) = delete;

    std::vector<char> CopyMsg(void) const {
        std::lock_guard<std::mutex> g(inner_lock_);
        return topic_message_;
    }

    GeneralChannelMessage* OpenChannel(const std::string& topic_name);

    void RenderDebugString(const Screen* s, int key, int* line_no);

    void set_has_message_come(bool b) { has_message_come_ = b; }

    enum class State { ShowDebugString, ShowInfo } current_state_;

    bool has_message_come_;

    std::atomic<int> frame_counter_;

    std::chrono::steady_clock::time_point last_time_;
    std::chrono::steady_clock::time_point msg_time_;
    std::chrono::steady_clock::time_point time_last_calc_ = std::chrono::steady_clock::now();
    mutable std::mutex inner_lock_;

    google::protobuf::Message* raw_msg_class_;

    //rtps
    std::string topic_name_;
    std::string message_type_;
    std::string proto_name_;
    eprosima::fastdds::dds::Topic* topic_;
    eprosima::fastdds::dds::DataReader* reader_;
    std::vector<char> topic_message_;

    //rtps
    // eprosima::fastdds::dds::DomainParticipant* _participant;

    //fast dds
    class MyListener : public eprosima::fastdds::dds::DataReaderListener {
       public:
        MyListener(GeneralChannelMessage* parents) : parents_(parents) {}

        ~MyListener() {}

        virtual void on_data_available(eprosima::fastdds::dds::DataReader* reader) override;
        void on_subscription_matched(DataReader* reader, const SubscriptionMatchedStatus& info) override;

        GeneralChannelMessage* parents_;
    } reader_listener_;
    //fast dds

    friend class CyberTopologyMessage;
    friend class GeneralMessage;
};  // GeneralChannelMessage

}  // namespace topic
}  //namespace netaos
}  //namespace hozon

#endif  // TOOLS_CVT_MONITOR_GENERAL_CHANNEL_MESSAGE_H_
