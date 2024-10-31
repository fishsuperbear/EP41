#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/types/DynamicDataFactory.h>
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/proto_method.h"
#include "topic_manager.hpp"

namespace hozon {
namespace netaos {
namespace topic {

// struct TopicInfo {
//     std::string topicName = "";
//     std::string typeName = "";
// };

using TopicInfo = hozon::netaos::data_tool_common::TopicInfo;

class SubBase {
   public:
    SubBase();
    virtual ~SubBase();
    void Start(std::vector<std::string> topics, bool register_normal_type = true);  //Initialization
    void Stop();

   protected:
    virtual void OnDataAvailable(eprosima::fastdds::dds::DataReader* reader){};
    virtual void OnSubscribed(TopicInfo topic_info){};
    virtual void OnSubscriptionMatched(eprosima::fastdds::dds::DataReader* reader, const eprosima::fastdds::dds::SubscriptionMatchedStatus& info){};
    virtual void OnNewTopic(TopicInfo topic_info);

    std::vector<std::string> _targetTopics;
    std::set<std::string> _subTopics;
    bool _isStop = false;
    std::map<eprosima::fastdds::dds::DataReader*, eprosima::fastdds::dds::Topic*> topics_;
    std::mutex topic_infos_mutex_;
    // std::map<std::string, HzStruct> topic_infos_;

    std::deque<TopicInfo> _newValibleTopic;
    std::mutex _condition_mtx;
    std::condition_variable _cv;

    bool _monitor_all = false;
    bool _method = false;
    bool _auto_subscribe = true;

    class MyListener : public eprosima::fastdds::dds::DataReaderListener {
       public:
        MyListener(SubBase* parent) : parent_(parent) {}

        ~MyListener() override {}

        void on_data_available(eprosima::fastdds::dds::DataReader* reader) override;
        // void on_type_information_received(eprosima::fastdds::dds::DomainParticipant* participant, const eprosima::fastrtps::string_255 topic_name, const eprosima::fastrtps::string_255 type_name,
        //                                   const eprosima::fastrtps::types::TypeInformation& type_information) override;
        void on_subscription_matched(eprosima::fastdds::dds::DataReader* reader, const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;

        SubBase* parent_;

        friend class SubBase;
    } reader_listener_;

    void Subscribe(const TopicInfo& new_topic_info);

    std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager_;

   private:
    void CheckNewTopic();

    std::thread check_topic_thread_;
};

}  // namespace topic
}  // namespace netaos
}  // namespace hozon