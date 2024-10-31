

#ifndef DDSRecorderSubscriber_H_
#define DDSRecorderSubscriber_H_

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>

#include <condition_variable>
#include <unordered_map>
#include <bag_message.hpp>
#include <fastrtps/types/TypeObjectFactory.h>
#include <topic_manager.hpp>
#include "recorder.h"
#include "sub_base.h"

namespace hozon {
namespace netaos {
namespace bag {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps;
using namespace hozon::netaos::data_tool_common;

enum TopicState { DISCONNECTED = 0, CONNECTED = 1 };

class DDSRecorderSubscriber : public hozon::netaos::topic::SubBase {

   public:
    DDSRecorderSubscriber();
    virtual ~DDSRecorderSubscriber();

    bool subscrib(std::string topicName, std::string type);
    bool subscrib(RecordOptions options);  //Initialization
    void registDataAvailableCallback(std::function<void(BagMessage*)>);
    void registSubscriptionMatchedCallback(std::function<void(const std::string& topic_name, const TopicState& state)>);
    void reset();
    void RegisterNewTopicCallback(std::function<void(TopicInfo topic_info)> callback);

   private:
    // eprosima::fastdds::dds::DomainParticipant* _participant;
    // eprosima::fastdds::dds::Subscriber* _subscriber;
    // std::map<DataReader*, std::pair<std::string, std::string>> _reader_topic;
    // std::map<DataReader*, eprosima::fastdds::dds::Topic*> topics_;
    // std::vector<std::string> _targetTopics;
    std::map<std::string, TopicState> _topic_state_map;
    std::mutex _subMutex;
    std::vector<std::function<void(BagMessage*)>> _callbackFunctionList;
    std::vector<std::function<void(const std::string& topic_name, const TopicState& state)>> _subscriptionMatchedCallbackList;
    std::vector<std::function<void(TopicInfo topic_info)>> _callbacks_new_topic;

    void SendSubscriptionMatchedEvent(const std::string& topic_name, const TopicState state);

   protected:
    virtual void OnDataAvailable(eprosima::fastdds::dds::DataReader* reader);
    virtual void OnSubscribed(TopicInfo topic_info);
    virtual void OnSubscriptionMatched(DataReader* reader, const SubscriptionMatchedStatus& info);
    virtual void OnNewTopic(TopicInfo topic_info);

    friend class RecorderImpl;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon

#endif /* TESTREADER_H_ */