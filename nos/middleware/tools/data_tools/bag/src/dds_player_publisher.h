#pragma once
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include "bag_message.hpp"
#include "topic_manager.hpp"

namespace hozon {
namespace netaos {
namespace bag {

using namespace eprosima::fastdds::dds;

class PlayerImpl;

class DDSPlayerPublisher {
   public:
    DDSPlayerPublisher(std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager) { topic_manager_ = topic_manager; };

    virtual ~DDSPlayerPublisher();
    std::map<std::string, DataWriter*> _topicWriterMap;
    bool prepareWriters(std::map<std::string, std::string> topicTypeMap, const bool play_protomethod);
    void Publish(BagMessage* bagMessage);
    bool InitWriters(std::map<std::string, std::string> topicTypeMap, const bool play_protomethod, bool with_original_h265);
    bool DeinitWriters();
    void SetUpdatePubTimeFlag(bool flag);

   private:
    // DomainParticipant* participant_ = nullptr;
    // Publisher* publisher_ = nullptr;
    std::vector<Topic*> topiclist_;
    bool update_pubtime_ = false;
    bool stopped_ = false;
    bool with_original_h265_ = false;
    // eprosima::fastdds::dds::TypeSupport type_;
    std::shared_ptr<hozon::netaos::data_tool_common::TopicManager> topic_manager_;
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon