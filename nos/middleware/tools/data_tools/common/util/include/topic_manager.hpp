#pragma once
#include <map>
#include <memory>
#include <mutex>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include "cm/include/cm_config.h"

namespace hozon {
namespace netaos {
namespace data_tool_common {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps;

enum ChangeType : int { CHANGE_PARTICIPANT = 1, CHANGE_TOPIC = 2 };

enum OperateType : int { OPT_JOIN = 1, OPT_LEAVE = 2 };

enum RoleType : int { ROLE_WRITER = 1, ROLE_READER = 2, ROLE_SERVER = 3, ROLE_CLIENT = 4, ROLE_PARTICIPANT = 5 };

enum DdsDataType : int32_t {
    kDdsDataType_SD = 0,
    kDdsDataType_SD_1,
    kDdsDataType_Normal,
    kDdsDataType_Lidar,
    kDdsDataType_LidarRaw,
    kDdsDataType_CameraYuv,
    kDdsDataType_CameraH265,
};

struct TopicInfo {
    std::string topicName = "";
    std::string typeName = "";
    //for monitor
    ChangeType change_type = CHANGE_TOPIC;
    OperateType operate_type = OPT_JOIN;
    RoleType role_type = ROLE_READER;
};

class TopicManager {
   public:
    TopicManager() : _partListener(this){};
    TopicManager(const TopicManager&) = delete;
    TopicManager& operator=(const TopicManager&) = delete;

    ~TopicManager();

    bool Init(bool sd = true);
    void DeInit();

    std::map<std::string, TopicInfo> GetTopicInfo();
    void RegistNewTopicCallback(std::function<void(TopicInfo topicInfo)>);
    std::shared_ptr<eprosima::fastdds::dds::DomainParticipant> GetParticipant(int32_t dds_data_type, bool direction);
    std::shared_ptr<eprosima::fastdds::dds::Subscriber> GetSubscriber(int32_t dds_data_type);
    std::shared_ptr<eprosima::fastdds::dds::Publisher> GetPublisher(int32_t dds_data_type);
    eprosima::fastdds::dds::DataReaderQos GetReaderQos(int32_t dds_data_type, const std::string &topic);
    eprosima::fastdds::dds::DataWriterQos GetWriterQos(int32_t dds_data_type, const std::string &topic);
    int32_t GetTopicDataType(std::string topic_name);
    uint32_t GetDomainId(int32_t dds_data_type);

   private:
    std::shared_ptr<DomainParticipant> CreateParticipant(int32_t dds_data_type, bool direction);
    DomainParticipantQos GetParticipantQos(int32_t dds_data_type, bool direction);

    std::map<std::string, TopicInfo> _topicInfoMap;
    std::recursive_mutex _topicInfoMutex;
    std::recursive_mutex _initMutex;

    std::vector<std::function<void(TopicInfo topicInfo)>> _callbackFunctionList;

    std::recursive_mutex _participant_map_mutex;
    std::map<int32_t, std::shared_ptr<DomainParticipant>> _participant_map;
    std::map<int32_t, std::shared_ptr<eprosima::fastdds::dds::Subscriber>> _participant_subscriber_map;
    std::map<int32_t, std::shared_ptr<eprosima::fastdds::dds::Publisher>> _participant_publisher_map;

    class TopicManagerSubListener : public DomainParticipantListener {
       public:
        TopicManagerSubListener(TopicManager* topic_manager) : _topic_manager(topic_manager) {}

        ~TopicManagerSubListener(){};

        virtual void on_type_information_received(DomainParticipant* participant, const string_255 topic_name, const string_255 type_name, const types::TypeInformation& type_information);

       private:
        TopicManager* _topic_manager;

    } _partListener;

    void AddTopicInfo(TopicInfo topicInfo);

    hozon::netaos::cm::CmQosConfig _cm_qos_config;
};

}  // namespace data_tool_common
}  //namespace netaos
}  //namespace hozon