#include "dds_helper.h"
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/DataWriterListener.hpp>

class Publisher {
public:
    int32_t Init(const std::string& topic, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type) {

        eprosima::fastdds::dds::TypeSupport type_support(topic_type);
        ReturnCode_t ret = type_support.register_type(DDSHelper::GetInstance()._participant);
        if (ret != 0) {
            LAT_LOG_ERROR << "Fail to regist type " << type_support.get_type_name();
            return -1;
        }

        _topic_name = topic;
        _topic = DDSHelper::GetInstance()._participant->create_topic(
                    topic, 
                    type_support.get_type_name(), 
                    eprosima::fastdds::dds::TOPIC_QOS_DEFAULT);
        if (_topic == nullptr) {
            LAT_LOG_ERROR << "Fail to create topic [" << topic << "].";
            return -1;
        }

        // create data writer
        eprosima::fastdds::dds::DataWriterQos writer_qos;
        writer_qos.endpoint().history_memory_policy = eprosima::fastrtps::rtps::PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
        writer_qos.history().depth = 1;
        writer_qos.history().kind = eprosima::fastdds::dds::KEEP_LAST_HISTORY_QOS;
        writer_qos.durability().kind = eprosima::fastdds::dds::VOLATILE_DURABILITY_QOS;
        writer_qos.reliability().kind = eprosima::fastdds::dds::BEST_EFFORT_RELIABILITY_QOS;
        writer_qos.data_sharing().on("");
        _writer = DDSHelper::GetInstance()._publisher->create_datawriter(
                        _topic, 
                        writer_qos);
        if (_writer == nullptr) {
            LAT_LOG_ERROR << "Fail to create data writer of [" << topic << "].";
            return -1;
        }

        // LAT_LOG_INFO << "Create publisher of topic " << topic;
        return 0;
    }

    int32_t Write(void* data) {
        bool ret = _writer->write(data);
        if (!ret) {
            LAT_LOG_ERROR << "Fail to send " << _topic_name;
            return -1;
        }

        return 0;
    }

private:
    eprosima::fastdds::dds::Topic* _topic;
    eprosima::fastdds::dds::DataWriter* _writer;
    std::string _topic_name;
};