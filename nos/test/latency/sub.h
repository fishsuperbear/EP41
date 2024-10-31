#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include "dds_helper.h"

using CB_TYPE = std::function<void()>;

class SubListener : public eprosima::fastdds::dds::DataReaderListener {
public:
    SubListener(CB_TYPE cb) {
        _cb = cb;
    }

    void on_data_available(eprosima::fastdds::dds::DataReader* reader) {
        _cb();
    }

private:
    CB_TYPE _cb;
};

class Subscriber {
public:
    int32_t Init(const std::string& topic, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> topic_type, CB_TYPE on_available) {
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

        eprosima::fastdds::dds::DataReaderQos reader_qos;
        reader_qos.endpoint().history_memory_policy = eprosima::fastrtps::rtps::PREALLOCATED_WITH_REALLOC_MEMORY_MODE;
        reader_qos.history().depth = 5;
        reader_qos.reliability().kind = eprosima::fastdds::dds::BEST_EFFORT_RELIABILITY_QOS;
        reader_qos.durability().kind = eprosima::fastdds::dds::VOLATILE_DURABILITY_QOS;
        reader_qos.data_sharing().on("");

        _listener = std::make_shared<SubListener>(on_available);
        _reader = DDSHelper::GetInstance()._subscriber->create_datareader(
                        _topic, 
                        reader_qos,
                        _listener.get());
        if (_reader == nullptr) {
            LAT_LOG_ERROR << "Fail to create data reader of [" << topic << "].";
            return -1;
        }

        // LAT_LOG_INFO << "Create subscriber of topic " << topic;
        return 0;
    }

    int32_t Take(void* data) {
        // if (_reader->wait_for_unread_message(eprosima::fastrtps::Duration_t(2, 0))) {
            
        //     eprosima::fastdds::dds::SampleInfo info;
        //     _reader->take_next_sample(data, &info);
            
        //     return 0;
        // }

        eprosima::fastdds::dds::SampleInfo info;
        eprosima::fastrtps::types::ReturnCode_t ret = _reader->take_next_sample(data, &info);
        if (ret() != eprosima::fastrtps::types::ReturnCode_t::ReturnCodeValue::RETCODE_OK) {
            LAT_LOG_ERROR << "Fail to recv data " << ret();
            return -1;
        }
        return 0;
    }

    void Stop() {
        _reader->close();
    }

private:
    std::string _topic_name;
    eprosima::fastdds::dds::Topic* _topic;
    eprosima::fastdds::dds::DataReader* _reader;
    std::shared_ptr<SubListener> _listener;
};