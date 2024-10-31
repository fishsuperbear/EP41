// Copyright 2019 Proyectos y Sistemas de Mantenimiento SL (eProsima).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file RecorderSubscriber.h
 *
 */

#ifndef HELLOWORLDSUBSCRIBER_H_
#define HELLOWORLDSUBSCRIBER_H_

#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/subscriber/SampleInfo.h>
#include <fastrtps/rtps/common/Types.h>

#include <fastrtps/types/DynamicPubSubType.h>
#include <fastrtps/types/DynamicTypePtr.h>

#include <fastrtps/attributes/SubscriberAttributes.h>

#include <map>
#include <rosbag2_cpp/writer.hpp>

using SerializedPayload_t = eprosima::fastrtps::rtps::SerializedPayload_t;

class RecorderSubscriber
{
public:

    RecorderSubscriber();

    virtual ~RecorderSubscriber();

    //!Initialize the subscriber
    bool init();

    //!RUN the subscriber
    void run();

    //!Run the subscriber until number samples have been received.
    void run(
            uint32_t number);

private:

    eprosima::fastdds::dds::DomainParticipant* mp_participant;

    eprosima::fastdds::dds::Subscriber* mp_subscriber;

    std::map<eprosima::fastdds::dds::DataReader*, eprosima::fastdds::dds::Topic*> topics_;

    std::map<eprosima::fastdds::dds::DataReader*, eprosima::fastrtps::types::DynamicType_ptr> readers_;

    std::map<eprosima::fastdds::dds::DataReader*, eprosima::fastrtps::types::DynamicData_ptr> datas_;

    eprosima::fastrtps::SubscriberAttributes att_;

    eprosima::fastdds::dds::DataReaderQos qos_;

public:
    friend class SubListener;

    class SubListener
        : public eprosima::fastdds::dds::DomainParticipantListener
    {
    public:

        SubListener(
                RecorderSubscriber* sub)
            : n_matched(0)
            , n_samples(0)
            , subscriber_(sub)
        {
            // file = fopen("binary.dat","wb+");
            // if(file == NULL)
            // {
            //     printf("写出文件出错");
            //     return;
            // }
            writer_ = std::make_unique<rosbag2_cpp::Writer>();
            rosbag2_storage::StorageOptions storage_options;
            storage_options.uri = "my_bag";
            storage_options.storage_id = "mcap";
            rosbag2_cpp::ConverterOptions converter_options{};
            writer_->open(storage_options, converter_options);
        }

        ~SubListener() override
        {
            std::cout << "~SubListener() start" << std::endl;
            writer_->~Writer();
            // if (file != nullptr)
            //     {
            //         fclose(file);
            //     }
        }

        void on_data_available(
                eprosima::fastdds::dds::DataReader* reader) override;

        void on_subscription_matched(
                eprosima::fastdds::dds::DataReader* reader,
                const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) override;

        void on_type_information_received(
                eprosima::fastdds::dds::DomainParticipant* participant,
                const eprosima::fastrtps::string_255 topic_name,
                const eprosima::fastrtps::string_255 type_name,
                const eprosima::fastrtps::types::TypeInformation& type_information) override;

        void on_type_discovery(
                eprosima::fastdds::dds::DomainParticipant* participant,
                const eprosima::fastrtps::rtps::SampleIdentity& request_sample_id,
                const eprosima::fastrtps::string_255& topic,
                const eprosima::fastrtps::types::TypeIdentifier* identifier,
                const eprosima::fastrtps::types::TypeObject* object,
                eprosima::fastrtps::types::DynamicType_ptr dyn_type) override;
        void DataWrite_CMode(SerializedPayload_t* payload); 

        int n_matched;

        uint32_t n_samples;

        RecorderSubscriber* subscriber_;

        std::map<std::string, std::string> topic_type_map_;

        eprosima::fastrtps::types::TypeInformation type_info_;
        // FILE* file;
        std::unique_ptr<rosbag2_cpp::Writer> writer_;
    }
    m_listener;

private:

    eprosima::fastrtps::types::DynamicPubSubType m_type;
};

#endif /* HELLOWORLDSUBSCRIBER_H_ */
