// Copyright 2016 Proyectos y Sistemas de Mantenimiento SL (eProsima).
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
 * @file BagPublisher.cpp
 *
 */

#include "BagPublisher.h"
#include "BagTypeObject.h"
#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/PublisherAttributes.h>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/qos/PublisherQos.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>



#include <thread>

using namespace eprosima::fastdds::dds;

BagPublisher::BagPublisher()
    : participant_(nullptr)
    , publisher_(nullptr)
    , topic_(nullptr)
    , writer_(nullptr)
    , type_(std::make_shared<BagPubSubType>())
{
}

bool BagPublisher::init( std::string::topic_name, std::string type_name, bool use_env)
{
    std::cout << "BagPublisher::init." << std::endl;
    // hello_.index(0);
    // hello_.message("Bag");
    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;
    pqos.wire_protocol().builtin.typelookup_config.use_server = true;
    pqos.name("Participant_pub");
    auto factory = DomainParticipantFactory::get_instance();

    if (use_env)
    {
        factory->load_profiles();
        factory->get_default_participant_qos(pqos);
    }

    participant_ = factory->create_participant(0, pqos);

    if (participant_ == nullptr)
    {
        return false;
    }

    //REGISTER THE TYPE
    type_.get()->auto_fill_type_information(true);
    type_.register_type(participant_);

    //CREATE THE PUBLISHER
    PublisherQos pubqos = PUBLISHER_QOS_DEFAULT;

    if (use_env)
    {
        participant_->get_default_publisher_qos(pubqos);
    }

    publisher_ = participant_->create_publisher(
        pubqos,
        nullptr);

    if (publisher_ == nullptr)
    {
        return false;
    }

    //CREATE THE TOPIC
    TopicQos tqos = TOPIC_QOS_DEFAULT;

    if (use_env)
    {
        participant_->get_default_topic_qos(tqos);
    }

    topic_ = participant_->create_topic(
        type_name,
        type_name,
        tqos);

    if (topic_ == nullptr)
    {
        return false;
    }
    std::cout << "init topic ok." << std::endl;
    // CREATE THE WRITER
    DataWriterQos wqos = DATAWRITER_QOS_DEFAULT;

    if (use_env)
    {
        publisher_->get_default_datawriter_qos(wqos);
    }
    std::cout << "init create_datawriter." << std::endl;
    writer_ = publisher_->create_datawriter(
        topic_,
        wqos,
        &listener_);
    // std::cout << "init create_datawriter2." << std::endl;
    // if (writer_ == nullptr)
    // {
    //     return false;
    // }
    // std::cout << "init  ok." << std::endl;
    return true;
}

BagPublisher::~BagPublisher()
{
    if (writer_ != nullptr)
    {
        publisher_->delete_datawriter(writer_);
    }
    if (publisher_ != nullptr)
    {
        participant_->delete_publisher(publisher_);
    }
    if (topic_ != nullptr)
    {
        participant_->delete_topic(topic_);
    }
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
}

void BagPublisher::PubListener::on_publication_matched(
        eprosima::fastdds::dds::DataWriter*,
        const eprosima::fastdds::dds::PublicationMatchedStatus& info)
{
    if (info.current_count_change == 1)
    {
        matched_ = info.total_count;
        firstConnected_ = true;
        std::cout << "Publisher matched." << std::endl;
    }
    else if (info.current_count_change == -1)
    {
        matched_ = info.total_count;
        std::cout << "Publisher unmatched." << std::endl;
    }
    else
    {
        std::cout << info.current_count_change
                  << " is not a valid value for PublicationMatchedStatus current count change" << std::endl;
    }
}

void BagPublisher::runThread(
        uint32_t samples,
        uint32_t sleep, //FILE* fid
        rosbag2_cpp::Reader* reader_)
{

    std::cout << "has_next1=" << reader_->has_next() << std::endl;
    if (samples == 0)
    {
        while (!stop_)
        {
            if (publish(reader_, false))
            {
                // std::cout << "Message: " << hello_.message() << " with index: " << hello_.index()
                //           << " SENT" << std::endl;
                std::cout  << "send Message: " << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
        }
    }
    else
    {
        for (uint32_t i = 0; i < samples; ++i)
        {
            if (!publish(reader_))
            {
                --i;
            }
            else
            {
                // std::cout << "Message: " << hello_.message() << " with index: " << hello_.index()
                //           << " SENT" << std::endl;
                std::cout  << "send Message: " << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
        }
    }
}

void BagPublisher::run(
        uint32_t samples,
        uint32_t sleep)
{

    std::unique_ptr<rosbag2_cpp::Reader> reader_ = std::make_unique<rosbag2_cpp::Reader>();
    reader_->open("test_data/mcap_bag");
    std::cout << "reader_->open(test_data/mcap_bag) ok" << std::endl;
    std::cout << "has_next=" <<  reader_->has_next() << std::endl;
    // reader_->


    //打开文件
    FILE* fid = fopen("/home/sw/work/source/Fast-DDS/Fast-DDS/examples/cpp/dds/TestSendSeriDataExample/test_data/binary.dat","rb");
    // if(fid == NULL)
    // {
    //     printf("打开文件出错");
    //     return;
    // }

    std::cout << "run start." << std::endl;
    stop_ = false;
    //std::thread thread(&BagPublisher::runThread, this, samples, sleep, fid);
    std::thread thread(&BagPublisher::runThread, this, samples, sleep, reader_.get());
    if (samples == 0)
    {
        std::cout << "Publisher running. Please press enter to stop the Publisher at any time." << std::endl;
        std::cin.ignore();
        stop_ = true;
    }
    else
    {
        std::cout << "Publisher running " << samples << " samples." << std::endl;
    }
    thread.join();
    if(fid != NULL)
    {
        fclose(fid);
    }
    reader_->close();
    std::cout << "run ok." << std::endl;
}

bool BagPublisher::publish(
        rosbag2_cpp::Reader* reader_,//FILE* fid,
        bool waitForListener)
{

    if (listener_.firstConnected_ || !waitForListener || listener_.matched_ > 0)
    {
        std::cout << "BagPublisher::publish start.." << std::endl;
        std::cout << "has_next2=" <<  reader_->has_next() << std::endl;
        TestWorld data;


        std::shared_ptr<rosbag2_storage::SerializedBagMessage> rosbag_message = reader_->read_next();
        std::cout << "read_next->read_next()ok" << std::endl;
        data.m_payload.length = rosbag_message->serialized_data->buffer_length;
        data.m_payload.reserve(data.m_payload.length);


        std::memcpy(
            data.m_payload.data,
            rosbag_message->serialized_data->buffer,
            data.m_payload.length);

        // hello_.index(hello_.index() + 1);
        //writer_->write(&hello_);
        //eprosima::fastrtps::rtps::SerializedPayload_t payload;

        // std::cout << "publish DataRead_CMode start" << std::endl;
        //将文件中的位置指针重新定位到文件开头
        //rewind(fp);
        // std::cout << "开始读取序列化数据:" << std::endl;
        // fread(&(data.m_payload.encapsulation), sizeof(uint16_t) ,1 , fid);
        // fread(&(data.m_payload.length), sizeof(uint32_t), 1 ,fid);
        // std::cout << "data.m_payload.length = " << data.m_payload.length << std::endl;
        // data.m_payload.reserve(data.m_payload.length); // 创建缓存大小
        // fread(data.m_payload.data, sizeof(unsigned char), data.m_payload.length ,fid);
        // fread(&(data.m_payload.max_size), sizeof(uint32_t), 1 ,fid);
        // fread(&(data.m_payload.pos), sizeof(uint32_t), 1 ,fid);

        writer_->write(&data);
        return true;
    }
    return false;
}

// bool BagPublisher::publish(
//         rosbag2_cpp::Reader* reader_,//FILE* fid,
//         bool waitForListener)
// {

//     if (listener_.firstConnected_ || !waitForListener || listener_.matched_ > 0)
//     {
//         std::cout << "BagPublisher::publish start.." << std::endl;
//         std::cout << "has_next2=" <<  reader_->has_next() << std::endl;
//         TestWorld data;


//         std::shared_ptr<rosbag2_storage::SerializedBagMessage> rosbag_message = reader_->read_next();
//         std::cout << "read_next->read_next()ok" << std::endl;
//         data.m_payload.length = rosbag_message->serialized_data->buffer_length;
//         data.m_payload.reserve(data.m_payload.length);


//         std::memcpy(
//             data.m_payload.data,
//             rosbag_message->serialized_data->buffer,
//             data.m_payload.length);

//         // hello_.index(hello_.index() + 1);
//         //writer_->write(&hello_);
//         //eprosima::fastrtps::rtps::SerializedPayload_t payload;

//         // std::cout << "publish DataRead_CMode start" << std::endl;
//         //将文件中的位置指针重新定位到文件开头
//         //rewind(fp);
//         // std::cout << "开始读取序列化数据:" << std::endl;
//         // fread(&(data.m_payload.encapsulation), sizeof(uint16_t) ,1 , fid);
//         // fread(&(data.m_payload.length), sizeof(uint32_t), 1 ,fid);
//         // std::cout << "data.m_payload.length = " << data.m_payload.length << std::endl;
//         // data.m_payload.reserve(data.m_payload.length); // 创建缓存大小
//         // fread(data.m_payload.data, sizeof(unsigned char), data.m_payload.length ,fid);
//         // fread(&(data.m_payload.max_size), sizeof(uint32_t), 1 ,fid);
//         // fread(&(data.m_payload.pos), sizeof(uint32_t), 1 ,fid);

//         writer_->write(&data);
//         return true;
//     }
// }