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
 * @file RecorderSubscriber.cpp
 *
 */

#include "RecorderSubscriber.h"
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/SubscriberAttributes.h>

#include <fastrtps/types/DynamicDataFactory.h>
#include <fastrtps/types/DynamicDataHelper.hpp>
#include <fastrtps/types/TypeObjectFactory.h>

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps;
using namespace eprosima::fastrtps::rtps;
using SerializedPayload_t = eprosima::fastrtps::rtps::SerializedPayload_t;

RecorderSubscriber::RecorderSubscriber() : mp_participant(nullptr), mp_subscriber(nullptr), m_listener(this) {}

bool RecorderSubscriber::init() {
    DomainParticipantQos pqos;
    pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationReaderANDSubscriptionWriter = true;
    pqos.wire_protocol().builtin.discovery_config.m_simpleEDP.use_PublicationWriterANDSubscriptionReader = true;
    pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = SIMPLE;
    pqos.wire_protocol().builtin.discovery_config.use_SIMPLE_EndpointDiscoveryProtocol = true;
    pqos.wire_protocol().builtin.typelookup_config.use_client = true;
    pqos.wire_protocol().builtin.use_WriterLivelinessProtocol = false;
    pqos.wire_protocol().builtin.discovery_config.leaseDuration = c_TimeInfinite;
    pqos.name("Participant_sub");

    //Do not enable entities on creation
    DomainParticipantFactoryQos factory_qos;
    factory_qos.entity_factory().autoenable_created_entities = false;
    DomainParticipantFactory::get_instance()->set_qos(factory_qos);

    StatusMask par_mask = StatusMask::subscription_matched() << StatusMask::data_available();
    mp_participant = DomainParticipantFactory::get_instance()->create_participant(0, pqos, &m_listener, par_mask);
    if (mp_participant == nullptr) {
        return false;
    }
    if (mp_participant->enable() != ReturnCode_t::RETCODE_OK) {
        DomainParticipantFactory::get_instance()->delete_participant(mp_participant);
        return false;
    }

    // CREATE THE COMMON READER ATTRIBUTES
    qos_ = DATAREADER_QOS_DEFAULT;
    qos_.durability().kind = eprosima::fastdds::dds::TRANSIENT_LOCAL_DURABILITY_QOS;
    qos_.reliability().kind = eprosima::fastdds::dds::RELIABLE_RELIABILITY_QOS;
    qos_.history().kind = eprosima::fastdds::dds::KEEP_LAST_HISTORY_QOS;
    qos_.history().depth = 30;
    qos_.resource_limits().max_samples = 50;
    qos_.resource_limits().allocated_samples = 20;

    return true;
}

RecorderSubscriber::~RecorderSubscriber() {
    std::cout << "RecorderSubscriber::~RecorderSubscriber() start 1" << std::endl;
    m_listener.~SubListener();
    for (const auto& it : topics_) {
        mp_subscriber->delete_datareader(it.first);
        mp_participant->delete_topic(it.second);
    }
    if (mp_subscriber != nullptr) {
        mp_participant->delete_subscriber(mp_subscriber);
    }

    DomainParticipantFactory::get_instance()->delete_participant(mp_participant);
    topics_.clear();
    readers_.clear();
    datas_.clear();
    std::cout << "RecorderSubscriber::~RecorderSubscriber() end" << std::endl;
}

void RecorderSubscriber::SubListener::on_subscription_matched(eprosima::fastdds::dds::DataReader* reader, const eprosima::fastdds::dds::SubscriptionMatchedStatus& info) {
    if (info.current_count_change == 1) {
        n_matched++;
        std::cout << "Subscriber matched" << std::endl;
    } else if (info.current_count_change == -1) {
        n_matched--;
        std::cout << "Subscriber unmatched" << std::endl;
        auto itr = subscriber_->readers_.find(reader);
        if (itr != subscriber_->readers_.end()) {
            subscriber_->readers_.erase(itr);
        }

        auto itd = subscriber_->datas_.find(reader);
        if (itd != subscriber_->datas_.end()) {
            subscriber_->datas_.erase(itd);
        }

        if (subscriber_->mp_subscriber != nullptr) {
            subscriber_->mp_subscriber->delete_datareader(reader);
        }
    } else {
        std::cout << "Subscriber received an invalid value for SubscriptionMatchedStatus." << std::endl;
    }
}

//采用C模式写二进制文件
// void RecorderSubscriber::SubListener::DataWrite_CMode(SerializedPayload_t* payload)
// {
//     std::cout << "开始记录序列化数据:" << std::endl;
//     //准备数据
//     // double pos[200];
//     // for(int i = 0; i < 200; i ++ )
//     //     pos[i] = i ;
//     //写出数据DataWrite_CMode
//     std::cout << "DataWrite_CMode start" << std::endl;

//     if(file == NULL)
//     {
//         printf("写出文件出错");
//         return;
//     }
//     fwrite(&(payload->encapsulation), sizeof(uint16_t), 1 , file);
//     fwrite(&(payload->length), sizeof(uint32_t), 1 , file);
//     fwrite(payload->data, sizeof(unsigned char), payload->length , file);
//     fwrite(&(payload->max_size), sizeof(uint32_t), 1 , file);
//     fwrite(&(payload->pos), sizeof(uint32_t), 1 , file);
//     std::cout << "记录序完成" << std::endl;
// }

void RecorderSubscriber::SubListener::on_data_available(eprosima::fastdds::dds::DataReader* reader) {
    auto dit = subscriber_->datas_.find(reader);

    if (dit != subscriber_->datas_.end()) {
        types::DynamicData_ptr data = dit->second;
        SampleInfo info;
        if (reader->take_next_sample(data.get(), &info) == ReturnCode_t::RETCODE_OK) {
            if (info.valid_data) {
                types::DynamicType_ptr type = subscriber_->readers_[reader];
                this->n_samples++;
                std::cout << "*Received data of type " << type->get_name() << std::endl;
                types::DynamicDataHelper::print(data);
                //开始记录数据

                TypeSupport m_type(std::make_shared<types::DynamicPubSubType>(type));
                auto func = m_type->getSerializedSizeProvider(data.get());
                uint32_t serLength = func();

                eprosima::fastrtps::rtps::SerializedPayload_t m_payload;
                m_payload.reserve(serLength);
                m_type->serialize(data.get(), &m_payload);

                eprosima::fastrtps::Time_t time_stamp;
                eprosima::fastrtps::Time_t::now(time_stamp);

                std::string topicName = subscriber_->topics_[reader]->get_name();

                std::cout << "Record message ! topic_name=" << topicName << "; type_name=" << type->get_name() << "; time_stamp=" << time_stamp << std::endl;
                writer_->write(m_payload, topicName, type->get_name(), time_stamp.to_ns());
                //DataWrite_CMode(&m_payload); //需用锁

                //eprosima::fastrtps::types::DynamicPubSubType* subtype = new eprosima::fastrtps::types::DynamicPubSubType(type);
            }
        }
    }
}

void RecorderSubscriber::SubListener::on_type_information_received(eprosima::fastdds::dds::DomainParticipant*, const eprosima::fastrtps::string_255 topic_name,
                                                                   const eprosima::fastrtps::string_255 type_name, const eprosima::fastrtps::types::TypeInformation& type_information) {
    std::cout << "topic_name=" << topic_name << "; type_name=" << type_name << std::endl;
    std::function<void(const std::string&, const types::DynamicType_ptr)> callback = [this, topic_name](const std::string& name, const types::DynamicType_ptr type) {
        std::cout << "Discovered type: " << name << " from topic " << topic_name << std::endl;

        if (subscriber_->mp_subscriber == nullptr) {
            //SubscriberAttributes Rparam;
            //Rparam = subscriber_->att_;
            //Rparam.topic = subscriber_->topic_;
            //Rparam.topic.topicName = topic_name;
            //Rparam.qos = subscriber_->qos_;
            subscriber_->mp_subscriber = subscriber_->mp_participant->create_subscriber(SUBSCRIBER_QOS_DEFAULT, nullptr);

            if (subscriber_->mp_subscriber == nullptr) {
                return;
            }
        }

        //CREATE THE TOPIC
        eprosima::fastdds::dds::Topic* topic = subscriber_->mp_participant->create_topic("HelloWorldTopic", name, TOPIC_QOS_DEFAULT);

        if (topic == nullptr) {
            return;
        }

        StatusMask sub_mask = StatusMask::subscription_matched() << StatusMask::data_available();
        DataReader* reader = subscriber_->mp_subscriber->create_datareader(topic, subscriber_->qos_, &subscriber_->m_listener, sub_mask);

        if (type == nullptr) {
            const types::TypeIdentifier* ident = types::TypeObjectFactory::get_instance()->get_type_identifier_trying_complete(name);

            if (nullptr != ident) {
                const types::TypeObject* obj = types::TypeObjectFactory::get_instance()->get_type_object(ident);

                types::DynamicType_ptr dyn_type = types::TypeObjectFactory::get_instance()->build_dynamic_type(name, ident, obj);

                if (nullptr != dyn_type) {
                    subscriber_->readers_[reader] = dyn_type;
                    types::DynamicData_ptr data(types::DynamicDataFactory::get_instance()->create_data(dyn_type));
                    subscriber_->datas_[reader] = data;
                } else {
                    std::cout << "ERROR: DynamicType cannot be created for type: " << name << std::endl;
                }
            } else {
                std::cout << "ERROR: TypeIdentifier cannot be retrieved for type: " << name << std::endl;
            }
        } else {
            subscriber_->topics_[reader] = topic;
            subscriber_->readers_[reader] = type;
            types::DynamicData_ptr data(types::DynamicDataFactory::get_instance()->create_data(type));
            subscriber_->datas_[reader] = data;
        }
    };

    subscriber_->mp_participant->register_remote_type(type_information, type_name.to_string(), callback);
}

void RecorderSubscriber::SubListener::on_type_discovery(eprosima::fastdds::dds::DomainParticipant* participant, const eprosima::fastrtps::rtps::SampleIdentity& request_sample_id,
                                                        const eprosima::fastrtps::string_255& topic, const eprosima::fastrtps::types::TypeIdentifier* identifier,
                                                        const eprosima::fastrtps::types::TypeObject* object, eprosima::fastrtps::types::DynamicType_ptr dyn_type) {
    //std::cout << "on_type_discovery: participant= " << participant-> << std::endl;
    std::cout << "on_type_discovery: topic= " << topic << std::endl;
    if (dyn_type != nullptr) {
        std::cout << "on_type_discovery: dyn_type= " << dyn_type->get_name() << std::endl;
    } else {
        std::cout << "dyn_type is null!" << std::endl;
    }

    //std::cout << "on_type_discovery: topic= " << topic << std::endl;
}

void RecorderSubscriber::run() {
    std::cout << "Subscriber running. Please press enter to stop the Subscriber" << std::endl;
    std::cin.ignore();
}

void RecorderSubscriber::run(uint32_t number) {
    std::cout << "Subscriber running until " << number << " samples have been received" << std::endl;
    while (number > this->m_listener.n_samples) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
