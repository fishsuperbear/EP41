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
 * @file HelloWorldSubscriber.cpp
 *
 */

#include "HelloWorldSubscriber.h"
#include <unordered_map>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/SampleInfo.hpp>
#include <fastdds/dds/subscriber/Subscriber.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/SubscriberAttributes.h>
#include "cm/include/cm_config.h"

using namespace eprosima::fastdds::dds;

template <typename T>
void PrintPoint(T* point) {
    std::cout << "x = " << point->x() << std::endl;
    std::cout << "y = " << point->y() << std::endl;
    std::cout << "z = " << point->z() << std::endl;
}

template <typename T>
void PrintTime(T* time) {
    std::cout << "time->sec() = " << time->sec() << std::endl;
    std::cout << "time->nsec() = " << time->nsec() << std::endl;
}

HelloWorldSubscriber::HelloWorldSubscriber() : participant_(nullptr), subscriber_(nullptr), topic_(nullptr), reader_(nullptr), type_(std::make_shared<ObjectVecPubSubType>()) {}

bool HelloWorldSubscriber::init(bool use_env) {
    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;

    pqos.transport().use_builtin_transports = false;
    auto udp_transport = std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();

    udp_transport->interfaceWhiteList = hozon::netaos::cm::default_network_List;
    pqos.transport().user_transports.push_back(udp_transport);
    pqos.wire_protocol().builtin.typelookup_config.use_client = true;
    pqos.name("Participant_sub");
    auto factory = DomainParticipantFactory::get_instance();

    if (use_env) {
        factory->load_profiles();
        factory->get_default_participant_qos(pqos);
    }

    participant_ = factory->create_participant(0, pqos);

    if (participant_ == nullptr) {
        return false;
    }

    //REGISTER THE TYPE
    type_.get()->auto_fill_type_information(true);
    type_.get()->auto_fill_type_object(false);
    type_.register_type(participant_);

    //CREATE THE SUBSCRIBER
    SubscriberQos sqos = SUBSCRIBER_QOS_DEFAULT;

    if (use_env) {
        participant_->get_default_subscriber_qos(sqos);
    }

    subscriber_ = participant_->create_subscriber(sqos, nullptr);

    if (subscriber_ == nullptr) {
        return false;
    }

    //CREATE THE TOPIC
    TopicQos tqos = TOPIC_QOS_DEFAULT;

    if (use_env) {
        participant_->get_default_topic_qos(tqos);
    }

    topic_ = participant_->create_topic("ComplexIdlTestingTopic", "ObjectVec", tqos);

    if (topic_ == nullptr) {
        return false;
    }

    // CREATE THE READER
    DataReaderQos rqos = DATAREADER_QOS_DEFAULT;
    rqos.reliability().kind = RELIABLE_RELIABILITY_QOS;

    if (use_env) {
        subscriber_->get_default_datareader_qos(rqos);
    }

    reader_ = subscriber_->create_datareader(topic_, rqos, &listener_);

    if (reader_ == nullptr) {
        return false;
    }

    return true;
}

HelloWorldSubscriber::~HelloWorldSubscriber() {
    if (reader_ != nullptr) {
        subscriber_->delete_datareader(reader_);
    }
    if (topic_ != nullptr) {
        participant_->delete_topic(topic_);
    }
    if (subscriber_ != nullptr) {
        participant_->delete_subscriber(subscriber_);
    }
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
}

void HelloWorldSubscriber::SubListener::on_subscription_matched(DataReader*, const SubscriptionMatchedStatus& info) {
    if (info.current_count_change == 1) {
        matched_ = info.total_count;
        std::cout << "Subscriber matched." << std::endl;
    } else if (info.current_count_change == -1) {
        matched_ = info.total_count;
        std::cout << "Subscriber unmatched." << std::endl;
    } else {
        std::cout << info.current_count_change << " is not a valid value for SubscriptionMatchedStatus current count change" << std::endl;
    }
}

void HelloWorldSubscriber::SubListener::on_data_available(DataReader* reader) {
    std::cout << "on_data_available start" << std::endl;
    SampleInfo info;
    if (reader->take_next_sample(&objvec_, &info) == ReturnCode_t::RETCODE_OK) {
        if (info.instance_state == ALIVE_INSTANCE_STATE) {
            samples_++;
            // Print your structure data here.
            // std::cout << "Message " << hello_.message() << " " << hello_.index() << " RECEIVED" << std::endl;

            for (int num = 0; num < 20; ++num) {
                auto obj = &(objvec_.obj().at(num));

                std::cout << "header().seq = " << obj->header().seq() << std::endl;
                std::cout << "header().data_timestamp_us = " << obj->header().data_timestamp_us() << std::endl;
                std::cout << "header().send_timestamp_us = " << obj->header().send_timestamp_us() << std::endl;

                std::cout << "type = " << obj->objectid() << std::endl;
                std::cout << "type = " << obj->type() << std::endl;
                std::cout << "detectsensor_current = " << obj->detectsensor_current() << std::endl;
                std::cout << "detectsensor_history = " << obj->detectsensor_history() << std::endl;
                std::cout << "maintenancestatus = " << obj->maintenancestatus() << std::endl;
                std::cout << "typeconfidence = " << obj->typeconfidence() << std::endl;
                std::cout << "existenceprobability = " << obj->existenceprobability() << std::endl;

                PrintPoint(&(obj->rectinfo().center()));
                PrintPoint(&(obj->rectinfo().centerstddev()));
                PrintPoint(&(obj->rectinfo().sizelwh()));
                PrintPoint(&(obj->rectinfo().sizestddev()));
                std::cout << "rectinfo().orientation = " << obj->rectinfo().orientation() << std::endl;
                std::cout << "rectinfo().orientationstddev = " << obj->rectinfo().orientationstddev() << std::endl;
                for (int i = 0; i < 8; ++i) {
                    PrintPoint(&obj->rectinfo().corners()[i]);
                }

                PrintPoint(&obj->velocityabs());
                PrintPoint(&obj->accelerationabs());
                PrintTime(&obj->timecreation());
                PrintTime(&obj->lastupdatedtime());

                for (int i = 0; i < 3; ++i) {
                    std::cout << "obj->sensorid()[i] = " << obj->sensorid()[i] << std::endl;
                }

                std::cout << "obj->motionpattern = " << obj->motionpattern() << std::endl;
                std::cout << "obj->motionpatternhistory = " << obj->motionpatternhistory() << std::endl;
                std::cout << "obj->brakelightst = " << obj->brakelightst() << std::endl;
                std::cout << "obj->turnlightst = " << obj->turnlightst() << std::endl;
                std::cout << "obj->nearside = " << obj->nearside() << std::endl;

                for (int i = 0; i < 3; ++i) {
                    std::cout << "obj->associatedconf()[i] = " << obj->associatedconf()[i] << std::endl;
                }
                std::cout << "obj->age(20) = " << obj->age() << std::endl;
            }
        }
    }
}

void HelloWorldSubscriber::run() {
    std::cout << "Subscriber running. Please press enter to stop the Subscriber" << std::endl;
    std::cin.ignore();
}

void HelloWorldSubscriber::run(uint32_t number) {
    std::cout << "Subscriber running until " << number << "samples have been received" << std::endl;
    while (number > listener_.samples_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}
