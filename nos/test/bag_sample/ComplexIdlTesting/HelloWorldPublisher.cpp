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
 * @file HelloWorldPublisher.cpp
 *
 */

#include "HelloWorldPublisher.h"
#include <thread>
#include <unordered_map>
#include <fastdds/dds/domain/DomainParticipantFactory.hpp>
#include <fastdds/dds/publisher/DataWriter.hpp>
#include <fastdds/dds/publisher/Publisher.hpp>
#include <fastdds/dds/publisher/qos/DataWriterQos.hpp>
#include <fastdds/dds/publisher/qos/PublisherQos.hpp>
#include <fastrtps/attributes/ParticipantAttributes.h>
#include <fastrtps/attributes/PublisherAttributes.h>
#include "HelloWorldTypeObject.h"
#include "cm/include/cm_config.h"

using namespace eprosima::fastdds::dds;

int cout = 0;

template <typename T>
void SetPoint(T* point) {
    point->x() = 1.1;
    point->y() = 2.2;
    point->z() = 3.3;
}

template <typename T>
void SetTime(T* time) {
    time->sec() = 2378347;
    time->nsec() = 54646843;
}

HelloWorldPublisher::HelloWorldPublisher() : objvec_(nullptr), participant_(nullptr), publisher_(nullptr), topic_(nullptr), writer_(nullptr), type_(std::make_shared<ObjectVecPubSubType>()) {}

bool HelloWorldPublisher::init(bool use_env) {
    std::cout << "init start" << std::endl;

    objvec_ = std::make_shared<ObjectVec>();

    objvec_->obj().resize(20);
    for (int num = 0; num < 20; ++num) {
        auto obj = &(objvec_->obj().at(num));

        obj->header().seq(1000);
        obj->header().data_timestamp_us(12345678);
        obj->header().send_timestamp_us(12345678);

        obj->objectid(123);
        obj->type(123);  // 障碍物类别
        obj->detectsensor_current(123);
        obj->detectsensor_history(123);
        obj->maintenancestatus(123);
        obj->typeconfidence(123.456);        // 障碍物类别置信度
        obj->existenceprobability(123.456);  // 障碍物存在的概率

        SetPoint(&(obj->rectinfo().center()));
        SetPoint(&(obj->rectinfo().centerstddev()));
        SetPoint(&(obj->rectinfo().sizelwh()));
        SetPoint(&(obj->rectinfo().sizestddev()));
        obj->rectinfo().orientation(100.123);
        obj->rectinfo().orientationstddev(100.123);
        obj->rectinfo().corners().resize(8);
        for (int i = 0; i < 8; ++i) {
            SetPoint(&obj->rectinfo().corners()[i]);
        }

        SetPoint(&obj->velocityabs());
        SetPoint(&obj->accelerationabs());
        SetTime(&obj->timecreation());
        SetTime(&obj->lastupdatedtime());

        obj->sensorid().resize(3);
        for (int i = 0; i < 3; ++i) {
            obj->sensorid()[i] = i;
        }

        obj->motionpattern(15);
        obj->motionpatternhistory(16);
        obj->brakelightst(17);
        obj->turnlightst(18);
        obj->nearside(19);

        obj->associatedconf().resize(3);
        for (int i = 0; i < 3; ++i) {
            obj->associatedconf()[i] = 3;
        }

        obj->age(20);
    }

    // hello_.index(0);
    // hello_.message("HelloWorld");
    DomainParticipantQos pqos = PARTICIPANT_QOS_DEFAULT;

    pqos.wire_protocol().builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::DiscoveryProtocol::SIMPLE;
    pqos.wire_protocol().builtin.use_WriterLivelinessProtocol = true;

    pqos.transport().use_builtin_transports = false;
    auto udp_transport = std::make_shared<eprosima::fastdds::rtps::UDPv4TransportDescriptor>();

    udp_transport->interfaceWhiteList = hozon::netaos::cm::default_network_List;
    pqos.transport().user_transports.push_back(udp_transport);

    pqos.wire_protocol().builtin.typelookup_config.use_server = true;
    pqos.name("Participant_pub");
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
    registerHelloWorldTypes();
    type_.get()->auto_fill_type_information(true);
    type_.get()->auto_fill_type_object(false);
    type_.register_type(participant_);

    std::cout << "type_.register_type:" << type_.get_type_name() << std::endl;

    //CREATE THE PUBLISHER
    PublisherQos pubqos = PUBLISHER_QOS_DEFAULT;

    if (use_env) {
        participant_->get_default_publisher_qos(pubqos);
    }

    publisher_ = participant_->create_publisher(pubqos, nullptr);

    if (publisher_ == nullptr) {
        return false;
    }

    //CREATE THE TOPIC
    TopicQos tqos = TOPIC_QOS_DEFAULT;

    if (use_env) {
        participant_->get_default_topic_qos(tqos);
    }

    topic_ = participant_->create_topic("ComplexIdlTestingTopic", "ObjectVec", tqos);

    std::cout << "Topic:" << topic_->get_name() << "  " << topic_->get_type_name() << std::endl;

    if (topic_ == nullptr) {
        return false;
    }

    // CREATE THE WRITER
    DataWriterQos wqos = DATAWRITER_QOS_DEFAULT;

    if (use_env) {
        publisher_->get_default_datawriter_qos(wqos);
    }

    writer_ = publisher_->create_datawriter(topic_, wqos, &listener_);

    if (writer_ == nullptr) {
        return false;
    }

    return true;
}

HelloWorldPublisher::~HelloWorldPublisher() {
    if (writer_ != nullptr) {
        publisher_->delete_datawriter(writer_);
    }
    if (publisher_ != nullptr) {
        participant_->delete_publisher(publisher_);
    }
    if (topic_ != nullptr) {
        participant_->delete_topic(topic_);
    }
    DomainParticipantFactory::get_instance()->delete_participant(participant_);
}

void HelloWorldPublisher::PubListener::on_publication_matched(eprosima::fastdds::dds::DataWriter*, const eprosima::fastdds::dds::PublicationMatchedStatus& info) {
    if (info.current_count_change == 1) {
        matched_ = info.total_count;
        firstConnected_ = true;
        std::cout << "Publisher matched." << std::endl;
    } else if (info.current_count_change == -1) {
        matched_ = info.total_count;
        std::cout << "Publisher unmatched." << std::endl;
    } else {
        std::cout << info.current_count_change << " is not a valid value for PublicationMatchedStatus current count change" << std::endl;
    }
}

void HelloWorldPublisher::runThread(uint32_t samples, uint32_t sleep) {
    if (samples == 0) {
        while (!stop_) {
            if (publish(false)) {
                // std::cout << "Message: " << hello_.message() << " with index: " << hello_.index() << " SENT" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
        }
    } else {
        for (uint32_t i = 0; i < samples; ++i) {
            if (!publish()) {
                --i;
            } else {
                // std::cout << "Message: " << hello_.message() << " with index: " << hello_.index() << " SENT" << std::endl;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(sleep));
        }
    }
}

void HelloWorldPublisher::run(uint32_t samples, uint32_t sleep) {
    stop_ = false;
    std::thread thread(&HelloWorldPublisher::runThread, this, samples, sleep);
    if (samples == 0) {
        std::cout << "Publisher running. Please press enter to stop the Publisher at any time." << std::endl;
        std::cin.ignore();
        stop_ = true;
    } else {
        std::cout << "Publisher running " << samples << " samples." << std::endl;
    }
    thread.join();
}

bool HelloWorldPublisher::publish(bool waitForListener) {
    if (listener_.firstConnected_ || !waitForListener || listener_.matched_ > 0) {
        std::cout << "publish one" << std::endl;
        // hello_.index(hello_.index() + 1);
        // hello_.message("swswswswHelloWorld");
        writer_->write(objvec_.get());
        cout++;
        return true;
    }
    return false;
}
