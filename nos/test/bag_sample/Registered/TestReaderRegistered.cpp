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
 * @file TestReaderRegistered.cpp
 *
 */

#include "TestReaderRegistered.h"

#include "fastrtps/rtps/RTPSDomain.h"
#include "fastrtps/rtps/participant/RTPSParticipant.h"
#include "fastrtps/rtps/reader/RTPSReader.h"

#include "fastrtps/rtps/attributes/HistoryAttributes.h"
#include "fastrtps/rtps/attributes/RTPSParticipantAttributes.h"
#include "fastrtps/rtps/attributes/ReaderAttributes.h"

#include "fastrtps/rtps/history/ReaderHistory.h"

#include "fastrtps/attributes/TopicAttributes.h"
#include "fastrtps/qos/ReaderQos.h"

using namespace eprosima::fastrtps;
using namespace eprosima::fastrtps::rtps;

TestReaderRegistered::TestReaderRegistered() : mp_participant(nullptr), mp_reader(nullptr), mp_history(nullptr), m_listener(this), rtfpart_listener(this) {}

TestReaderRegistered::~TestReaderRegistered() {
    RTPSDomain::removeRTPSParticipant(mp_participant);
    delete (mp_history);
}

bool TestReaderRegistered::init() {
    //CREATE PARTICIPANT
    RTPSParticipantAttributes PParam;
    PParam.builtin.discovery_config.discoveryProtocol = eprosima::fastrtps::rtps::DiscoveryProtocol::SIMPLE;
    PParam.builtin.use_WriterLivelinessProtocol = true;
    mp_participant = RTPSDomain::createParticipant(0, PParam, &rtfpart_listener);
    if (mp_participant == nullptr) {
        return false;
    }
    //CREATE READERHISTORY
    HistoryAttributes hatt;
    hatt.payloadMaxSize = 255;
    mp_history = new ReaderHistory(hatt);

    //CREATE READER
    ReaderAttributes ratt;
    Locator_t loc(22222);
    ratt.endpoint.unicastLocatorList.push_back(loc);
    mp_reader = RTPSDomain::createRTPSReader(mp_participant, ratt, mp_history, &m_listener);  //生成reader
    if (mp_reader == nullptr) {
        return false;
    }

    return true;
}

bool TestReaderRegistered::reg() {
    std::cout << "Registering Reader" << std::endl;
    TopicAttributes Tatt;
    Tatt.topicKind = NO_KEY;
    Tatt.topicDataType = "string";
    Tatt.topicName = "exampleTopic";
    // Tatt.topicDataType = "HelloWorld";
    // Tatt.topicName = "HelloWorldTopic";
    ReaderQos Rqos;
    return mp_participant->registerReader(mp_reader, Tatt, Rqos);
}

void TestReaderRegistered::run() {
    printf("Press Enter to stop the Reader.\n");
    std::cin.ignore();
}

void TestReaderRegistered::MyListener::onNewCacheChangeAdded(RTPSReader* reader, const CacheChange_t* const change) {
    std::cout << "onNewCacheChangeAdded" << std::endl;
    printf("Received: %s\n", change->serializedPayload.data);
    reader->getHistory()->remove_change((CacheChange_t*)change);
    n_received++;
}

void TestReaderRegistered::RTPSPartListener::onWriterDiscovery(RTPSParticipant* participant, WriterDiscoveryInfo&& info) {
    std::cout << "Participant: onWriterDiscovery: m_typeName = " << info.info.typeName() << "; topic_name=" << info.info.topicName() << std::endl;

    //CREATE READERHISTORY
    HistoryAttributes hatt;
    hatt.payloadMaxSize = 255;
    eprosima::fastrtps::rtps::ReaderHistory* mp_history = new ReaderHistory(hatt);
    // _reader_handle_map[std::string(info.info.topicName())] = mp_history; //记录topic-reader

    //CREATE READER
    ReaderAttributes ratt;
    Locator_t loc(22222);
    ratt.endpoint.unicastLocatorList.push_back(loc);                                                                                                                        //指定端口
    eprosima::fastrtps::rtps::RTPSReader* temp_reader = RTPSDomain::createRTPSReader(_readerRegistered->mp_participant, ratt, mp_history, &_readerRegistered->m_listener);  //生成reader
    std::pair<std::string, std::string> temp(info.info.topicName(), info.info.typeName());
    _readerRegistered->_reader_topic_type[temp_reader] = temp;  //记录resder-topic-type

    std::cout << "Registering topic_name:" << info.info.topicName() << "; topic_name=" << info.info.typeName() << std::endl;
    TopicAttributes Tatt;
    Tatt.topicKind = NO_KEY;
    Tatt.topicName = info.info.topicName();
    Tatt.topicDataType = info.info.typeName();

    ReaderQos Rqos;
    _readerRegistered->mp_participant->registerReader(temp_reader, Tatt, Rqos);
}
