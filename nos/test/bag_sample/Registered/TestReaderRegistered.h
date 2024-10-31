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
 * @file TestReaderRegistered.h
 *
 */

#ifndef TESTREADERREGISTERED_H_
#define TESTREADERREGISTERED_H_

#include "fastrtps/rtps/rtps_fwd.h"

#include "fastdds/rtps/history/ReaderHistory.h"
#include "fastdds/rtps/participant/RTPSParticipantListener.h"
#include "fastdds/rtps/reader/RTPSReader.h"
#include "fastrtps/rtps/reader/ReaderListener.h"

#include <fastrtps/types/TypeObjectFactory.h>

#include <unordered_map>

#include <rosbag2_cpp/writer.hpp>

using namespace eprosima::fastrtps::rtps;
using namespace eprosima::fastrtps;

class TestReaderRegistered {
   public:
    TestReaderRegistered();
    virtual ~TestReaderRegistered();
    eprosima::fastrtps::rtps::RTPSParticipant* mp_participant;
    eprosima::fastrtps::rtps::RTPSReader* mp_reader;
    eprosima::fastrtps::rtps::ReaderHistory* mp_history;
    std::map<eprosima::fastrtps::rtps::RTPSReader*, std::string> _reader_topic_map;
    std::map<eprosima::fastrtps::rtps::RTPSReader*, std::pair<std::string, std::string>> _reader_topic_type;
    bool init();  //Initialization
    bool reg();   //Register
    void run();   //Run

    class MyListener : public eprosima::fastrtps::rtps::ReaderListener {
       public:
        MyListener(TestReaderRegistered* readerRegistered) : n_received(0), n_matched(0), _readerRegistered(readerRegistered) {
            writer_ = std::make_unique<rosbag2_cpp::Writer>();
            rosbag2_storage::StorageOptions storage_options;
            storage_options.uri = "my_bag";
            storage_options.storage_id = "mcap";
            rosbag2_cpp::ConverterOptions converter_options{};
            writer_->open(storage_options, converter_options);
        }

        ~MyListener() {}

        /**
         * @brief Method called when the liveliness of a reader changes
         * @param reader The reader
         * @param status The liveliness changed status
         */
        virtual void on_liveliness_changed(RTPSReader* reader, const eprosima::fastdds::dds::LivelinessChangedStatus& status) {
            static_cast<void>(reader);
            static_cast<void>(status);
        }

        /**
         * This method is called when a new Writer is discovered, with a Topic that
         * matches that of a local reader, but with an offered QoS that is incompatible
         * with the one requested by the local reader
         * @param reader Pointer to the RTPSReader.
         * @param qos A mask with the bits of all incompatible Qos activated.
         */
        virtual void on_requested_incompatible_qos(RTPSReader* reader, eprosima::fastdds::dds::PolicyMask qos) { std::cout << "reader: on_requested_incompatible_qos" << std::endl; }

        /**
         * This method is called when the reader detects that one or more samples have been lost.
         *
         * @param reader                         Pointer to the RTPSReader.
         * @param sample_lost_since_last_update  The number of samples that were lost since the last time this
         *                                       method was called for the same reader.
         */
        virtual void on_sample_lost(RTPSReader* reader, int32_t sample_lost_since_last_update) { std::cout << "reader: on_sample_lost" << std::endl; }

        /**
         * @brief Method called when the discovery information of a writer regarding a reader changes.
         *
         * @param reader       The reader.
         * @param reason       The reason motivating this method to be called.
         * @param writer_guid  The GUID of the writer for which the discovery information changed.
         * @param writer_info  Discovery information about the writer. Will be @c nullptr for reason @c REMOVED_WRITER.
         */
        virtual void on_writer_discovery(RTPSReader* reader, WriterDiscoveryInfo::DISCOVERY_STATUS reason, const GUID_t& writer_guid, const WriterProxyData* writer_info) {
            std::cout << "reader: on_writer_discovery" << std::endl;
        }

        /**
         * This method is called when the reader rejects a samples.
         *
         * @param reader  Pointer to the RTPSReader.
         * @param reason  Indicates reason for sample rejection.
         * @param change  Pointer to the CacheChange_t. This is a const pointer to const data
         *                to indicate that the user should not dispose of this data himself.
         */
        virtual void on_sample_rejected(RTPSReader* reader, eprosima::fastdds::dds::SampleRejectedStatusKind reason, const CacheChange_t* const change) {
            std::cout << "reader: on_sample_rejected" << std::endl;
        }

        /**
         * This method is called when new CacheChange_t objects are made available to the user.
         * @note This method is currently never called. Implementation will be added in future releases.
         *
         * @param [in]  reader                            Pointer to the reader performing the notification.
         * @param [in]  writer_guid                       GUID of the writer from which the changes were received.
         * @param [in]  first_sequence                    Sequence number of the first change made available.
         * @param [in]  last_sequence                     Sequence number of the last change made available.
         *                                                It will always be greater or equal than @c first_sequence.
         * @param [out] should_notify_individual_changes  Whether the individual changes should be notified by means of
         *                                                @c onNewCacheChangeAdded.
         */
        virtual void on_data_available(  //不需要
            RTPSReader* reader, const GUID_t& writer_guid, const SequenceNumber_t& first_sequence, const SequenceNumber_t& last_sequence, bool& should_notify_individual_changes) override {
            std::cout << "should_notify_individual_changes=" << should_notify_individual_changes << std::endl;
            std::cout << "reader: on_data_available" << std::endl;
            if (_readerRegistered->_reader_topic_type.find(reader) != _readerRegistered->_reader_topic_type.end()) {
                //std::cout << "receive " <<  _readerRegistered->_reader_topic_type[reader].first << std::endl;

                // eprosima::fastrtps::rtps::ReaderListener * temp = reader->getListener();
                reader->getListener();
                // std::cout << "reader topic: " << dynamic_cast<MyListener*>(temp)->_topic << std::endl;

                eprosima::fastrtps::rtps::CacheChange_t* change;
                eprosima::fastrtps::rtps::WriterProxy* writer;

                reader->nextUnreadCache(&change, &writer);

                std::cout << "data length=" << change->serializedPayload.length << std::endl;

                //开始记录数据
                std::cout << "record:  " << _readerRegistered->_reader_topic_type[reader].first << " " << _readerRegistered->_reader_topic_type[reader].second << std::endl;
                eprosima::fastrtps::Time_t time_stamp;
                eprosima::fastrtps::Time_t::now(time_stamp);
                writer_->write(change->serializedPayload, _readerRegistered->_reader_topic_type[reader].first, _readerRegistered->_reader_topic_type[reader].second, time_stamp.to_ns());

                printf("data=: %s\n", change->serializedPayload.data);
                reader->getHistory()->remove_change((CacheChange_t*)change);
            }
        }

        /**
         * This method is called when a new CacheChange_t is added to the ReaderHistory.
         * @param reader Pointer to the reader.
         * @param change Pointer to the CacheChange_t. This is a const pointer to const data
         * to indicate that the user should not dispose of this data himself.
         * To remove the data call the remove_change method of the ReaderHistory.
         * reader->getHistory()->remove_change((CacheChange_t*)change).
         */
        void onNewCacheChangeAdded(eprosima::fastrtps::rtps::RTPSReader* reader, const eprosima::fastrtps::rtps::CacheChange_t* const change) override;

        //  发现符合的要求的发布者
        void onReaderMatched(eprosima::fastrtps::rtps::RTPSReader*, eprosima::fastrtps::rtps::MatchingInfo& info) override {
            std::cout << "reader: onReaderMatched" << std::endl;
            if (info.status == eprosima::fastrtps::rtps::MATCHED_MATCHING) {
                n_matched++;
            }
        }

        uint32_t n_received;
        uint32_t n_matched;
        TestReaderRegistered* _readerRegistered;
        std::string _topic;
        std::string _type;
        std::unique_ptr<rosbag2_cpp::Writer> writer_;
    } m_listener;

    class RTPSPartListener : public eprosima::fastrtps::rtps::RTPSParticipantListener {
       public:
        RTPSPartListener(TestReaderRegistered* registered) : _readerRegistered(registered) {}

        ~RTPSPartListener() {}

        virtual void onParticipantDiscovery(eprosima::fastrtps::rtps::RTPSParticipant* participant, ParticipantDiscoveryInfo&& info) {
            //std::cout << " Participant: onParticipantDiscovery: name = " << info.info.m_participantName << std::endl;
        }

        virtual void onReaderDiscovery(RTPSParticipant* participant, ReaderDiscoveryInfo&& info) { std::cout << "Participant: onReaderDiscovery: m_typeName = " << info.info.typeName() << std::endl; }

        virtual void onWriterDiscovery(RTPSParticipant* participant, WriterDiscoveryInfo&& info);  //发布者会为每个topic创建一个writer

        virtual void on_type_discovery(RTPSParticipant* participant, const SampleIdentity& request_sample_id, const string_255& topic, const types::TypeIdentifier* identifier,
                                       const types::TypeObject* object, types::DynamicType_ptr dyn_type) {
            // std::cout << "Participant: on_type_discovery: topic = " << topic << std::endl;
        }

        virtual void on_type_dependencies_reply(RTPSParticipant* participant, const SampleIdentity& request_sample_id, const types::TypeIdentifierWithSizeSeq& dependencies) {
            std::cout << "Participant: on_type_dependencies_reply" << std::endl;

            for (size_t i = 0; i < dependencies.size(); ++i) {
                const types::TypeIdentifierWithSize& type_id_with_size = dependencies[i];
                //const types::TypeIdentifier& type_id = type_id_with_size.type_id();
                const uint32_t type_size = type_id_with_size.typeobject_serialized_size();

                std::cout << "type_size=" << type_size << std::endl;

                // Use the RTPS protocol to retrieve the TypeObject
                // const eprosima::fastrtps::types::TypeObject* type_obj = participant->getTypes(type_id);

                // // Create a new DynamicType using the retrieved TypeObject
                // DynamicType_ptr dynamic_type = DynamicTypeBuilderFactory::get_instance()->create_type(type_obj);
            }
        }

        virtual void on_type_information_received(RTPSParticipant* participant, const string_255& topic_name, const string_255& type_name, const types::TypeInformation& type_information) {
            std::cout << "Participant: on_type_information_received. topic_name=" << topic_name << std::endl;
        }

        // std::unordered_map<std::string, eprosima::fastrtps::rtps::ReaderHistory*> _reader_handle_map;
        TestReaderRegistered* _readerRegistered;

    } rtfpart_listener;
};

#endif /* TESTREADER_H_ */
