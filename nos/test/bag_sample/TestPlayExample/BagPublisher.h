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
 * @file BagPublisher.h
 *
 */

#ifndef BagPUBLISHER_H_
#define BagPUBLISHER_H_

#include "BagPubSubTypes.h"

#include <fastdds/dds/publisher/DataWriterListener.hpp>
#include <fastdds/dds/topic/TypeSupport.hpp>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <rosbag2_cpp/reader.hpp>

class BagPublisher
{
public:

    BagPublisher();

    virtual ~BagPublisher();

    //!Initialize
    bool init(std::string::topic_name, std::string type_name, bool use_env);

    //!Publish a sample
//     bool publish(
//             rosbag2_cpp::Reader* reader_,
//             bool waitForListener = true);

    bool publish(
            rosbag2_cpp::Reader* reader_,
            bool waitForListener = true);

//     //!Run for number samples
//     void run(
//             uint32_t number,
//             uint32_t sleep);

private:

    //Bag hello_;

    eprosima::fastdds::dds::DomainParticipant* participant_;

    eprosima::fastdds::dds::Publisher* publisher_;

    eprosima::fastdds::dds::Topic* topic_;

    eprosima::fastdds::dds::DataWriter* writer_;

    bool stop_;

    class PubListener : public eprosima::fastdds::dds::DataWriterListener
    {
    public:

        PubListener()
            : matched_(0)
            , firstConnected_(false)
        {
        }

        ~PubListener() override
        {
        }

        void on_publication_matched(
                eprosima::fastdds::dds::DataWriter* writer,
                const eprosima::fastdds::dds::PublicationMatchedStatus& info) override;

        int matched_;

        bool firstConnected_;
    }
    listener_;

    void runThread(
            uint32_t number,
            uint32_t sleep, rosbag2_cpp::Reader* reader_);

    eprosima::fastdds::dds::TypeSupport type_;
};



#endif /* BagPUBLISHER_H_ */
