#pragma once
#include <mutex>
#include <queue>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/types/DynamicDataFactory.h>
#include <google/protobuf/descriptor.pb.h>
#include "latency.h"
#include "idl/generated/cm_protobuf.h"
#include "sub_base.h"

namespace hozon {
namespace netaos {
namespace topic {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps;

class LatencyImpl : public SubBase {
public:
    LatencyImpl();
    void Start(LatencyOptions latency_options);  //Initialization
    void Stop();
    virtual ~LatencyImpl();

protected:
    virtual void OnDataAvailable(eprosima::fastdds::dds::DataReader* reader) override;

private:
    LatencyOptions latency_options_;
    std::map<std::string, std::vector<std::string>> adflite_process_topics_map_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon