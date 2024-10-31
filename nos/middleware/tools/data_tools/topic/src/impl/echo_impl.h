#pragma once
#include <mutex>
#include <queue>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/types/DynamicDataFactory.h>
#include <google/protobuf/descriptor.pb.h>
#include "echo.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_someipbuf.h"
#include "sub_base.h"

namespace hozon {
namespace netaos {
namespace topic {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps;

class EchoImpl : public SubBase {
public:
    EchoImpl();
    void Start(EchoOptions echo_options);  //Initialization
    void Stop();
    ~EchoImpl();

protected:
    virtual void OnDataAvailable(eprosima::fastdds::dds::DataReader* reader) override;

private:
    std::string json_format_path_ = "./";
    std::map<std::string, std::vector<std::string>> adflite_process_topics_map_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon