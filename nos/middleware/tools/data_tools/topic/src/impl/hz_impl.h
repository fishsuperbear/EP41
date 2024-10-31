#pragma once
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <fastdds/dds/domain/DomainParticipant.hpp>
#include <fastdds/dds/domain/DomainParticipantListener.hpp>
#include <fastdds/dds/subscriber/DataReader.hpp>
#include <fastdds/dds/subscriber/qos/DataReaderQos.hpp>
#include <fastrtps/types/DynamicDataFactory.h>
#include "hz.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/proto_method.h"
#include "sub_base.h"

namespace hozon {
namespace netaos {
namespace topic {

using namespace eprosima::fastdds::dds;
using namespace eprosima::fastrtps;

struct HzStruct {
    uint count = 0;
    double alit_fre = 0;
    std::chrono::duration<double> min_delta = std::chrono::duration<double>(10.0);
    std::chrono::duration<double> max_delta = std::chrono::duration<double>(0);
    std::chrono::steady_clock::time_point last_time;
    std::chrono::steady_clock::time_point first_time;
    bool is_matched = true;
};

class HzImpl : public SubBase {
   public:
    HzImpl();
    void Start(HzOptions hz_options);  //Initialization
    void Stop();
    virtual ~HzImpl();

   protected:
    virtual void OnDataAvailable(eprosima::fastdds::dds::DataReader* reader) override;
    virtual void OnSubscribed(TopicInfo topic_info) override;

   private:
    void OutputHzInfos();
    HzOptions hz_options_;
    std::map<std::string, HzStruct> topic_infos_;
    std::chrono::steady_clock::time_point _last_print_time;
    std::thread hzinfo_thread_;
};

}  // namespace topic
}  //namespace netaos
}  //namespace hozon