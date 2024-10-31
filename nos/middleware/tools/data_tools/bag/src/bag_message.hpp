#pragma once
// #include <fastdds/rtps/common/SerializedPayload.h>
#include <bag_data_type.h>

namespace hozon {
namespace netaos {
namespace bag {

using namespace eprosima::fastrtps::rtps;
using namespace eprosima::fastrtps;

struct BagMessage {
    std::string topic;
    std::string type;
    int64_t time;

    BagDataType data;

    // BagMessage(const BagMessage&) = delete;
    const BagMessage& operator=(const BagMessage&) = delete;

    //!Default constructor
    BagMessage() : topic(""), type(""), time(0) {}
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon