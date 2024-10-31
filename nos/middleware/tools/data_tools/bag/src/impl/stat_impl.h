#pragma once
// #include <dirent.h>
// #include <signal.h>
// #include <sys/stat.h>
#include <iostream>
#include <vector>

// #include "google/protobuf/message.h"
// #include "idl/generated/cm_protobufPubSubTypes.h"
// #include "idl/generated/cm_protobufTypeObject.h"
// #include "proto_factory/proto_factory.h"
// #include "data_tools_logger.hpp"
// #include "reader.h"

// #include "proto/dreamview/point_cloud.pb.h"
// #include "proto/perception/perception_obstacle.pb.h"
// #include "proto/soc/sensor_image.pb.h"  // proto 数据变量

namespace hozon {
namespace netaos {
namespace bag {

enum class StatErrorCode { K_success = 0, K_failed = 1, K_too_many_topics = 2 };

struct StatImplOptions {
    std::string url = "";
    std::vector<std::string> topics_list;
};

class StatImplmpl {
   public:
    StatImplmpl();
    ~StatImplmpl();
    StatErrorCode Start(const StatImplOptions& check_option);
};

}  // namespace bag
}  //namespace netaos
}  //namespace hozon