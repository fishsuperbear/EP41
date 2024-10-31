#pragma once
#include <cstdint>
#include <memory>
#include <fastdds/dds/topic/TopicDataType.hpp>
#include "idl/generated/servicebase.h"
#include "idl/generated/chassis_ota_method.h"
#include "cm/include/method.h"
#include "proto/planning/lanemarkers_lane_line.pb.h"
#include "idl/generated/chassis_ota_methodPubSubTypes.h"
#include "idl/generated/servicebasePubSubTypes.h"
namespace hozon {
namespace netaos {
namespace sensor {

class ChassisServer : public hozon::netaos::cm::Server<ChassisOtaMethod, ChassisOtaMethod> {
public: 
    ChassisServer(): hozon::netaos::cm::Server<ChassisOtaMethod, ChassisOtaMethod>
        (std::make_shared<ChassisOtaMethodPubSubType>(), std::make_shared<ChassisOtaMethodPubSubType>()) {  }
    ~ChassisServer() = default;

    int32_t Process(const std::shared_ptr<ChassisOtaMethod> req_data,
                    std::shared_ptr<ChassisOtaMethod> resp_data);
};

}
}
}
