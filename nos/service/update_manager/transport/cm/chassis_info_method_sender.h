#pragma once

#include "cm/include/method.h"
#include "idl/generated/chassis_ota_methodPubSubTypes.h"
#include "update_manager/common/data_def.h"

namespace hozon {
namespace netaos {
namespace update {

using namespace hozon::netaos::cm;

class ChassisInfoMethodSender {
public:
    ChassisInfoMethodSender();
    ~ChassisInfoMethodSender();

    void Init();
    void DeInit();
    bool ChassisMethodSend(std::unique_ptr<chassis_info_t>& output_info);

private:
    std::shared_ptr<Client<ChassisOtaMethod, ChassisOtaMethod>> chassis_info_client_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon