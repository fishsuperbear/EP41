
#include "chassis_info_method_sender.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {


ChassisInfoMethodSender::ChassisInfoMethodSender()

{
}

ChassisInfoMethodSender::~ChassisInfoMethodSender()
{
}

void
ChassisInfoMethodSender::Init()
{
    UPDATE_LOG_I("ChassisInfoMethodSender::Init");
    std::shared_ptr<ChassisOtaMethodPubSubType> req_chassis_type = std::make_shared<ChassisOtaMethodPubSubType>();
    std::shared_ptr<ChassisOtaMethodPubSubType> resp_chassis_type = std::make_shared<ChassisOtaMethodPubSubType>();
    chassis_info_client_ = std::make_shared<Client<ChassisOtaMethod, ChassisOtaMethod>>(req_chassis_type, resp_chassis_type);
    chassis_info_client_->Init(0, "/soc/chassis_ota_method");
}

void
ChassisInfoMethodSender::DeInit()
{
    UPDATE_LOG_I("ChassisInfoMethodSender::DeInit");
    if (nullptr != chassis_info_client_) {
        chassis_info_client_->Deinit();
        chassis_info_client_ = nullptr;
    }
}

bool 
ChassisInfoMethodSender::ChassisMethodSend(std::unique_ptr<chassis_info_t>& output_info){
    UM_DEBUG << "ChassisInfoMethodSender::ChassisMethodSend.";
    if (nullptr == chassis_info_client_) {
        UM_ERROR << "ChassisInfoMethodSender::ChassisMethodSend chassis_info_client_ is nullptr.";
        return false;
    }

    // if cm method not connect, wait connect(timeout: 5s)
    for (int i  = 0; i < 100; i++) {
        if (0 == chassis_info_client_->WaitServiceOnline(0)) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    std::shared_ptr<ChassisOtaMethod> req_chassis = std::make_shared<ChassisOtaMethod>();
    std::shared_ptr<ChassisOtaMethod> resq_chassis = std::make_shared<ChassisOtaMethod>();
    int iResult = chassis_info_client_->Request(req_chassis, resq_chassis, 50);
    if (0 != iResult) {
        UM_ERROR << "ChassisInfoMethodSender::ChassisMethodSend request failed.";
        return false;
    }

    UM_INFO << "ChassisInfoMethodSender::ChassisMethodSend VCU_ActGearPosition: " << resq_chassis->VCU_ActGearPosition()
                                                        << " BDCS10_AC_OutsideTemp: " << resq_chassis->BDCS10_AC_OutsideTemp()
                                                        << " ICU2_Odometer: " << resq_chassis->ICU2_Odometer()
                                                        << " BDCS1_PowerManageMode: " << resq_chassis->BDCS1_PowerManageMode()
                                                        << " Ignition_status: " << resq_chassis->Ignition_status()
                                                        << " ESC_VehicleSpeedValid: " << resq_chassis->ESC_VehicleSpeedValid()
                                                        << " ESC_VehicleSpeed: " << resq_chassis->ESC_VehicleSpeed();
    output_info->gear_display = resq_chassis->VCU_ActGearPosition();
    output_info->vehicle_speed_vaid = resq_chassis->ESC_VehicleSpeedValid();
    output_info->vehicle_speed = resq_chassis->ESC_VehicleSpeed();
    return true;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
