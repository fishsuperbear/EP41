#include "devm_server.h"
#include "devm_server_logger.h"
#include "cfg/include/config_param.h"
#include "devm_data_gathering.h"
#include "function_statistics.h"

namespace hozon {
namespace netaos {
namespace devm_server {

DevmServer::DevmServer() {
    devm_udp_mcu_version_ = std::make_shared<DevmUdpMcuVersion>();
    devm_udp_temp_vol_ = std::make_shared<DevmUdpTempAndVol>();
    devm_ver_check_ = std::make_shared<DevmVerCheck>();
    devm_server_zmq_ = std::make_shared<DevmServerImplZmq>();
}

DevmServer::~DevmServer() {
}

void DevmServer::Init() {
    FunctionStatistics func("DevmServer::Init, ");
    DevmDataGathering::GetInstance().Init();
    devm_server_zmq_->Init();
    ConfigParam::Instance()->Init(2000);
    devm_udp_mcu_version_->Init();
    DeviceInfomation::getInstance()->Init();
}

void DevmServer::Run() {
    udp_mcu_version_thread_ = std::thread(&DevmUdpMcuVersion::Run, devm_udp_mcu_version_);
    udp_temp_vol_thread_ = std::thread(&DevmUdpTempAndVol::Run, devm_udp_temp_vol_);
    ver_check_thread_ = std::thread(&DevmVerCheck::Run, devm_ver_check_);
}

void DevmServer::DeInit() {
    FunctionStatistics func("DevmServer::DeInit finish, ");
    devm_server_zmq_->DeInit();

    devm_udp_mcu_version_->SetStopFlag();
    devm_ver_check_->SetStopFlag();
    devm_udp_temp_vol_->SetStopFlag();

    if (udp_mcu_version_thread_.joinable()) {
        udp_mcu_version_thread_.join();
    }

    devm_udp_mcu_version_->DeInit();
    if (udp_temp_vol_thread_.joinable()) {
        udp_temp_vol_thread_.join();
    }

    if (ver_check_thread_.joinable()) {
        ver_check_thread_.join();
    }
    ConfigParam::Instance()->DeInit();
}

}  // namespace devm_server
}  // namespace netaos
}  // namespace hozon