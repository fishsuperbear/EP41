/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: main function definition
 */

#include <signal.h>

#include <string>

#include "can_parser_chassis.h"
#include "canstack_manager.h"
#include "chassis_publisher.h"
#include "canstack_logger.h"
#include "chassis_subscriber.h"
#include "config_loader.h"

#define CONFIG_FILE_PATH "/opt/app/1/runtime_service/hz_chassis/conf/hz_chassis.yaml"

hozon::netaos::canstack::CanStackManager* cans_manager = nullptr;
int g_stopFlag = 0;
const std::string defaultCanName1 = "can9";
std::mutex mtx;
std::condition_variable cv;

using namespace hozon::netaos::canstack;
using namespace hozon;

void SigHandler(int signum) {
    CAN_LOG_INFO << "Chassis SigHandler enter, signum [" << signum << "]";
    g_stopFlag = 1;
    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_all();
}

int main(int argc, char* argv[]) {
    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    if (!hozon::netaos::canstack::ConfigLoader::LoadConfig(CONFIG_FILE_PATH)) {
        // if (fmServerAvailable) {
        //     FaultInfo_t faultInfo = GetModuleInitFaultInfo(defaultCanName1, ModuleInitErrorCase::LOAD_CONFIG_ERROR);
        //     hozon::fm::HzFMAgent::Instance()->ReportFault(hozon::fm::HzFMAgent::Instance()->GenFault(faultInfo.faultId, faultInfo.faultObj), 1);
        //     /* ReportFault CONFIG_FILE_PATH ERROR! */
        // } else {
        //     /* LoadConfig CONFIG_FILE_PATH fmServerAvailable does not exit! */
        // }
        // execClient.ReportExecutionState(ara::exec::ExecutionState::kTerminating);
        return 0;
    } else {
        /* hozon::netaos::canstack::ConfigLoader::LoadConfig ok! */
    }

    if (ConfigLoader::log_app_name_.size() < 1) {
        hozon::netaos::canstack::SensorLogger::GetInstance().InitLogger("Chassis", static_cast<LogLevel>(ConfigLoader::log_level_), ConfigLoader::log_mode_,ConfigLoader::log_file_);
        /* ConfigLoader::log_app_name_.size() < 1! */

    } else {
        hozon::netaos::canstack::SensorLogger::GetInstance().InitLogger(ConfigLoader::log_app_name_[0], static_cast<LogLevel>(ConfigLoader::log_level_), ConfigLoader::log_mode_,ConfigLoader::log_file_);
        /* hozon::netaos::canstack::SensorLogger::GetInstance().InitLogger ok! */
    }

    hozon::netaos::canstack::SensorLogger::GetInstance().CreateLogger("Chassis", "Chassis can stack",static_cast<LogLevel>(ConfigLoader::log_level_));
	// hozon::common::PlatformCommon::Init("H004",ConfigLoader::log_level_,ConfigLoader::log_mode_);
	CAN_LOG_INFO << "hz_chassis process start.";

    if (hozon::netaos::canstack::ConfigLoader::can_port_.size() < 1) {
        CAN_LOG_ERROR << "LoadConfig can port num wrong!,can_port.size(): " << hozon::netaos::canstack::ConfigLoader::can_port_.size();
        // if (fmServerAvailable) {
        //     FaultInfo_t faultInfo = GetModuleInitFaultInfo(defaultCanName1, ModuleInitErrorCase::LOAD_CONFIG_ERROR);
        //     hozon::fm::HzFMAgent::Instance()->ReportFault(hozon::fm::HzFMAgent::Instance()->GenFault(faultInfo.faultId, faultInfo.faultObj), 1);
        //     CAN_LOG_ERROR << "ReportFault can_port.size() ERROR!";
        // } else {
        //     CAN_LOG_ERROR << "LoadConfig can_port_.size() fmServerAvailable does not exit!";
        // }
        // execClient.ReportExecutionState(ara::exec::ExecutionState::kTerminating);
        return 0;
    } else {
        CAN_LOG_INFO << "hozon::netaos::canstack::ConfigLoader::LoadConfig ok! can_port_: " << hozon::netaos::canstack::ConfigLoader::can_port_.size();
    }

    std::string canName1 = defaultCanName1;
    CAN_LOG_INFO << "canName1: " << canName1;

    if (true == hozon::netaos::canstack::ConfigLoader::version_for_EP40_) {
        CAN_LOG_INFO << "version is EP40:";
    } else {
        CAN_LOG_INFO << "version is EP30:";
    }

    CAN_LOG_INFO << "before CanStackManager get Instance";
    cans_manager = hozon::netaos::canstack::CanStackManager::Instance();

    CAN_LOG_INFO << "before canParser get Instance";
    hozon::netaos::canstack::CanParser* canParser = hozon::netaos::canstack::chassis::CanParserChassis::Instance();

    CAN_LOG_INFO << "before publisher get Instance";
    hozon::netaos::canstack::chassis::ChassisPublisher* publisher = hozon::netaos::canstack::chassis::ChassisPublisher::Instance();

    CAN_LOG_INFO << "before subscriber get Instance";
    hozon::netaos::canstack::chassis::ChassisSubscriber* subscriber = hozon::netaos::canstack::chassis::ChassisSubscriber::Instance();
    CAN_LOG_INFO << "after subscriber get Instance";

    publisher->PassCanName(canName1);

    CAN_LOG_INFO << "before CanStackManager Init";
    int res = cans_manager->Init(canName1, canParser, publisher, subscriber);
    if (res < 0) {
        CAN_LOG_ERROR << "The return value res of cans_manager is less than 0,res: " << res;
        // execClient.ReportExecutionState(ara::exec::ExecutionState::kTerminating);
        return 0;
    } else {
        cans_manager->Start();
        CAN_LOG_INFO << "cans_manager start ok!";
    }
    CAN_LOG_INFO << "after CanStackManager Init";
    while (!g_stopFlag) {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck);
    }

    if (cans_manager != nullptr) {
        CAN_LOG_INFO << "before cans_manager->Stop()";
        cans_manager->Stop();
        CAN_LOG_INFO << "after cans_manager->Stop()";
    } else {
        CAN_LOG_WARN << "cans_manager is nullptr";
    }

    CAN_LOG_INFO << "hz_chassis process finished.";
    return 0;
}
