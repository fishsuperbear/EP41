/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: main function definition
 */

#include <signal.h>

#include <string>

#include "can_parser_ins_pvatb.h"
// #include "can_stack_utils.h"
#include "canstack_e2e.h"
#include "canstack_logger.h"
#include "canstack_manager.h"
#include "config_loader.h"
// #include "fault_report.h"
#include "ins_pvatb_publisher.h"

using namespace std;

#ifdef INS_PVATB_DEBUG_ON
#define CONFIG_FILE_PATH "conf/hz_ins.yaml"
#else
#define CONFIG_FILE_PATH "/opt/app/1/runtime_service/hz_ins/conf/hz_ins.yaml"
#endif

hozon::netaos::canstack::CanStackManager* cans_manager = nullptr;
int g_stopFlag = 0;
const std::string defaultCanName = "can7";

std::mutex mtx;
std::condition_variable cv;

using namespace hozon::netaos::canstack;
using namespace hozon;

void SigHandler(int signum) {
    g_stopFlag = 1;
    if (cans_manager != nullptr) {
        cans_manager->Stop();
    }

    // execClient.ReportExecutionState(ara::exec::ExecutionState::kTerminating);

    std::unique_lock<std::mutex> lck(mtx);
    cv.notify_one();
}

int main(int argc, char* argv[]) {
    // execClient.ReportExecutionState(ara::exec::ExecutionState::kRunning);

    signal(SIGINT, SigHandler);
    signal(SIGTERM, SigHandler);

    #ifdef INS_PVATB_DEBUG_ON
    std::string binary_path = std::string(argv[0]);
    size_t pos = 0;
    for(size_t i = 0; i < binary_path.size(); i++) {
        if(binary_path[i] == '/') {
            pos = i;
        }
    }
    std::string folder_path = binary_path.substr(0, pos);
    std::string conf_path = folder_path + "/../" + std::string(CONFIG_FILE_PATH);
    #else 
    std::string conf_path = std::string(CONFIG_FILE_PATH);
    #endif
    CAN_ERALY_LOG << "config path: " << conf_path;

    if (!hozon::netaos::canstack::ConfigLoader::LoadConfig(conf_path)) {
        // hozon::netaos::canstack::CanBusReport::Instance().ReportModuleInitFault(defaultCanName, ModuleInitErrorCase::LOAD_CONFIG_ERROR);
        // execClient.ReportExecutionState(ara::exec::ExecutionState::kTerminating);
        return 0;
    }
    CAN_ERALY_LOG << "log app name: " <<  ConfigLoader::log_app_name_[0] \
                  << "log level: " << ConfigLoader::log_level_\
                  << "log mode: " << ConfigLoader::log_mode_\
                  << "log file: " << ConfigLoader::log_file_;
    if (ConfigLoader::log_app_name_.size() < 1) {
        hozon::netaos::canstack::SensorLogger::GetInstance().InitLogger(
                "INS", static_cast<LogLevel>(ConfigLoader::log_level_), ConfigLoader::log_mode_,  ConfigLoader::log_file_);
    } else {
        hozon::netaos::canstack::SensorLogger::GetInstance().InitLogger(
            ConfigLoader::log_app_name_[0], static_cast<LogLevel>(ConfigLoader::log_level_),
                 ConfigLoader::log_mode_,  ConfigLoader::log_file_);
    }
    hozon::netaos::canstack::SensorLogger::GetInstance().CreateLogger(
            "INS", "Ins can stack", static_cast<LogLevel>(ConfigLoader::log_level_));
    CAN_LOG_INFO << "Can stack main begin...";

    if (hozon::netaos::canstack::ConfigLoader::can_port_.size() < 1) {
        CAN_LOG_INFO << "LoadConfig no can port!";
        return 0;
    }

    std::string canName = defaultCanName;
    // if (hozon::netaos::canstack::CanStackUtils::IsValidCanIf(hozon::canstack::ConfigLoader::can_port_[0])) {
    //     canName = hozon::netaos::canstack::ConfigLoader::can_port_[0];
    // } else {
    //     CAN_LOG_ERROR << hozon::canstack::ConfigLoader::can_port_[0] << " is invalid! use default can: " << canName;
    // }

    hozon::netaos::canstack::E2ESupervision::Instance()->Init(conf_path);

    cans_manager = hozon::netaos::canstack::CanStackManager::Instance();
    hozon::netaos::canstack::CanParser* canParser = hozon::netaos::ins_pvatb::CanParserInsPvatb::Instance();
    hozon::netaos::ins_pvatb::InsPvatbPublisher* publisher = hozon::netaos::ins_pvatb::InsPvatbPublisher::Instance();

    int res = cans_manager->Init(canName, canParser, publisher, nullptr);
    if (res < 0) {
        return 0;
    } else {
        cans_manager->Start();
    }

    while (!g_stopFlag) {
        std::unique_lock<std::mutex> lck(mtx);
        cv.wait(lck);
    }

    return 0;
}
