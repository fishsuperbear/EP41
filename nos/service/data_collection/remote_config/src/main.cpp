/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: main.cpp
 * @Date: 2023/12/13
 * @Author: shenda
 * @Desc: --
 */

#include <signal.h>
#include <exception>
#include <future>
#include <thread>
#include "em/include/exec_client.h"
#include "remote_config/include/remote_config.h"
#include "utils/include/dc_logger.hpp"
#include "dc_download.h"
#include "config_param.h"

using namespace hozon::netaos::dc;
using namespace hozon::netaos::em;
std::promise<void> g_stop_promise;
std::future<void> g_stop_future = g_stop_promise.get_future();

void signalHandler(int signum) {
    g_stop_promise.set_value();
}

int main(int argc, char** argv) {
    try {
        std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
        int32_t ret = execli->ReportState(ExecutionState::kRunning);
        if (ret) {
            DC_SERVER_LOG_WARN << "remote_config report running failed";
        }
        hozon::netaos::log::InitLogging("DCCFG", "NETAOS REMOTECONFIGSERVER", hozon::netaos::log::LogLevel::kTrace, hozon::netaos::log::HZ_LOG2CONSOLE | hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024), true);
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        DC_SERVER_LOG_DEBUG << "remote_config main";
        auto cfg_param = hozon::netaos::cfg::ConfigParam::Instance();
        cfg_param->Init(3000);
        Download download;
        download.start();

        // hozon::netaos::dc::MCUClient mcu_client;
        // std::thread maintain_service_thread([&mcu_client] {
        //     try {
        //         mcu_client.Init();
        //     } catch (const std::exception& e) {
        //         DC_SERVER_LOG_ERROR << "remote_config thread exception: " << e.what();
        //     }
        // });
        // maintain_service_thread.detach();
        g_stop_future.wait();
        // mcu_client.Deinit();

        download.stop();
        cfg_param->DeInit();
        ret = execli->ReportState(ExecutionState::kTerminating);
        if (ret) {
            DC_SERVER_LOG_WARN << "remote_config report terminating failed";
        }
    } catch (const std::exception& e) {
        DC_SERVER_LOG_ERROR << "remote_config main exception: " << e.what();
    }

    // signal(SIGINT, SigHandler);
    // signal(SIGTERM, SigHandler);
    // tspCommon::GetInstance().Init();

    // std::string dc_config_service_name = "tcp://*:90332";
    // std::unique_ptr<DcConfigServerImpl> server_ = std::make_unique<DcConfigServerImpl>();
    // server_->Start(dc_config_service_name);

    // while (!stopFlag_) {
    //     std::this_thread::sleep_for(std::chrono::milliseconds(500));
    // }

    // std::cout << "server->Stop before!~" << std::endl;
    // server_->Stop();
    // tspCommon::GetInstance().Deinit();
    // std::cout << "server->Stop after!~" << std::endl;
    return 0;
}
