/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: main.cpp
 * @Date: 2023/11/23
 * @Author: shenda
 * @Desc: --
 */

#include "em/include/exec_client.h"
#include "mcu/include/MCUClient.h"
#include "utils/include/dc_logger.hpp"

#include <signal.h>
#include <exception>
#include <future>
#include <thread>

std::promise<void> g_stop_promise;
std::future<void> g_stop_future = g_stop_promise.get_future();

void signalHandler(int signum) {
    g_stop_promise.set_value();
}

int main() {
    using namespace hozon::netaos::dc;
    using namespace hozon::netaos::em;
    try {

        const char* debug_mcu_path="/opt/usr/col/bag/dc_mcu_debug_mode_on";
        struct stat stat_data;
        if ((stat(debug_mcu_path, &stat_data) == 0) && (S_ISREG(stat_data.st_mode))) {
            hozon::netaos::log::InitLogging("DCMCU", "NETAOS DCMCU", hozon::netaos::log::LogLevel::kDebug, hozon::netaos::log::HZ_LOG2CONSOLE |hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024), true);
        } else {
            hozon::netaos::log::InitLogging("DCMCU", "NETAOS DCMCU", hozon::netaos::log::LogLevel::kInfo, hozon::netaos::log::HZ_LOG2FILE, "/opt/usr/log/soc_log/", 10, (20 * 1024 * 1024), true);
        }
        DC_SERVER_LOG_INFO << "dc_mcu main start";
        std::shared_ptr<ExecClient> execli = std::make_shared<ExecClient>();
        int32_t ret = execli->ReportState(ExecutionState::kRunning);
        if (ret) {
            DC_SERVER_LOG_WARN << "dc_mcu report running failed";
        }
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        hozon::netaos::dc::MCUClient mcu_client;
        std::thread maintain_service_thread([&mcu_client] {
            try {
                mcu_client.Init();
            } catch (const std::exception& e) {
                DC_SERVER_LOG_ERROR << "dc_mcu thread exception: " << e.what();
            }
        });
        maintain_service_thread.detach();
        g_stop_future.wait();
        mcu_client.Deinit();
        ret = execli->ReportState(ExecutionState::kTerminating);
        if (ret) {
            DC_SERVER_LOG_WARN << "dc_mcu report terminating failed";
        }
    } catch (const std::exception& e) {
        DC_SERVER_LOG_ERROR << "dc_mcu main exception: " << e.what();
    }
    return 0;
}
