/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: update check
 */
#include "update_manager/update_check/update_check.h"

#include <fstream>
#include <sys/statfs.h>
#include <limits.h>

#include "update_manager/common/common_operation.h"
#include "update_manager/log/update_manager_logger.h"
#include "update_manager/config/update_settings.h"
#include "update_manager/agent/diag_agent.h"
#include "update_manager/mcu/mcu_uds_peri.h"

namespace hozon {
namespace netaos {
namespace update {


UpdateCheck::UpdateCheck()
{
    chassis_data = std::make_unique<chassis_info_t>();
    state_client_ = std::make_unique<hozon::netaos::sm::StateClientZmq>();
    time_mgr_ = std::make_unique<TimerManager>();
}

void
UpdateCheck::Init()
{
    UM_INFO << "UpdateCheck::Init.";
    if (nullptr != time_mgr_) {
        time_mgr_->Init();
    }
    McuUdsPeri::Instance()->SetMcuUdsCallback(std::bind(&UpdateCheck::McuUdsCallback, this, std::placeholders::_1));
    UM_INFO << "UpdateCheck::Init Done.";
}

void
UpdateCheck::Deinit()
{
    UM_INFO << "UpdateCheck::Deinit.";
    if (nullptr != time_mgr_) {
        time_mgr_->DeInit();
    }
    UM_INFO << "UpdateCheck::Deinit Done.";
}

int16_t
UpdateCheck::UpdatePreConditionCheck()
{
    //TODO: check ign & ready & gear & hand brake & voltage & speed & charging state
    auto res = DiagAgent::Instance()->SendChassisInfo(chassis_data);
    if (!res) {
        UPDATE_LOG_D("get chassis info error, pre check error, please try again !");
        return -2;
    }
    UPDATE_LOG_D("UpdateCheck::UpdatePreConditionCheck gear_display is : %d, vehicle_speed_vaid is : %s, vehicle_speed is : %.2f"
        ,chassis_data->gear_display, chassis_data->vehicle_speed_vaid ? "true" : "false", chassis_data->vehicle_speed);

    // udisk free space check.
    PathCreate(UpdateSettings::Instance().PathForWork());
    struct statfs diskInfo;
    statfs(UpdateSettings::Instance().PathForWork().c_str(), &diskInfo);
    uint64_t blocksize = diskInfo.f_bsize;
    uint64_t totalsize = blocksize * diskInfo.f_blocks;
    uint64_t freeDisk = diskInfo.f_bfree * blocksize;
    uint64_t availableDisk = diskInfo.f_bavail * blocksize;
    if (availableDisk * 100 / totalsize < 25) {
        UPDATE_LOG_D("upgrade path: %s only %ld%% space left need do clear, aviable: %ldMB, free: %ldMB, total: %ldMB.",
            UpdateSettings::Instance().PathForUpgrade().c_str(), availableDisk * 100 / totalsize, availableDisk>>20, freeDisk>>20, totalsize>>20);

        PathClear(UpdateSettings::Instance().PathForUpgrade());
        return -1;
    }
    return 0;
}

int16_t 
UpdateCheck::UpdateModeChange(const std::string& mode)
{
    UPDATE_LOG_I("UpdateStateChange");
    std::string currtMode{""};
    int retries = 0;
    while (retries < max_retries) {
        auto cur = state_client_->GetCurrMode(currtMode);
        if (cur != 0) {
            UPDATE_LOG_E("GetCurrMode error, code is : %d", cur);
            retries++;
            UM_DEBUG << "Retry " << retries << "...";
            std::this_thread::sleep_for(std::chrono::seconds(2));
            continue;
        }
        if (currtMode == mode) {
            UPDATE_LOG_D("no need change mode !");
            return 0;
        }
        auto change = state_client_->SwitchMode(mode);
        if (change == 0) {
            UPDATE_LOG_D("SwitchMode success !");
            return 0;
        } else {
            UPDATE_LOG_E("SwitchMode error, code is : %d", change);
            retries++;
            UM_DEBUG << "Retry " << retries << "...";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    return -1;
}

bool 
UpdateCheck::RestartUpdateService()
{
    UPDATE_LOG_I("RestartUpdateService");
    auto res = state_client_->ProcRestart(UPDATE_SERVICE);
    if (res != 0) {
        UM_ERROR << "ProcRestart svp_update error, code is : " << res;
        return false;
    }
    UM_DEBUG << "ProcRestart svp_update succ.";
    return true;
}


int16_t 
UpdateCheck::GetGear()
{
    return static_cast<int16_t>(chassis_data->gear_display);
}

float 
UpdateCheck::GetSpeed()
{
    if (chassis_data->vehicle_speed_vaid) {
        // do nothing
    } else {
        chassis_data->vehicle_speed = -1;
    }
    return chassis_data->vehicle_speed;
}

bool 
UpdateCheck::GetSpaceEnough()
{
    PathCreate(UpdateSettings::Instance().PathForWork());
    struct statfs diskInfo;
    statfs(UpdateSettings::Instance().PathForWork().c_str(), &diskInfo);
    uint64_t blocksize = diskInfo.f_bsize;
    uint64_t totalsize = blocksize * diskInfo.f_blocks;
    uint64_t availableDisk = diskInfo.f_bavail * blocksize;
    if (availableDisk * 100 / totalsize < 25) {
        return false;
    }
    return true;
}

void 
UpdateCheck::StartTimer(uint16_t second)
{
    UPDATE_LOG_D("UpdateCheck::Start timmer !");
    time_mgr_->StartFdTimer(time_fd_um_, second * 1000, std::bind(&UpdateCheck::UpdateTimeout, this), NULL, false);
}

void 
UpdateCheck::UpdateTimeout()
{
    UPDATE_LOG_D("FD Timer timeout, switch into Normal Mode.");
    UpdateModeChange(UPDATE_MODE_NORMAL);
}

void 
UpdateCheck::McuUdsCallback(const McuUdsMsg& udsdate)
{
    UPDATE_LOG_D("UpdateCheck::McuUdsCallback.");
    if (udsdate.at(1) == 0x03)
    {
        UM_DEBUG << "receive mcu [28 03 01], now to switch Mode into [Update]";
        UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_OTA);
        time_mgr_->StopFdTimer(time_fd_um_);
        StartTimer(5*60);

    } else if (udsdate.at(1) == 0x00) {
        UM_DEBUG << "receive mcu [28 00 01], now to switch Mode into [Normal]";
        UpdateCheck::Instance().UpdateModeChange(UPDATE_MODE_NORMAL);
    } else {
        // go on
    }
}

}  // namespace update
}  // namespace netaos
}  // namespace hozon
