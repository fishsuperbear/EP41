/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: update check
 */
#ifndef UPDATE_CHECK_H
#define UPDATE_CHECK_H

#include <stdint.h>
#include <vector>
#include <string>

#include "update_manager/common/data_def.h"
#include "sm/include/state_client_zmq.h"
#include "update_manager/common/timer_manager.h"

namespace hozon {
namespace netaos {
namespace update {


class UpdateCheck {
public:
    static UpdateCheck &Instance()
    {
        static UpdateCheck instance;
        return instance;
    }

    void Init();
    void Deinit();
    int16_t UpdatePreConditionCheck();
    int16_t UpdateModeChange(const std::string& mode);
    DiagUpdateStatus GetUpdateMode();
    bool RestartUpdateService();

    int16_t GetGear();
    bool GetSpeedVaid();
    float GetSpeed();
    bool GetSpaceEnough();
    void StartTimer(uint16_t mins);
private:
    UpdateCheck();
    UpdateCheck(const UpdateCheck &);
    UpdateCheck & operator = (const UpdateCheck &);
private:
    void UpdateTimeout();
    void McuUdsCallback(const McuUdsMsg& udsdate);

private:
    std::unique_ptr<chassis_info_t> chassis_data;
    std::unique_ptr<hozon::netaos::sm::StateClientZmq> state_client_;
    std::unique_ptr<TimerManager> time_mgr_;
    int time_fd_um_ = -1;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_CHECK_H