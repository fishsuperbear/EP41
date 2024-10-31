/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_security_mgr.h is designed for diagnostic security manager.
 */

#ifndef DIAG_SERVER_SECURITY_MGR_H
#define DIAG_SERVER_SECURITY_MGR_H

#include <mutex>
#include <set>
#include "diag/common/include/timer_manager.h"
#include "diag/diag_server/include/common/diag_server_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class DiagServerSecurityMgr {

public:
    static DiagServerSecurityMgr* getInstance();

    void Init();
    void DeInit();

    void SessionStatusChange(DiagServerSessionCode session);

    void AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage);

    void PositiveResponse(const DiagServerUdsMessage& udsMessage);
    void NegativeResponse(const DiagServerNrcErrc errorCode);

private:
    void SetCurrentLevel(const uint8_t level);
    uint8_t GetCurrentLevel();

    void DealwithSecurityAccessData(const DiagServerUdsMessage& udsMessage);

    void StartTimer();
    void StopTimer();
    void Timeout(void * data);

private:
    DiagServerSecurityMgr();
    DiagServerSecurityMgr(const DiagServerSecurityMgr &);
    DiagServerSecurityMgr & operator = (const DiagServerSecurityMgr &);

private:
    static std::mutex mtx_;
    static DiagServerSecurityMgr* instance_;
    uint8_t current_level;
    uint8_t step_; // 0:default 1: level1_seed 2:level1_key 5:levelfbl_seed 6:levelfbl_key
    uint32_t seed_;
    int access_err_count_;
    int time_fd_;
    std::unique_ptr<TimerManager> time_mgr_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_SERVER_SECURITY_MGR_H
