/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: gdb infomation
 */

#pragma once

#include <string>
#include <memory>


namespace hozon {
namespace netaos {
namespace phm_server {

class ThreadPool;
class TimerManager;
class PhmHealthGdbMonitor
{
public:
    PhmHealthGdbMonitor();
    ~PhmHealthGdbMonitor();

    void StartInhibitCollectTimer();
    void CheckGdbDir();
    void DoCheckGdbDir();
    void CheckGdbDirSize();
    void CheckGdbInvalidFile();
    void StartCollectGdbInfo(const std::string& processName);
    int DoCollectGdbInfo(const std::string& processName);

private:
    std::shared_ptr<ThreadPool> m_spThreadPool;
    std::shared_ptr<TimerManager> m_spTimerManager;
};


}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
