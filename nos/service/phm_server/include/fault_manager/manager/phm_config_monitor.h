/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: mcu fault service
 */

#pragma once

#include <string>

namespace hozon {
namespace netaos {
namespace phm_server {

class PhmCfgMonitor
{
public:
    static PhmCfgMonitor* getInstance();
    void Init();
    void DeInit();

    void powerModeCb(const std::string& clientname, const std::string& key, const std::uint8_t& value);
    void calibrateStatusCb(const std::string& clientname, const std::string& key, const std::uint8_t& value);
    void status85Cb(const std::string& clientname, const std::string& key, const std::uint8_t& value);
    void modeReqCb(const std::string& clientname, const std::string& key, const std::uint8_t& value);
    void modeReqTimeoutCb(void* data);

private:
    PhmCfgMonitor();
    ~PhmCfgMonitor();
    PhmCfgMonitor(const PhmCfgMonitor&);
    PhmCfgMonitor& operator=(const PhmCfgMonitor&);

    static std::mutex mtx_;
    static PhmCfgMonitor* instance_;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
