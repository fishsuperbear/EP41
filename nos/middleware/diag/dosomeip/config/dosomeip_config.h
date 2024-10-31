/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip config
 */

#ifndef DOSOMEIP_CONFIG_H
#define DOSOMEIP_CONFIG_H

#include <stdint.h>
#include <mutex>
#include <algorithm>

namespace hozon {
namespace netaos {
namespace diag {

class DoSomeIPConfig {
public:
    static DoSomeIPConfig* Instance();
    bool LoadConfig();

    uint16_t GetInitMaxTimeout();
    uint16_t GetResponseMaxTimeout();
    std::vector<uint16_t> GetDoipAddressList();
    std::vector<uint16_t> GetDocanAddressList();
    uint16_t GetDoSomeIPProxyAddress();

    bool IsDoIPAddress(const uint16_t& address);
    bool IsDoCanAddress(const uint16_t& address);
    bool IsDoSomeipProxyAddress(const uint16_t& address);


private:
    DoSomeIPConfig();
    ~DoSomeIPConfig();
    DoSomeIPConfig(const DoSomeIPConfig&);
    DoSomeIPConfig& operator=(const DoSomeIPConfig&);
    char* GetJsonAll(const char* fname);
    bool ParseJSON(char* jsonstr);

    static DoSomeIPConfig* instancePtr_;
    static std::mutex instance_mtx_;

    uint16_t initTimeout_;
    uint16_t respTimeout_;
    std::vector<uint16_t> doipAddressList_;
    std::vector<uint16_t> docanAddressList_;
    uint16_t doSomeIPProxyAddress_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DOSOMEIP_CONFIG_H