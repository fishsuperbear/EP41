/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip config
 */

#ifndef SOMEIP_CONFIG_H
#define SOMEIP_CONFIG_H

#include <stdint.h>
#include <mutex>
#include <algorithm>
#include <map>


class SomeIPConfig {
public:
    static SomeIPConfig* Instance();
    bool LoadConfig();

    void printUdsReqs() const;
    std::vector<uint8_t> getUdsReqById(uint16_t id) const;
    void setRespResultById(uint16_t id, bool success);
    uint16_t getSourceAdr();
    uint16_t getTargetAdr();

private:
    SomeIPConfig();
    ~SomeIPConfig();
    SomeIPConfig(const SomeIPConfig&);
    SomeIPConfig& operator=(const SomeIPConfig&);
    char* GetJsonAll(const char* fname);
    bool ParseJSON(char* jsonstr);

    static SomeIPConfig* instancePtr_;
    static std::mutex instance_mtx_;
    std::uint16_t source_adr_;
    std::uint16_t target_adr_;
    std::map<uint16_t, std::vector<uint8_t>> udsReqs_;
};

#endif  // SOMEIP_CONFIG_H