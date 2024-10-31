/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: persistency
 */

#ifndef PHM_SERVER_PERSISTENCY_H
#define PHM_SERVER_PERSISTENCY_H

#include <fstream>
#include <vector>

#include "per/include/key_value_storage.h"

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace hozon::netaos::per;
const std::string FM_PER_PATH = "/opt/usr/col/fm";

class PHMServerPersistency {

public:
    static PHMServerPersistency* getInstance();
    void Init();
    void DeInit();

    // Key-Value PER Operation
    void GetValue(const std::string& keyValueFile, const std::string& key, uint64_t& value);
    bool SetValue(const std::string& keyValueFile, const std::string& key, const uint64_t value);
    void GetValue(const std::string& keyValueFile, const std::string& key, std::string& value);
    bool SetValue(const std::string& keyValueFile, const std::string& key, const std::string& value);
    bool GetAllKeys(const std::string& keyValueFile, std::vector<std::string>& vecKey);
    bool RemoveKey(const std::string& keyValueFile, const std::string& key);
    bool IsHasKey(const std::string& keyValueFile, const std::string& key);
    bool SyncKeyValueFile(const std::string& keyValueFile);

    // Std File Opeartion
    bool WriteFile(const std::string& file, const std::string& data, const std::ios::openmode mode = std::ios::app);
    bool CopyFile(const std::string& from, const std::string& to);

private:
    // key-value
    void CreateKeyValueFile();

private:
    PHMServerPersistency();
    PHMServerPersistency(const PHMServerPersistency &);
    PHMServerPersistency& operator = (const PHMServerPersistency &);

private:
    static PHMServerPersistency* m_ptrInstance;
    static std::mutex mtx_;

    SharedHandle<KeyValueStorage> m_ptrFaultLock;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_SERVER_PERSISTENCY_H