/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: persistency
 */

#include <sys/stat.h>

#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_persistency.h"

// key-value
const std::string FAULT_LOCK_LIST_FILE = "hz_fault_lock_list.json";
const std::string DEFAULT_FAULT_LOCK_LIST_PATH = "/app/runtime_service/phm_server/conf/phm_fault_lock_list.json";

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace std;

PHMServerPersistency* PHMServerPersistency::m_ptrInstance = nullptr;
std::mutex PHMServerPersistency::mtx_;

PHMServerPersistency::PHMServerPersistency()
: m_ptrFaultLock(nullptr)
{
}

PHMServerPersistency*
PHMServerPersistency::getInstance()
{
    if (nullptr == m_ptrInstance) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == m_ptrInstance) {
            m_ptrInstance = new PHMServerPersistency();
        }
    }

    return m_ptrInstance;
}

void
PHMServerPersistency::Init()
{
    PHMS_INFO << "PHMServerPersistency::Init enter!";
    CreateKeyValueFile();
}

void
PHMServerPersistency::DeInit()
{
    PHMS_INFO << "PHMServerPersistency::DeInit enter!";
    if (m_ptrInstance != nullptr) {
        delete m_ptrInstance;
        m_ptrInstance = nullptr;
    }
}

// key-value
void
PHMServerPersistency::CreateKeyValueFile()
{
    PHMS_INFO << "PHMServerPersistency::CreateKeyValueFile enter!";
    // create fault lock file
    std::string faultLockPath = FM_PER_PATH + "/" + FAULT_LOCK_LIST_FILE;
    if (0 != access(faultLockPath.c_str(), 0)) {
        std::ofstream ofs(faultLockPath);
        ofs.close();
        CopyFile(DEFAULT_FAULT_LOCK_LIST_PATH, faultLockPath);
    }

    if (m_ptrFaultLock) {
        PHMS_WARN << "PHMServerPersistency::CreateKeyValueFile m_ptrFaultLock already create";
        return;
    }

    // Open key-value storage
    StorageConfig config;
    config.redundancy_config.redundant_count = 0;
    config.redundancy_config.auto_recover = false;
    config.redundancy_config.redundant_dirpath = FM_PER_PATH;
    config.serialize_format = "json";
    auto result = OpenKeyValueStorage(faultLockPath, config);
    if (!result.HasValue()) {
        PHMS_ERROR << "PHMServerPersistency::CreateKeyValueFile OpenKeyValueStorage failed due to: " << result.Error().Value();
        return;
    }

    m_ptrFaultLock = std::move(result).Value();
}

// key-value
void
PHMServerPersistency::GetValue(const std::string& keyValueFile, const std::string& key, uint64_t& value) {
    PHMS_DEBUG << "PHMServerPersistency::GetValue keyValueFile: " << keyValueFile << ", key: " << key;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "FaultLock::GetValue m_ptrFaultLock == nullptr";
            return;
        }

        if (!IsHasKey(keyValueFile, key)) {
            return;
        }

        m_ptrFaultLock->GetValue(key, value);
    }
}

bool
PHMServerPersistency::SetValue(const std::string& keyValueFile, const std::string& key, const uint64_t value) {
    PHMS_DEBUG << "PHMServerPersistency::SetValue keyValueFile: " << keyValueFile << ", key: " << key << ", value: " << value;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "PHMServerPersistency::SetValue m_ptrFaultLock == nullptr";
            return false;
        }

        if (!m_ptrFaultLock->SetValue(key, value)) {
            PHMS_ERROR << "PHMServerPersistency::SetValue failed.";
            return false;
        }
    }

    return true;
}

void
PHMServerPersistency::GetValue(const std::string& keyValueFile, const std::string& key, std::string& value) {
    PHMS_DEBUG << "PHMServerPersistency::GetValue keyValueFile: " << keyValueFile << ", key: " << key;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "FaultLock::GetValue m_ptrFaultLock == nullptr";
            return;
        }

        if (!IsHasKey(keyValueFile, key)) {
            return;
        }

        m_ptrFaultLock->GetValue(key, value);
    }
}

bool
PHMServerPersistency::SetValue(const std::string& keyValueFile, const std::string& key, const std::string& value) {
    PHMS_DEBUG << "PHMServerPersistency::SetValue keyValueFile: " << keyValueFile << ", key: " << key << ", value: " << value;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "PHMServerPersistency::SetValue m_ptrFaultLock == nullptr";
            return false;
        }

        if (!m_ptrFaultLock->SetValue(key, value)) {
            PHMS_ERROR << "PHMServerPersistency::SetValue failed.";
            return false;
        }
    }

    return true;
}

bool
PHMServerPersistency::GetAllKeys(const std::string& keyValueFile,std::vector<std::string>& vecKey) {
    PHMS_INFO << "PHMServerPersistency::GetAllKeys keyValueFile: " << keyValueFile;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "PHMServerPersistency::GetAllKeys m_ptrFaultLock == nullptr";
            return false;
        }

        auto vec = m_ptrFaultLock->GetAllKeys().Value();
        if (vec.size() <= 0) {
            PHMS_ERROR << "PHMServerPersistency::GetAllKeys get failed.";
            return false;
        }

        vecKey.assign(vec.begin(), vec.end());
    }

    return true;
}

bool
PHMServerPersistency::RemoveKey(const std::string& keyValueFile, const std::string& key) {
    PHMS_DEBUG << "PHMServerPersistency::RemoveKey keyValueFile: " << keyValueFile << ", key: " << key;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "PHMServerPersistency::RemoveKey m_ptrFaultLock == nullptr";
            return false;
        }

        // Remove key.
        auto removeResult = m_ptrFaultLock->RemoveKey(key);
        if (!removeResult.HasValue()) {
            PHMS_WARN << "PHMServerPersistency::RemoveKey failed due to: " << removeResult.Error().Value();
            return false;
        }
    }

    return true;
}

bool
PHMServerPersistency::IsHasKey(const std::string& keyValueFile, const std::string& key) {
    PHMS_DEBUG << "PHMServerPersistency::IsHasKey keyValueFile: " << keyValueFile << ", key: " << key;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "PHMServerPersistency::IsHasKey m_ptrFaultLock == nullptr";
            return false;
        }

        if (!m_ptrFaultLock->HasKey(key).Value()) {
            return false;
        }
    }

    return true;
}

bool
PHMServerPersistency::SyncKeyValueFile(const std::string& keyValueFile) {
    PHMS_DEBUG << "PHMServerPersistency::SyncKeyValueFile enter! keyValueFile: " << keyValueFile;
    if (FAULT_LOCK_LIST_FILE == keyValueFile) {
        if (!m_ptrFaultLock) {
            PHMS_ERROR << "PHMServerPersistency::SyncKeyValueFile m_ptrFaultLock == nullptr";
            return false;
        }

        auto syncResult = m_ptrFaultLock->SyncToStorage();
        if (!syncResult.HasValue()) {
            PHMS_ERROR << "PHMServerPersistency::SyncKeyValueFile SyncToStorage failed due to: " << syncResult.Error().Value();
            return false;
        }
    }

    return true;
}

bool
PHMServerPersistency::WriteFile(const std::string& file, const std::string& data, const std::ios::openmode mode)
{
    // PHMS_INFO << "PHMServerPersistency::WriteFile enter! file: " << file;
    if (access(FM_PER_PATH.c_str(), 0) != 0) {
        PHMS_ERROR << "PHMServerPersistency::WriteFile Dir " << FM_PER_PATH << " not exists";
        return false;
    }

    std::string path = FM_PER_PATH + "/" + file;
    std::ofstream ostrm(path, mode);

    ostrm.write(data.c_str(), data.size());
    ostrm.close();
    PHMS_DEBUG << "PHMServerPersistency::WriteFile content " << data << " size " << data.size();
    return true;
}

bool
PHMServerPersistency::CopyFile(const std::string& from, const std::string& to)
{
    PHMS_INFO << "PHMServerPersistency::CopyFile from " << from << " to " << to;
    std::ifstream input(from, ios::binary);
    if (!input.good()) {
        PHMS_WARN << "PHMServerPersistency::CopyFile file " << from << "open failed!";
        return false;
    }

    std::ofstream output(to, ios::binary);
    if (!output.good()) {
        PHMS_WARN << "PHMServerPersistency::CopyFile file " << to << "open failed!";
        input.close();
        return false;
    }

    output << input.rdbuf();
    input.close();
    output.close();
    return true;
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon