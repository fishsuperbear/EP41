/*
 * Copyright (c) Hozon Auto Co., Ltd. 2022-2022. All rights reserved.
 * Description: fault lock signal manager
 */

#ifndef PHM_FAULT_LOCK_SIGNAL_MANAGER_H
#define PHM_FAULT_LOCK_SIGNAL_MANAGER_H

#include <mutex>
#include <vector>
#include <unordered_map>

namespace hozon {
namespace netaos {
namespace phm_server {

class FaultLockSignalManager {

public:
    static FaultLockSignalManager* getInstance();
    void Init();
    void DeInit();

    void SetFaultSignalNum(const uint64_t key, const uint32_t faultStatus, const uint32_t lockedNumber = 0, const bool lockFlag = false);
    uint64_t GetFaultData(const bool lockFlag = false);

private:
    void LoadSignalValueMap();
    void LoadFaultSignalNumMap();
    void LoadFaultSignalMap();

private:
    FaultLockSignalManager();
    FaultLockSignalManager(const FaultLockSignalManager &);
    FaultLockSignalManager & operator = (const FaultLockSignalManager &);

private:
    static FaultLockSignalManager* m_ptrInstance;
    static std::mutex mtx_;

    // Unlocked signal value map  key: signalName    value: signalValue
    std::unordered_map<std::string, uint64_t> m_signalValueMap;
    // Blocked signal value map  key: signalName    value: signalBlockedValue
    std::unordered_map<std::string, uint64_t> m_signalBlockValueMap;
    // Locked signal value map  key: signalName    value: signalLockValue
    std::unordered_map<std::string, uint64_t> m_signalLockValueMap;
    // Lock fault signal map  key: faultId * 100 + faultObj    value: vector<signalName>
    std::unordered_map<uint64_t, std::vector<std::string>> m_faultSignalMap;
    // Runtime signal num map  key: signalName    value: signalNum
    std::unordered_map<std::string, uint32_t> m_runtimeSignalNumMap;
    // Blocked signal num map  key: signalName    value: signalblockNum
    std::unordered_map<std::string, uint32_t> m_blockSignalNumMap;
    // Locked signal num map  key: signalName    value: signalLockNum
    std::unordered_map<std::string, uint32_t> m_lockSignalNumMap;
};

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon
#endif  // PHM_FAULT_LOCK_SIGNAL_MANAGER_H
