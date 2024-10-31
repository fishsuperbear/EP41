/*
 * Copyright (c) Hozon Auto Co., Ltd. 2022-2022. All rights reserved.
 * Description: fault lock signal manager
 */

#include <regex>

#include "phm_server/include/common/phm_server_logger.h"
#include "phm_server/include/common/phm_server_config.h"
#include "phm_server/include/common/phm_server_persistency.h"
#include "phm_server/include/fault_lock/phm_fault_lock_signal_manager.h"

namespace hozon {
namespace netaos {
namespace phm_server {

using namespace std;

std::vector<std::string> SIGNAL_NAME = {
    "ADCS8_SystemFault", "ADCS8_FVC_30_Fail", "ADCS8_FVC_120_Fail", "ADCS8_SVC_RL_Fail",
    "ADCS8_SVC_FL_Fail", "ADCS8_SVC_RR_Fail", "ADCS8_SVC_FR_Fail", "ADCS8_RVC_Fail",
    "ADCS8_AVMFrontCAMFault", "ADCS8_AVMLeftCAMFault", "ADCS8_AVMRightCAMFault", "ADCS8_AVMRearCAMFault",
    "ADCS8_FRS_Fail", "ADCS8_SRR_FL_Fail", "ADCS8_SRR_FR_Fail", "ADCS8_SRR_RL_Fail",
    "ADCS8_SRR_RR_Fail", "ADCS8_LeftLidarFault", "ADCS8_RightlidarFault", "ADCS8_PA_FPAS_SensorFaultStsFRC",
    "ADCS8_PA_FPAS_SensorFaultStsFRM", "ADCS8_PA_FPAS_SensorFaultStsFLM", "ADCS8_PA_FPAS_SensorFaultStsFLC", "ADCS8_PA_RPAS_SensorFaultStsSRR",
    "ADCS8_PA_RPAS_SensorFaultStsSRL", "ADCS8_PA_FPAS_SensorFaultStsSFR", "ADCS8_PA_FPAS_SensorFaultStsSFL", "ADCS8_PA_RPAS_SensorFaultStsRRC",
    "ADCS8_PA_RPAS_SensorFaultStsRRM", "ADCS8_PA_RPAS_SensorFaultStsRLM", "ADCS8_PA_RPAS_SensorFaultStsRLC", "ADCS9_CDCS_CertificateState"
};

std::vector<uint64_t> SIGNAL_JUDGE_VALUE = {
    0x3, 0xC, 0x30, 0xC0,
    0x300, 0xC00, 0x3000, 0xC000,
    0x30000, 0xC0000, 0x300000, 0xC00000,
    0x3000000, 0xC000000, 0x30000000, 0xC0000000,
    0x300000000, 0xC00000000, 0x3000000000, 0xC000000000,
    0x30000000000, 0xC0000000000, 0x300000000000, 0xC00000000000,
    0x3000000000000, 0xC000000000000, 0x30000000000000, 0xC0000000000000,
    0x300000000000000, 0xC00000000000000, 0x3000000000000000, 0xC000000000000000
};

std::vector<uint64_t> SIGNAL_BLOCK_VALUE = {
    0x2, 0x4, 0x10, 0x40,
    0x100, 0x400, 0x1000, 0x4000,
    0x10000, 0x40000, 0x100000, 0x400000,
    0x1000000, 0x4000000, 0x10000000, 0x40000000,
    0x100000000, 0x400000000, 0x1000000000, 0,
    0, 0, 0, 0,
    0, 0, 0, 0,
    0, 0, 0, 0x4000000000000000
};

std::vector<uint64_t> SIGNAL_LOCK_VALUE_OTS = {
    0x2, 0x8, 0x20, 0x80,
    0x200, 0x800, 0x2000, 0x8000,
    0x20000, 0x80000, 0x200000, 0x800000,
    0x2000000, 0x8000000, 0x20000000, 0x80000000,
    0x200000000, 0x800000000, 0x2000000000, 0x4000000000,
    0x10000000000, 0x40000000000, 0x100000000000, 0x400000000000,
    0x1000000000000, 0x4000000000000, 0x10000000000000, 0x40000000000000,
    0x100000000000000, 0x400000000000000, 0x1000000000000000, 0
};

std::vector<uint64_t> SIGNAL_VALUE_OTA = {
    0x1, 0x8, 0x20, 0x80,
    0x200, 0x800, 0x2000, 0x8000,
    0x20000, 0x80000, 0x200000, 0x800000,
    0x2000000, 0x8000000, 0x20000000, 0x80000000,
    0x200000000, 0x800000000, 0x2000000000, 0x4000000000,
    0x10000000000, 0x40000000000, 0x100000000000, 0x400000000000,
    0x1000000000000, 0x4000000000000, 0x10000000000000, 0x40000000000000,
    0x100000000000000, 0x400000000000000, 0x1000000000000000, 0
};

std::vector<uint64_t> SIGNAL_LOCK_VALUE_OTA = {
    0x2, 0xC, 0x30, 0xC0,
    0x300, 0xC00, 0x3000, 0xC000,
    0x30000, 0xC0000, 0x300000, 0xC00000,
    0x3000000, 0xC000000, 0x30000000, 0xC0000000,
    0x300000000, 0xC00000000, 0x3000000000, 0x8000000000,
    0x20000000000, 0x80000000000, 0x200000000000, 0x800000000000,
    0x2000000000000, 0x8000000000000, 0x20000000000000, 0x80000000000000,
    0x200000000000000, 0x800000000000000, 0x2000000000000000, 0
};

// Lock fault record path.
const std::string FAULT_LOCK_LIST_FILE = "hz_fault_lock_list.json";
const std::string LOCK_FAULT_SIGNAL = "LockFaultSignal";

const std::string REGEX = "-";

std::mutex FaultLockSignalManager::mtx_;
FaultLockSignalManager* FaultLockSignalManager::m_ptrInstance = nullptr;

FaultLockSignalManager::FaultLockSignalManager()
{
}

FaultLockSignalManager*
FaultLockSignalManager::getInstance()
{
    if (nullptr == m_ptrInstance) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == m_ptrInstance) {
            m_ptrInstance = new FaultLockSignalManager();
        }
    }

    return m_ptrInstance;
}

void
FaultLockSignalManager::Init()
{
    PHMS_INFO << "FaultLockSignalManager::Init.";
    LoadSignalValueMap();
    LoadFaultSignalNumMap();
    LoadFaultSignalMap();
}

void
FaultLockSignalManager::DeInit()
{
    PHMS_INFO << "FaultLockSignalManager::DeInit.";
    m_signalValueMap.clear();
    m_signalBlockValueMap.clear();
    m_signalLockValueMap.clear();
    m_faultSignalMap.clear();
    m_runtimeSignalNumMap.clear();
    m_blockSignalNumMap.clear();
    m_lockSignalNumMap.clear();

    if (nullptr != m_ptrInstance) {
        delete m_ptrInstance;
        m_ptrInstance = nullptr;
    }
}

void
FaultLockSignalManager::SetFaultSignalNum(const uint64_t key, const uint32_t faultStatus, const uint32_t lockedNumber, const bool lockFlag)
{
    PHMS_INFO << "FaultLockSignalManager::SetFaultSignalNum key: " << key << ", faultStatus: " << faultStatus << ", lockedNumber: " << lockedNumber << ", lockFlag: " << lockFlag << ".";
    std::lock_guard<std::mutex> lck(mtx_);
    auto itrFaultSignal = m_faultSignalMap.find(key);
    if (itrFaultSignal == m_faultSignalMap.end()) {
        return;
    }

    bool isBlockedFault = PHMServerConfig::getInstance()->IsBlockedFault(key);
    for (auto& item : itrFaultSignal->second) {
        if (faultStatus) {
            if (lockFlag) {
                m_lockSignalNumMap[item]++;
            }
            else {
                if (isBlockedFault) {
                    m_blockSignalNumMap[item]++;
                }
                else {
                    m_runtimeSignalNumMap[item]++;
                }
            }
        }
        else {
            if (lockFlag) {
                if (m_lockSignalNumMap[item]) {
                    if ((lockedNumber > 0) && (m_lockSignalNumMap[item] >= lockedNumber)) {
                        m_lockSignalNumMap[item] -= lockedNumber;
                    }
                }
            }
            else {
                if (isBlockedFault) {
                    if (m_blockSignalNumMap[item]) {
                        m_blockSignalNumMap[item]--;
                    }
                }
                else {
                    if (m_runtimeSignalNumMap[item]) {
                        m_runtimeSignalNumMap[item]--;
                    }
                }
            }
        }
    }

    if (lockFlag) {
        std::string faultSignal = "";
        for (uint i = 0; i < SIGNAL_NAME.size();) {
            faultSignal += std::to_string(m_lockSignalNumMap[SIGNAL_NAME[i]]);
            ++i;
            if (i < SIGNAL_NAME.size()) {
                faultSignal += REGEX;
            }
        }

        if (!(PHMServerPersistency::getInstance()->SetValue(FAULT_LOCK_LIST_FILE, LOCK_FAULT_SIGNAL, faultSignal))) {
            PHMS_WARN << "FaultLockSignalManager::SetFaultSignalNum SetValue failed.";
        }
    }
}

uint64_t
FaultLockSignalManager::GetFaultData(const bool lockFlag)
{
    PHMS_DEBUG << "FaultLockSignalManager::GetFaultData lockFlag: " << lockFlag << ".";
    uint64_t faultData = 0;
    if (lockFlag) {
        for (auto& item : SIGNAL_NAME) {
            if (m_lockSignalNumMap[item]) {
                faultData |= m_signalLockValueMap[item];
            }
        }

        if (0 == m_lockSignalNumMap["ADCS9_CDCS_CertificateState"]) {
            faultData |= NORMAL_FAULT_LOCK_VALUE;
        }
    }else {
        for (auto& item : SIGNAL_NAME) {
            if (m_lockSignalNumMap[item]) {
                faultData |= m_signalLockValueMap[item];
            }
            else {
                if (PHMServerConfig::getInstance()->GetLockFaultCurrentVersion() == "ots") {
                    if (m_blockSignalNumMap[item]) {
                        faultData |= m_signalBlockValueMap[item];
                    }
                    else {
                        if (m_runtimeSignalNumMap[item]) {
                            faultData |= m_signalValueMap[item];
                        }
                    }
                }
                else if (PHMServerConfig::getInstance()->GetLockFaultCurrentVersion() == "ota") {
                    if (m_runtimeSignalNumMap[item]) {
                        faultData |= m_signalValueMap[item];
                    }
                    else {
                        if (m_blockSignalNumMap[item]) {
                            faultData |= m_signalBlockValueMap[item];
                        }
                    }
                }
            }
        }

        if ((0 == m_lockSignalNumMap["ADCS9_CDCS_CertificateState"]) && (0 == m_runtimeSignalNumMap["ADCS9_CDCS_CertificateState"])) {
            faultData |= NORMAL_FAULT_LOCK_VALUE;
        }
    }

    return faultData;
}

static
vector<std::string> Split(const std::string& inputStr, const std::string& regexStr = REGEX)
{
    std::regex re(regexStr);
    std::sregex_token_iterator first {inputStr.begin(), inputStr.end(), re, -1}, last;
    return {first, last};
}

void
FaultLockSignalManager::LoadSignalValueMap()
{
    PHMS_INFO << "FaultLockSignalManager::LoadSignalValueMap.";
    if (PHMServerConfig::getInstance()->GetLockFaultCurrentVersion() == "ots") {
        for (uint i = 0; i < SIGNAL_NAME.size(); i++) {
            m_signalValueMap.insert(std::make_pair(SIGNAL_NAME[i], 0));
            m_signalBlockValueMap.insert(std::make_pair(SIGNAL_NAME[i], SIGNAL_BLOCK_VALUE[i]));
            m_signalLockValueMap.insert(std::make_pair(SIGNAL_NAME[i], SIGNAL_LOCK_VALUE_OTS[i]));
        }

        m_signalValueMap[SIGNAL_NAME[0]] = 0x1;
    }
    else if (PHMServerConfig::getInstance()->GetLockFaultCurrentVersion() == "ota") {
        for (uint i = 0; i < SIGNAL_NAME.size(); i++) {
            m_signalValueMap.insert(std::make_pair(SIGNAL_NAME[i], SIGNAL_VALUE_OTA[i]));
            m_signalBlockValueMap.insert(std::make_pair(SIGNAL_NAME[i], SIGNAL_BLOCK_VALUE[i]));
            m_signalLockValueMap.insert(std::make_pair(SIGNAL_NAME[i], SIGNAL_LOCK_VALUE_OTA[i]));
        }
    }
    else {
        PHMS_ERROR << "FaultLockSignalManager::LoadSignalValueMap lock fault version error.";
    }
}

void
FaultLockSignalManager::LoadFaultSignalNumMap()
{
    PHMS_INFO << "FaultLockSignalManager::LoadFaultSignalNumMap.";
    std::string faultSignal = "0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0";
    PHMServerPersistency::getInstance()->GetValue(FAULT_LOCK_LIST_FILE, LOCK_FAULT_SIGNAL, faultSignal);
    auto vecSignal = Split(faultSignal);
    if(vecSignal.size() != SIGNAL_NAME.size()) {
        PHMS_ERROR << "FaultLockSignalManager::LoadFaultSignalNumMap faultSignal data error.";
        return;
    }

    for (uint i = 0; i < SIGNAL_NAME.size(); i++) {
        m_runtimeSignalNumMap.insert(std::make_pair(SIGNAL_NAME[i], 0));
        m_blockSignalNumMap.insert(std::make_pair(SIGNAL_NAME[i], 0));
        m_lockSignalNumMap.insert(std::make_pair(SIGNAL_NAME[i], std::strtoul(vecSignal[i].c_str(), 0, 10)));
    }

    if (!(PHMServerPersistency::getInstance()->SetValue(FAULT_LOCK_LIST_FILE, LOCK_FAULT_SIGNAL, faultSignal))) {
        PHMS_WARN << "FaultLockSignalManager::LoadFaultSignalNumMap SetValue failed.";
    }
}

void
FaultLockSignalManager::LoadFaultSignalMap()
{
    PHMS_INFO << "FaultLockSignalManager::LoadFaultSignalMap.";
    std::vector<FaultLockInfo> faultLockInfos;
    PHMServerConfig::getInstance()->GetLockFaultInfos(faultLockInfos);
    if(faultLockInfos.size() <= 0) {
        PHMS_ERROR << "FaultLockSignalManager::LoadFaultSignalMap faultLockInfos data error.";
        return;
    }

    uint32_t faultId = 0;
    uint32_t faultObj = 0;
    uint64_t faultData = 0;
    uint64_t key = 0;
    std::vector<std::string> signal;
    for (auto& faultLockInfo : faultLockInfos) {
        signal.clear();
        faultId = faultLockInfo.faultId;
        faultObj = faultLockInfo.faultObj;
        faultData = std::strtoul(faultLockInfo.lockFaultToHMIData.c_str(), 0, 16);
        uint64_t judgeValue = 0;
        for (uint i = 0; i < SIGNAL_JUDGE_VALUE.size(); i++) {
            judgeValue = faultData & (SIGNAL_JUDGE_VALUE[i]);
            if (judgeValue) {
                signal.push_back(SIGNAL_NAME[i]);
            }
        }

        key = faultId * 100 + faultObj;
        m_faultSignalMap.insert(std::make_pair(key, signal));
    }
}

}  // namespace phm_server
}  // namespace netaos
}  // namespace hozon