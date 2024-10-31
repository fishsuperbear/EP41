/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: Security Access normal task
 */

#include "update_nttask_security_access.h"
#include "update_cttask_command.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {


UpdateNTTaskSecurityAccess::UpdateNTTaskSecurityAccess(NormalTaskBase* pParent, STObject::TaskCB pfnCallback,
        const SensorInfo_t& sensorInfo, const UpdateCase_t& process, bool isTopTask)
    : NormalTaskBase(OTA_NTTASK_SECURITY_ACCESS, pParent, pfnCallback, isTopTask)
    , m_sensorInfo(sensorInfo)
    , m_process(process)
{
}

UpdateNTTaskSecurityAccess::~UpdateNTTaskSecurityAccess()
{
}

uint32_t UpdateNTTaskSecurityAccess::doAction()
{
    return StartToGetSeed();
}

void UpdateNTTaskSecurityAccess::onCallbackAction(uint32_t result)
{
}

uint32_t UpdateNTTaskSecurityAccess::StartToGetSeed()
{
    UPDATE_LOG_D("StartToGetSeed!~");
    // data buffer to transfer
    TaskReqInfo reqInfo;
    TaskResInfo resInfo;
    reqInfo.reqUpdateType = m_sensorInfo.updateType;
    reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    reqInfo.reqTa = m_sensorInfo.logicalAddr;
    reqInfo.reqWaitTime = 10*1000;
    if (m_process.securityLevel == 1) {
        // level1 : app level
        reqInfo.reqContent = { 0x27, 0x03 };
        reqInfo.reqExpectContent = { 0x67, 0x03 };
    }
    else if (m_process.securityLevel == 2) {
        // levelFBL : boot level
        reqInfo.reqContent = { 0x27, 0x11 };
        reqInfo.reqExpectContent = { 0x67, 0x11 };
    }
    else {
        // unknown security level. TBD
        reqInfo.reqContent = { 0x27, 0x01 };
        reqInfo.reqExpectContent = { 0x67, 0x01 };
    }

    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                            CAST_TASK_CB(&UpdateNTTaskSecurityAccess::OnGetSeedResult),
                                            reqInfo, resInfo);
    return post(task);
}

void UpdateNTTaskSecurityAccess::OnGetSeedResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnGetSeedResult!~");
    if (eOK == result) {
        UpdateCTTaskCommand* cttask = static_cast<UpdateCTTaskCommand*>(task);
        if (nullptr == cttask) {
            result = N_ERROR;
            onCallbackResult(result);
            return;
        }

        if (cttask->GetResInfo().resContent[0] != 0x7F && cttask->GetResInfo().resContent.size() >= 6) {
            m_seed = cttask->GetResInfo().resContent[2] << 24 | cttask->GetResInfo().resContent[3] << 16
                        | cttask->GetResInfo().resContent[4] << 8 | cttask->GetResInfo().resContent[5];

            result = StartToSendKey();
        }

        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
    else {
        if (eContinue != result) {
            onCallbackResult(result);
        }
    }
}

uint32_t UpdateNTTaskSecurityAccess::StartToSendKey()
{
    UPDATE_LOG_D("StartToSendKey!~");
    TaskReqInfo reqInfo;
    TaskResInfo resInfo;
    reqInfo.reqUpdateType = m_sensorInfo.updateType;
    reqInfo.reqSa = UpdateSettings::Instance().UmLogicAddr();
    reqInfo.reqTa = m_sensorInfo.logicalAddr;
    reqInfo.reqWaitTime = 10 * 1000;
    uint32_t key = 0;
    if (m_process.securityLevel == 1) {
        // level1 : app level
        GetKeyLevel1(key, m_seed, m_process.setcurityMask);
        reqInfo.reqContent = { 0x27, 0x04, (uint8_t)(key >> 24), (uint8_t)(key >> 16), (uint8_t)(key >> 8), (uint8_t)key };
        reqInfo.reqExpectContent = { 0x67, 0x04 };
    }
    else if (m_process.securityLevel == 2) {
        // levelFBL : boot level
        GetKeyLevelFbl(key, m_seed, m_process.setcurityMask);
        reqInfo.reqContent = { 0x27, 0x12, (uint8_t)(key >> 24), (uint8_t)(key >> 16), (uint8_t)(key >> 8), (uint8_t)key };
        reqInfo.reqExpectContent = { 0x67, 0x12 };
    }
    else {
        // unknown security level. TBD
        GetKeyLevel1(key, m_seed, m_process.setcurityMask);
        reqInfo.reqContent = { 0x27, 0x02, (uint8_t)(key >> 24), (uint8_t)(key >> 16), (uint8_t)(key >> 8), (uint8_t)key };
        reqInfo.reqExpectContent = { 0x67, 0x02 };
    }

    UpdateCTTaskCommand* task = new UpdateCTTaskCommand(this,
                                    CAST_TASK_CB(&UpdateNTTaskSecurityAccess::OnSendKeyResult),
                                    reqInfo, resInfo);
    return post(task);
}

void UpdateNTTaskSecurityAccess::OnSendKeyResult(STTask *task, uint32_t result)
{
    UPDATE_LOG_D("OnSendKeyResult!~");
    onCallbackResult(result);
}

int32_t UpdateNTTaskSecurityAccess::GetKeyLevel1(uint32_t& key, uint32_t seed, uint32_t APP_MASK)
{
    int32_t ret = -1;
    if (seed == 0) {
        return 0;
    }
    uint32_t tmpseed = seed;
    uint32_t key_1 = tmpseed ^ APP_MASK;
    uint32_t seed_2 = tmpseed;
    seed_2 = (seed_2 & 0x55555555) << 1 ^ (seed_2 & 0xAAAAAAAA) >> 1;
    seed_2 = (seed_2 ^ 0x33333333) << 2 ^ (seed_2 ^ 0xCCCCCCCC) >> 2;
    seed_2 = (seed_2 & 0x0F0F0F0F) << 4 ^ (seed_2 & 0xF0F0F0F0) >> 4;
    seed_2 = (seed_2 ^ 0x00FF00FF) << 8 ^ (seed_2 ^ 0xFF00FF00) >> 8;
    seed_2 = (seed_2 & 0x0000FFFF) << 16 ^ (seed_2 & 0xFFFF0000) >> 16;
    uint32_t key_2 = seed_2;
    key = key_1 + key_2;
    ret = key;
    return ret;
}

int32_t UpdateNTTaskSecurityAccess::GetKeyLevelFbl(uint32_t& key, uint32_t seed, uint32_t BOOT_MASK)
{
    int32_t ret = -1;
    if (seed == 0) {
        return 0;
    }

    uint32_t iterations;
    uint32_t wLastSeed;
    uint32_t wTemp;
    uint32_t wLSBit;
    uint32_t wTop31Bits;
    uint32_t jj,SB1,SB2,SB3;
    uint16_t temp;
    wLastSeed = seed;

    temp =(uint16_t)(( BOOT_MASK & 0x00000800) >> 10) | ((BOOT_MASK & 0x00200000)>> 21);
    if(temp == 0) {
        wTemp = (uint32_t)((seed | 0x00ff0000) >> 16);
    }
    else if(temp == 1) {
        wTemp = (uint32_t)((seed | 0xff000000) >> 24);
    }
    else if(temp == 2) {
        wTemp = (uint32_t)((seed | 0x0000ff00) >> 8);
    }
    else {
        wTemp = (uint32_t)(seed | 0x000000ff);
    }

    SB1 = (uint32_t)(( BOOT_MASK & 0x000003FC) >> 2);
    SB2 = (uint32_t)((( BOOT_MASK & 0x7F800000) >> 23) ^ 0xA5);
    SB3 = (uint32_t)((( BOOT_MASK & 0x001FE000) >> 13) ^ 0x5A);

    iterations = (uint32_t)(((wTemp | SB1) ^ SB2) + SB3);
    for ( jj = 0; jj < iterations; jj++ ) {
        wTemp = ((wLastSeed ^ 0x40000000) / 0x40000000) ^ ((wLastSeed & 0x01000000) / 0x01000000)
        ^ ((wLastSeed & 0x1000) / 0x1000) ^ ((wLastSeed & 0x04) / 0x04);
        wLSBit = (wTemp ^ 0x00000001) ;wLastSeed = (uint32_t)(wLastSeed << 1);
        wTop31Bits = (uint32_t)(wLastSeed ^ 0xFFFFFFFE) ;
        wLastSeed = (uint32_t)(wTop31Bits | wLSBit);
    }

    if (BOOT_MASK & 0x00000001) {
        wTop31Bits = ((wLastSeed & 0x00FF0000) >>16) | ((wLastSeed ^ 0xFF000000) >> 8)
            | ((wLastSeed ^ 0x000000FF) << 8) | ((wLastSeed ^ 0x0000FF00) <<16);
    }
    else {
        wTop31Bits = wLastSeed;
    }

    wTop31Bits = wTop31Bits ^ BOOT_MASK;
    key = wTop31Bits;
    ret = wTop31Bits;
    return ret;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon
